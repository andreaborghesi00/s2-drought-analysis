import Clusterer
import S2Extractor
import Utils
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 18})
plt.rcParams["font.weight"] = "bold"


if __name__ == "__main__":

    data = {
        0:{
            "base_dir": "case/palermo/april",
            "location": "Riserva naturale Pizzo Cane, Pizzo Trigna, Grotta Mazzamuto",
            "aoi_path": "coordinates_case_palermo.geojson",
            "extra_text": "april"
        },
        1:{
            "base_dir": "case/palermo/july",
            "location": "Riserva naturale Pizzo Cane, Pizzo Trigna, Grotta Mazzamuto",
            "aoi_path": "coordinates_case_palermo.geojson",
            "extra_text": "july"
        },
        2:{
        "base_dir": "control/tevere/april",
            "location": "Parco Fluviale del Tevere",
            "aoi_path": "coordinates_control_tevere.geojson",
            "extra_text": "april"  
        },
        3:{
            "base_dir": "control/tevere/july",
            "location": "Parco Fluviale del Tevere",
            "aoi_path": "coordinates_control_tevere.geojson",
            "extra_text": "july"
        }
    }

    # initial configuration    
    years = np.arange(2017, 2025) # 2017 - 2024
    n_clusters = 6
    cmap = plt.cm.get_cmap("viridis", n_clusters+1)
    bands = ("04", "8A", "11", "12", "03")

    # main loop
    for base_dir_idx in range(len(data)):
        combined_bands_years = None
        combined_water_ndwi_mask = None
        
        print(f"Processing {data[base_dir_idx]['location']} - {data[base_dir_idx]['base_dir']}")
        for i, year in tqdm(enumerate(years)):
            base_dir = os.path.join("data", data[base_dir_idx]["base_dir"])
            aoi_path = data[base_dir_idx]["aoi_path"]

            band_paths = Utils.find_sentinel2_bands(year, bands, base_dir=base_dir)

            if band_paths is None:
                continue

            extractor = S2Extractor.Extractor()
            extractor.set_aoi(aoi_path)

            # extract bands and apply mask
            red_raster_04 = extractor.extract_band_data(band_paths[f"B{bands[0]}"])
            nir_raster_8a = extractor.extract_band_data(band_paths[f"B{bands[1]}"])
            swir_raster_11 = extractor.extract_band_data(band_paths[f"B{bands[2]}"])
            swir_raster_12 = extractor.extract_band_data(band_paths[f"B{bands[3]}"])
            green_raster_03 = extractor.extract_band_data(band_paths[f"B{bands[4]}"])
            
            # crop masked area
            red_raster_04 = extractor.crop_raster(red_raster_04)
            nir_raster_8a = extractor.crop_raster(nir_raster_8a)
            swir_raster_11 = extractor.crop_raster(swir_raster_11)
            swir_raster_12 = extractor.crop_raster(swir_raster_12)
            green_raster_03 = extractor.crop_raster(green_raster_03)

            # spectral indices computation
            # NDWI (Gao)
            ndwi_raster = (nir_raster_8a - swir_raster_11) / (nir_raster_8a + swir_raster_11)
            
            # NDVI
            ndvi_raster = (nir_raster_8a - red_raster_04) / (nir_raster_8a + red_raster_04)
            
            # MSI
            msi_raster = swir_raster_11 / nir_raster_8a
            
            # NDMI
            ndmi_raster = (nir_raster_8a - swir_raster_12) / (nir_raster_8a + swir_raster_12)

            # MSAVI2
            msavi2_raster = (0.5 * (2 * (nir_raster_8a + 1) - np.sqrt((2 * nir_raster_8a + 1) ** 2 - 8 * (nir_raster_8a - red_raster_04))))

            # NDWI (McFeeters) (water body detection)
            ndwi_mcfeeters_raster = (green_raster_03 - nir_raster_8a) / (green_raster_03 + nir_raster_8a) 

            # urban mask
            urban_msavi2_mask = np.zeros_like(msavi2_raster)
            urban_msavi2_mask[msavi2_raster < 0.25] = 1

            # water mask
            water_ndwi_mask = np.zeros_like(ndwi_mcfeeters_raster)
            water_ndwi_mask[ndwi_mcfeeters_raster > 0] = 1
            
            # combine bands depth-wise and apply mask
            combined_bands = Utils.combine_bands((ndwi_raster, msi_raster, ndmi_raster, msavi2_raster), urban_mask=urban_msavi2_mask, water_mask=water_ndwi_mask)
            # set nan values to 0
            combined_bands[np.isnan(combined_bands)] = 0
            
            # append yearly data
            combined_bands_years = combined_bands[np.newaxis, ...] if i == 0 else np.concatenate((combined_bands_years, combined_bands[np.newaxis, ...]), axis=0)
            combined_water_ndwi_mask = water_ndwi_mask[np.newaxis, ...] if i == 0 else np.concatenate((combined_water_ndwi_mask, water_ndwi_mask[np.newaxis, ...]), axis=0)

        clusterer = Clusterer.Clusterer()
        clusterer.algorithm = "kmeans"

        ## elbow method
        # optimal_k = clusterer.elbow_method(combined_bands_years.reshape(-1, combined_bands_years[0].shape[2]), 10, save_path=os.path.join("output", data[base_dir_idx]["base_dir"], f"elbow_method.png"))
        
        # kmeans clustering
        labels, _ = clusterer.fit_predict(combined_bands_years.reshape(-1, combined_bands_years[0].shape[2]), n_clusters=n_clusters)
        
        # sort labels
        sorted_labels = clusterer.sort_labels_by_ndwi_ndmi(combined_bands_years, labels)
        sorted_labels_raster = clusterer.labels_to_raster(
            sorted_labels,
            (
                combined_bands_years.shape[0],
                combined_bands_years.shape[1],
                combined_bands_years.shape[2],
            ),
        )
        
        # apply water mask
        for year in range(sorted_labels_raster.shape[0]):
            sorted_labels_raster[year][combined_water_ndwi_mask[year] == 1] = n_clusters # special label for water, the highest cluster index

        # plot rastered labels and histograms
        cluster_distribution_years = np.zeros((n_clusters+1, len(years)))
        for i, year in tqdm(enumerate(years)):
            curr_labels = sorted_labels_raster[i]
            cluster_distribution = clusterer.cluster_distribution(labels=curr_labels, n_clusters=n_clusters)
            cluster_distribution_years[:, i] = cluster_distribution

            clusterer.plot_labels_raster(
                labels=curr_labels,
                # title=f"Kmeans K={n_clusters} - {data[base_dir_idx]["location"]} - {year}",
                title="", # no title for report
                n_clusters=n_clusters,
                cmap = cmap,
                save_path=os.path.join("output", data[base_dir_idx]["base_dir"], f"drought_maps/{data[base_dir_idx]["location"]}_K={n_clusters}+1_{year}{f"_{data[base_dir_idx]["extra_text"]}"}.png"),
            )

            clusterer.plot_labels_histogram(
                labels=curr_labels,
                # title=f"Kmeans K={n_clusters+1} - {data[base_dir_idx]["location"]} - {year}",
                title="", # no title for report
                n_clusters=n_clusters+1, # cluster 1 will be border, 7 will be water. i know it's confusing, but i'm not gonna use this in the report
                save_path=os.path.join("output", data[base_dir_idx]["base_dir"], f"cluster_hist/{data[base_dir_idx]["location"]}_{n_clusters+1}_{year}{f"_{data[base_dir_idx]["extra_text"]}"}.png"),
            )

        # plot cluster distribution over years
        plt.figure(figsize=(12, 9))
        for i in np.arange(1,n_clusters+1): plt.plot(years, cluster_distribution_years[i], label=f"Cluster {i}", c=cmap.colors[i], linewidth=3, marker='o', markersize=10)    
        plt.xlabel("Year")
        plt.ylabel("Cluster Distribution")
        # plt.title(f"Cluster Distribution - K={n_clusters} - {data[base_dir_idx]['location']} - {data[base_dir_idx]['extra_text']}") # no title for report
        plt.ylim(0, 0.6)
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join("output", data[base_dir_idx]["base_dir"], f"cluster_hist/cluster_distribution_{n_clusters}{f"_{data[base_dir_idx]["extra_text"]}"}.png"))
        plt.close()

        # intercluster distance
        intercluster_distance_mat = clusterer.inter_cluster_distance_mat(X=combined_bands_years, labels=sorted_labels, n_clusters=n_clusters+1)

        # remove first and last row and column, the cluster 0 contains border, while the last is an artificial cluster containing water bodies, not obtained from clustering
        intercluster_distance_mat = intercluster_distance_mat[1:, 1:] # remove water and border cluster
        intercluster_distance_mat = intercluster_distance_mat[:-1, :-1] # remove water and border cluster

        plt.figure(figsize=(12, 12))
        plt.imshow(intercluster_distance_mat, cmap="viridis")
        plt.colorbar()
        # plt.title(f"Intercluster Distance Matrix - K={n_clusters} - {data[base_dir_idx]['location']} - {data[base_dir_idx]['extra_text']}\n (Water cluster excluded)") # no title for report
        plt.xlabel("Cluster Index")
        plt.ylabel("Cluster Index")
        plt.xticks(np.arange(n_clusters-1), np.arange(1, n_clusters))
        plt.yticks(np.arange(n_clusters-1), np.arange(1, n_clusters))
        for i in range(n_clusters-1):
            for j in range(n_clusters-1):
                plt.text(j, i, f"{intercluster_distance_mat[i, j]:.2f}", ha="center", va="center", color="black")
        plt.savefig(os.path.join("output", data[base_dir_idx]["base_dir"], f"intercluster_distance_{data[base_dir_idx]["location"]}_{data[base_dir_idx]["extra_text"]}.png"))
        plt.close()

        clusterer.clusters_summary(combined_bands_years,
                                sorted_labels_raster,
                                save_path=os.path.join("output",
                                                        data[base_dir_idx]["base_dir"],
                                                        f"cluster_summaries/summary_{n_clusters}_{data[base_dir_idx]['extra_text']}.csv"))