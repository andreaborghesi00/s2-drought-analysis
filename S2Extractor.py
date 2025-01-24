from shapely.geometry import Polygon, mapping
from shapely.ops import transform
from pyproj import Transformer
import rasterio
from rasterio.mask import geometry_mask
import json
import numpy as np

class Extractor():
    def __init__(self):
        self.aoi = None
        self.aoi_coordinates = None

    def set_aoi(self, aoi_path: str) -> None:
        """
        Loads and sets the area of interest (AOI) from a GeoJSON file.

        Parameters:
            aoi_path (str): The path to the GeoJSON file containing the AOI geometry.

        Sets:
            self.aoi_coordinates (numpy.ndarray): Array of the AOI coordinates.
            self.aoi (shapely.geometry.Polygon): Polygon representing the AOI.

        Returns:
            None
        """
        with open(aoi_path, "r") as f:
            geojson_data = json.load(f)

        self.aoi_coordinates = np.array(geojson_data["coordinates"][0])
        self.aoi = Polygon(self.aoi_coordinates)

    def extract_band_data(self, band_path: str) -> np.ndarray:
        """
        Extracts band data from a given raster file, optionally masking it to a specified area of interest (AOI).

        Parameters
        ----------
        band_path : str
            The path to the raster band file to be read.

        Returns
        -------
        numpy.ndarray
            An array containing the raster band data. If an AOI is set, returns the band data masked to the AOI,
            where pixels outside the AOI are assigned NaN.
        """
        with rasterio.open(band_path) as src:
            complete_band = src.read(1)
            if self.aoi is None:
                return complete_band
            else:
                raster_crs = src.crs
                raster_transform = src.transform
                raster_shape = (src.height, src.width)

                transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
                reprojected_polygon = transform(transformer.transform, self.aoi)
                polygon_geometry = [mapping(reprojected_polygon)]

                mask = geometry_mask(
                    geometries=polygon_geometry,
                    out_shape=raster_shape,
                    transform=raster_transform,
                    invert=True
                )

                return np.where(mask, complete_band, np.nan)

    def crop_raster(self, raster: np.ndarray) -> np.ndarray:
        """
        Crops the input raster to the smallest bounding box containing all non-NaN values.

        Parameters:
        raster (np.ndarray): The input raster array with potential NaN values.

        Returns:
        np.ndarray: A cropped raster array containing only the non-NaN values.
                    If there are no non-NaN values, returns None.
        """
        non_nan_indices = np.argwhere(~np.isnan(raster))

        if non_nan_indices.size > 0:
            min_row, min_col = non_nan_indices.min(axis=0)
            max_row, max_col = non_nan_indices.max(axis=0)
            return raster[min_row:max_row+1, min_col:max_col+1]
        else:
            print("No valid data in the raster within the AOI.")
            return None