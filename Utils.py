import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

def find_sentinel2_bands(year: int, bands: tuple, base_dir: str = "data") -> dict:
    """
    Finds the file paths for specified Sentinel-2 bands for a given year.

    Args:
        year (int): The year of the Sentinel-2 data.
        bands (tuple): A tuple of band numbers to search for.
        base_dir (str, optional): The base directory where the data is stored. Defaults to "data".

    Returns:
        dict: A dictionary where keys are band identifiers (e.g., "B04") and values are the file paths to the corresponding band images.
              Returns None if any of the specified bands are not found.

    Raises:
        Warning: Prints a warning message if no matching file is found for a specified band.
    """
    pattern = os.path.join(
        base_dir,
        str(year),
        "S2*.SAFE",
        "GRANULE",
        "L2*",
        "IMG_DATA",
        "R20m",
        "*_B{}_20m.jp2" 
    )

    band_paths = {}
    for band in bands:
        band_pattern = pattern.format(band)
        matches = glob.glob(band_pattern, recursive=True)
        if matches:
            #assuming only one matching file per band
            band_paths[f"B{band}"] = matches[0]
        else:
            print(f"Warning: No matching file found for band B{band:02}")
            return None

    return band_paths


def combine_bands(bands: tuple, urban_mask: np.ndarray = None, water_mask: np.ndarray = None) -> np.ndarray:
    """
    Combines multiple bands into a single multi-dimensional array and applies optional urban and water masks.
    Parameters:
    bands (tuple): A tuple of numpy arrays representing the bands to be combined. Must contain at least one band.
    urban_mask (np.ndarray, optional): A numpy array representing the urban mask. Must have the same shape as the bands.
    water_mask (np.ndarray, optional): A numpy array representing the water mask. Must have the same shape as the bands.
    Returns:
    np.ndarray: A combined multi-dimensional array of the input bands with urban and water masks applied.
    Raises:
    ValueError: If the bands array is empty.
    ValueError: If the shape of the urban mask does not match the shape of the bands.
    ValueError: If the shape of the water mask does not match the shape of the bands.
    """
    if len(bands) == 0:
        raise ValueError("The bands array must contain at least one band.")
    
    if urban_mask is not None and bands[0].shape != urban_mask.shape:
        raise ValueError("The bands and urban mask must have the same shape.")
    
    if water_mask is not None and bands[0].shape != water_mask.shape:
        raise ValueError("The bands and water mask must have the same shape.")
    
    combined_bands = np.dstack(bands)
    if urban_mask is None: return combined_bands

    if urban_mask is not None: combined_bands[urban_mask == 1] = 0
    if water_mask is not None: combined_bands[water_mask == 1] = 0
    return combined_bands

# Debugging methods

def plot_bands(bands: np.ndarray, title: str, cmap: str = "viridis") -> None:
    if len(bands) == 0:
        raise ValueError("The bands array must contain at least one band.")
    
    fig, ax = plt.subplots(1, len(bands), figsize=(8*len(bands), 7))
    for i, band in enumerate(bands):
        ax[i].imshow(band, cmap=cmap)
        ax[i].set_title(f"Band {i+1}")
        ax[i].axis("off")
        fig.colorbar(ax[i].imshow(band, cmap=cmap), ax=ax[i], orientation="horizontal")
    fig.suptitle(title)
    plt.close()
    
def plot_raster(raster: np.ndarray, title: str, save_path: str, cbar_ticks = None, **kwargs,) -> None:
    plt.figure(figsize=(15, 7))
    plt.imshow(raster, **kwargs)
    plt.title(title)
    plt.axis("off")
    plt.colorbar(ticks=cbar_ticks)
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()
    
        
def plot_histogram(data: np.ndarray, title: str, save_path: str, xticks=None, **kwargs) -> None:
    plt.figure(figsize=(15, 7))
    plt.hist(data, **kwargs)
    plt.xticks(xticks)
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()
    
