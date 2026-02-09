import os
import numpy as np
import rasterio
from rasterio.windows import Window
import pyproj
import pandas as pd
import logging
import time


logging.basicConfig(filename='patch_extraction_2008.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def extract_patch_from_raster(raster_path, lat, lon, patch_size, transformer):
    with rasterio.open(raster_path) as src:
      
        x, y = transformer.transform(lon, lat)
        
        row, col = src.index(x, y)
        
        half_size = patch_size // 2
        window = Window(col - half_size, row - half_size, patch_size, patch_size)
        patch = src.read(window=window)
    return patch, row, col

def create_custom_mask(main_patch, patch_size, label, nodata_value):
    if np.any(main_patch != nodata_value):
        mask = np.full((patch_size, patch_size), 2, dtype=np.int8)
        target_value = 1 if label == 1 else 0
        
        # Set the mask value to target_value for all valid (non-nodata) pixels
        mask[main_patch != nodata_value] = target_value
        
        return mask
    else:
        return None
        
# Function to thicken the lines (dilate) in the mask
def thicken_mask(mask, kernel_size=(5, 5), iterations=2):
    kernel = np.ones(kernel_size, np.uint8)
    
    # Convert mask to int16 for processing
    mask = mask.astype(np.int16)
    
    # Handle different mask values
    mask_0 = (mask == 0).astype(np.uint8)
    mask_1 = (mask == 1).astype(np.uint8)
    
    # Dilate the regions of the mask
    dilated_0 = cv2.dilate(mask_0, kernel, iterations=iterations)
    dilated_1 = cv2.dilate(mask_1, kernel, iterations=iterations)
    
    # Recombine the thickened areas into the mask
    new_mask = np.copy(mask)
    new_mask[dilated_0 == 1] = 0
    new_mask[dilated_1 == 1] = 1
    
    return new_mask.astype(np.int16)

def save_as_tiff(data, filename, crs, transform):
    # Ensure data is 3D: (bands, height, width)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        for i in range(data.shape[0]):
            dst.write(data[i], i + 1) 

# Define paths and parameters
raster_path = '2008_nlcd.tif'  
data = pd.read_excel('2008.xlsx')  
patch_size = 256

# Prepare additional band paths
additional_band_paths = [f'distance_band_{k}.tif' for k in range(1, 22)]
additional_band_paths += [f'distance_band_epa{k}.tif' for k in range(1, 22)]


additional_band_paths += [
    'flow.tif'
]
water_path = 'merged_raster_water.tif'
# Define coordinate transformation
with rasterio.open(raster_path) as src:
    crs = src.crs
    transform = src.transform
    transformer = pyproj.Transformer.from_crs("epsg:4326", crs, always_xy=True)

patches = []
masks = []

logging.info("Starting patch extraction...")

for idx, row in data.iterrows():
    lat, lon = row['latitude'], row['longitude']
    main_patch, patch_row, patch_col = extract_patch_from_raster(raster_path, lat, lon, patch_size, transformer)
    water_patch, patch_row, patch_col = extract_patch_from_raster(water_path, lat, lon, patch_size, transformer)
    logging.info(f"Main patch shape: {main_patch.shape}")  
    
    additional_patches = []
    for band_path in additional_band_paths:
        additional_patch, _, _ = extract_patch_from_raster(band_path, lat, lon, patch_size, transformer)
        logging.info(f"Additional patch shape before adding new axis: {additional_patch.shape}")  
        additional_patches.append(additional_patch)  # Add new axis to match dimensions
    
    logging.info(f"Additional patches shape: {[patch.shape for patch in additional_patches]}")  
    
    stacked_patch = np.vstack([main_patch, *additional_patches])
    mask = create_custom_mask(water_patch[0], patch_size, row['presence'], 65535)
    if mask is not None:
        
        dilated_mask = thicken_mask(mask)
        
    
        save_as_tiff(dilated_mask, f"masks_2022/train_sep_patch_{idx}_mask.tif", crs, transform)
        
        
        masks.append(dilated_mask)
        patches.append(stacked_patch)
        logging.info(f"Processed patch {idx+1}/{len(data)} at (lat: {lat}, lon: {lon})")
        
        
        # Save the patch with the additional mask band
        save_as_tiff(stacked_patch, f"images_2022/train_sep_2013_patch_{idx}_merged.tif", crs, transform)
        
        
    else:
        logging.info(f"Discarded patch {idx+1}/{len(data)} due to nodata values.")
    
