import numpy as np
import rasterio
import os
import logging
import pyproj
import pandas as pd
from rasterio.windows import Window
import numpy as np
import rasterio
import os
import logging
from scipy.ndimage import convolve

# Configure logging
logging.basicConfig(filename='noise_masking_whole.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_land_cover_probability(land_cover):
    land_cover_probs = {
        24: 0.9, 23: 0.8, 22: 0.7, 21: 0.6, 82: 0.5, 81: 0.4, 90: 0.3, 95: 0.2,
        43: 0.1, 42: 0.1, 41: 0.1, 52: 0.1, 51: 0.1, 73: 0.1, 74: 0.1, 72: 0.1, 71: 0.1,
        31: 0.05, 11: 0.05, 12: 0.01
    }
    return land_cover_probs.get(land_cover, 0.1)

def precompute_land_cover_probabilities(image, window_size):
    land_cover_band = image[0]  # Assuming the first band is land cover
    land_cover_probs = np.vectorize(get_land_cover_probability)(land_cover_band)
    

    # Log unique land cover probabilities
    unique_probs, counts = np.unique(land_cover_probs, return_counts=True)
    
    # Create a box filter of the same window size
    kernel = np.ones((window_size, window_size))

    # Convolve the land cover probabilities to get the local average
    local_land_cover_prob = convolve(land_cover_probs, kernel, mode='constant', cval=0)
    
    # Normalize the local probabilities by the window area
    local_land_cover_prob /= (window_size ** 2)

    return local_land_cover_prob

    
# Define file paths to Excel files
excel_2019 = '2018_new.xlsx'
excel_2008 = '2008_new.xlsx'
excel_2021 = '2022_new.xlsx'

# Function to read lat/lon from Excel
def read_lat_lon_from_excel(excel_file, index):
    df = pd.read_excel(excel_file)
    lat = df.at[index, 'Latitude']
    lon = df.at[index, 'Longitude']
    return lat, lon

# Transformer setup
with rasterio.open('2008_nlcd.tif') as src:
    crs = src.crs
    transformer = pyproj.Transformer.from_crs("epsg:4326", crs, always_xy=True)

def extract_patch_from_raster(raster_path, lat, lon, patch_size, transformer):
    """Extracts a patch of given size around a lat/lon coordinate from the raster."""
    with rasterio.open(raster_path) as src:
        x, y = transformer.transform(lon, lat)
        row, col = src.index(x, y)
        half_size = patch_size // 2
        window = Window(col - half_size, row - half_size, patch_size, patch_size)
        #patch = src.read(window=window)
        patch = src.read(1, window=window)
    return patch

# Step 2: Skip saved (x, y) and apply exponential decay for other pixels
def apply_exponential_decay(pixel, saved_coords, saved_probs,central_pixel_pos, pixel_value, decay_rate=0.1):
    total_prob = 0
    for (x_saved, y_saved), prob_saved in zip(saved_coords, saved_probs):
        
        distance_from_central = np.linalg.norm(np.array([x_saved, y_saved]) - np.array([pixel[0], pixel[1]]))
        if ((prob_saved == 1 and pixel_value == 0) or (prob_saved == 0 and pixel_value == 1)):
            central_pixel_prob = 1 - np.exp(-distance_from_central/70)
        else:
            central_pixel_prob = np.exp(-distance_from_central/70)
       
        
        total_prob += central_pixel_prob
    return total_prob / len(saved_probs) if len(saved_probs) > 0 else 0
    

def get_pixel_probability(pixel_value, x, y, central_value, prefix, non_2_vals):
    """Determines probability based on pixel value comparisons."""
    
    logging.info(pixel_value)
    
    # Determine the year based on the prefix
    if prefix == "train_sep_patch_2022_":
        val = 2022
    elif prefix == "train_sep_2008_patch_":
        val = 2008
    elif prefix == "train_sep_patch_":
        val = 2018
   

    logging.info(f'val: {val}')
    # Get the last digit of the central value
    central_last_digit = central_value
    segment_first_four = None
    pixel_last_digit = None
    pixel_is_higher = None
    
    
    # Check the number of digits in pixel_value
    digit_length = len(str(pixel_value))
    logging.info(f'digit_length: {digit_length}')
    
    if digit_length >= 5:
        # Split pixel_value into parts of 5 digits each
        segments = [int(str(pixel_value)[i:i + 5]) for i in range(0, digit_length, 5)]
        
        # Check for matches in the first four digits of each segment
        for segment in segments:
            segment_first_four = int(str(segment)[:4])
            pixel_last_digit = segment % 10
            
            
            if segment_first_four == val:
                # Return the last digit of the matching segment
                logging.info(f'segment_first_four == val: {segment_first_four}')
                pixel_last_digit = segment % 10  # Update pixel_last_digit
                
                return pixel_last_digit
            else:
                continue
    
    logging.info(f'segment_first_four: {segment_first_four}')
    if len(non_2_vals) == 1:
        return pixel_last_digit
    elif (x == 256) and (y == 256):
        return pixel_last_digit

    # Default case, if no rule applies
    logging.info("no get_pixel_probability")

def normalize_distances_per_band(distances_per_band, global_min_max):
    normalized_distances = []
    for distances, (min_dist, max_dist) in zip(distances_per_band, global_min_max):
        distances = np.array(distances)
        if max_dist > min_dist:
            normalized_distances.append((distances - min_dist) / (max_dist - min_dist))
        else:
            normalized_distances.append(np.zeros_like(distances))
    return normalized_distances

def load_image(image_path):
    with rasterio.open(image_path) as src:
        bands = [src.read(i + 1) for i in range(src.count)]
    logging.info(f'Loaded image: {image_path}')
    return np.stack(bands)

def load_mask(mask_path):
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    logging.info(f'Loaded mask: {mask_path}')
    return mask

def save_noise_mask(output_path, noise_mask):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    with rasterio.open(output_path, 'w', driver='GTiff', height=noise_mask.shape[0], width=noise_mask.shape[1], count=1, dtype=noise_mask.dtype) as dst:
        dst.write(noise_mask, 1)
    logging.info(f'Saved noise mask: {output_path}')




import rasterio
import numpy as np

# Define flow direction mapping
FLOW_DIRECTIONS = {
    64: (-1, 0),  # North
    128: (-1, 1),  # Northeast
    1: (0, 1),   # East
    2: (1, 1),   # Southeast
    4: (1, 0),   # South
    8: (1, -1),  # Southwest
    16: (0, -1),  # West
    32: (-1, -1)  # Northwest
}
    
    
def assign_downstream_probabilities1(img, noise, haz, x, y):
    """
    Assign probabilities to downstream and upstream pixels from a starting location.

    Args:
        img (array): The input raster image containing multiple bands.
        noise (array): An array representing probabilities to update.
        haz (array): Hazard array to determine propagation conditions.
        x (int): Row index of the starting pixel.
        y (int): Column index of the starting pixel.

    Returns:
        Updated `noise` array with downstream and upstream probabilities assigned.
    """
    flow_dir = img[43]
    
    # Function to propagate probabilities
    def propagate_probability(row, col, current_prob, flow_direction_map):
        if (row, col) in visited:
            return  # Skip already visited pixels
        visited.add((row, col))  # Mark pixel as visited

        if row < 0 or row >= flow_dir.shape[0] or col < 0 or col >= flow_dir.shape[1]:
            return  # Out of bounds
        if haz[row, col] == 2:
            return  # Stop propagation at hazard value 2
            
        if ((haz[row,col] == 1 and haz[x,y] == 0) or (haz[row,col]==0 and haz[x,y]==1)):
            noise[row, col] += 0.1
        elif ((haz[row,col] == 1 and haz[x,y] == 1) or (haz[row,col]==0 and haz[x,y]==0)):
            noise[row, col] += 0.9
         
        noise[row, col] = min(noise[row, col], 1)  # Cap probability at 1

        # Get flow direction for the current pixel
        flow_value = flow_dir[row, col]
        if flow_value not in flow_direction_map:
            return

        # Get downstream offset
        dr, dc = flow_direction_map[flow_value]
        next_row, next_col = row + dr, col + dc

      
        propagate_probability(next_row, next_col, 0.9, flow_direction_map)

    # Define flow direction maps
    downstream_map = FLOW_DIRECTIONS  # Standard downstream directions
    
    if haz[x, y] == 1:
        visited = set()  # Track visited pixels
        propagate_probability(x, y, base_probability, downstream_map)
        visited = set()  # Track visited pixels
    elif haz[x, y] == 0:
        visited = set()  # Track visited pixels
        propagate_probability(x, y, base_probability, downstream_map)
        visited = set()  # Track visited pixels

    for i in range(noise.shape[0]):
        for j in range(noise.shape[1]):
            if haz[i, j] != 2 and noise[i, j] == 0:  # If haz is not 2 and noise is 0
                noise[i, j] = 0.2 # Set the value to 0.2

    return noise






def calculate_probability(pixel, bands, central_pixel_pos, local_land_cover_prob, global_min_max, window_size, central_pixel_prob, pixel_value, noise):
    x, y = pixel
    central_x, central_y = central_pixel_pos
  
   
    if(pixel_value==1):
        land_cover_prob = local_land_cover_prob[y, x]
    else:
        land_cover_prob = 1 - local_land_cover_prob[y, x]

    
    distances_per_band = [bands[band][y, x] for band in range(1, 43)]
    
    # Normalize distances across bands
    normalized_distances = normalize_distances_per_band(distances_per_band, global_min_max)
    
    # Calculate the distance probability
    inverse_distance_sum = np.sum([np.sum(1 / (1 + dist)) for dist in normalized_distances])
    if(pixel_value==1):
        distance_prob = inverse_distance_sum / len(normalized_distances)
    else:
        distance_prob = 1 - (inverse_distance_sum / len(normalized_distances))
    
   
    prob = 0.20 * land_cover_prob + 0.40 * distance_prob + 0.30*downstream_flow + 0.10*central_pixel_prob
    prob = np.clip(prob, 0, 1)
   
    return prob

def create_noise_masks(image_dir, mask_dir, mask_haz_dir, output_dir, window_size=42, patch_size=256):
    """Main function to create noise masks based on image and mask directories."""
    image_files = [f for f in os.listdir(image_dir) if f.endswith('_merged.tif')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('_mask.tif')]
    mask_haz_files = [f for f in os.listdir(mask_haz_dir) if f.endswith('_mask.tif')]

    for image_file in image_files:
        base_name = image_file.replace('_merged.tif', '')
        mask_file = base_name + '_mask.tif'
        mask_haz_file = base_name + '_mask.tif'
       
        if mask_file in mask_files and mask_haz_file in mask_haz_files:
            prefix=None
            saved_coords = []
            saved_probs = []
            img_path = os.path.join(image_dir, image_file)
            mask_path = os.path.join(mask_dir, mask_file)
            mask_haz_path = os.path.join(mask_haz_dir, mask_haz_file)
            output_path = os.path.join(output_dir, base_name + '_mask.tif')
            
            img = load_image(img_path)
            mask = load_mask(mask_path)
            mask_haz = load_mask(mask_haz_path)
          
            noise_mask = np.where(mask_haz != 2, 0, 1).astype(np.float32)
           
            local_land_cover_prob = precompute_land_cover_probabilities(img, window_size)
           

            base_name_cleaned = base_name.strip().lower()
            logging.info(f"Base name being checked: '{base_name_cleaned}'")
            
           
            if base_name_cleaned.startswith("train_sep_patch_2022_"):
                logging.info("Entered the 2022 case (startswith)")
                logging.info(f"Matched base name: {base_name_cleaned}")
                prefix = "train_sep_patch_2022_"
                excel_file = excel_2021
                excel_file_name = "excel_2021"
                logging.info(excel_file_name)
            elif base_name_cleaned.startswith("train_sep_patch_"):
                prefix = "train_sep_patch_"
                excel_file = excel_2019
                excel_file_name = "excel_2019"
                logging.info(excel_file_name)
            elif base_name_cleaned.startswith("train_sep_2013_patch_"):
                prefix = "train_sep_2013_patch_"
                excel_file = excel_2013
                excel_file_name = "excel_2013"
                logging.info(excel_file_name)
            elif base_name_cleaned.startswith("train_sep_2008_patch_"):
                prefix = "train_sep_2008_patch_"
                excel_file = excel_2008
                excel_file_name = "excel_2008"
                logging.info(excel_file_name)
            elif "train_sep_patch_2022_" in base_name_cleaned:
                #logging.info("Entered the 2022 case (in check)")
                #logging.info(f"Matched base name with 'in': {base_name_cleaned}")
                prefix = "train_sep_patch_2022_"
                excel_file = excel_2021
                excel_file_name = "excel_2021"
                logging.info(excel_file_name)
            else:
                logging.warning(f"Base name did not match any condition: '{base_name_cleaned}'")

            index = int(image_file.split('_')[-2])
            logging.info(index)
            lat, lon = read_lat_lon_from_excel(excel_file, index)

            
            patch = extract_patch_from_raster("masks_new.tif", lat, lon, patch_size, transformer)
            logging.info(patch.shape)
           
            non_2_vals = patch[patch != 2]
            central_pixel = mask[patch_size // 2, patch_size // 2]
          
            central_pixel_pos = (patch_size // 2, patch_size // 2)

          
            for x in range(patch.shape[0]):
                for y in range(patch.shape[1]):
                    if ((patch[x, y] != 2)):
                        
                        logging.info(f'patch[x, y] != 2: {patch[x, y]}')
                        digit_length = len(str(patch[x, y]))
    
    
                        if digit_length >= 5:
                            # Split pixel_value into parts of 5 digits each
                            segments = [int(str(patch[x, y])[i:i + 5]) for i in range(0, digit_length, 5)]
                            
                            # Check for matches in the first four digits of each segment
                            for segment in segments:
                                segment_first_four = int(str(segment)[:4])
                                pixel_last_digit = segment % 10
                                check=0
                                if segment_first_four == 2018:
                                    # Return the last digit of the matching segment
                                    logging.info(f'segment_first_four == val: {segment_first_four}')
                                    pixel_last_digit = segment % 10  # Update pixel_last_digit
                                    check=1
                                    break
                                    
                                else:
                                    continue
                
                        if(check==1):
                            prob = get_pixel_probability(patch[x, y],x,y, central_pixel,prefix,non_2_vals)
                         
                            logging.info(f'saved coord: {patch[x, y]}')
                            saved_coords.append((x, y))
                            saved_probs.append(prob)
                            noise_mask = assign_downstream_probabilities1(img, noise_mask, mask, x,y)
                            noise_mask[x, y] = 1
                        
                        
            # Process pixels that are 2 in the mask
            for x in range(mask.shape[0]):
                for y in range(mask.shape[1]):
                    if mask[x, y] == 2:
                        if mask_haz[x, y] == 0 or mask_haz[x, y] == 1:
                            noise_mask[x, y] = 0  # Set to 0 if mask_haz is 0 or 1
                        elif mask_haz[x, y] == 2:
                            noise_mask[x, y] = 1  # Set to 1 if mask_haz is 2
                            
           
            logging.info(len(saved_coords))
         
           
            for (i, j) in np.argwhere((mask == 1) | (mask==0)):
                pixel = (j, i)
                # Skip the saved (x, y) pixels
                if (i, j) in saved_coords:
                    continue
                pixel_value = mask[i, j]
                # Calculate the central pixel probability using exponential decay
                central_pixel_prob = apply_exponential_decay((i, j), saved_coords, saved_probs, central_pixel_pos, pixel_value)
                bands = [img[band] for band in range(45)]
                prob = calculate_probability(pixel, bands, central_pixel_pos, local_land_cover_prob, global_min_max, window_size, central_pixel_prob, pixel_value, noise_mask )
               
                noise_mask[i, j] = prob
           
            logging.info(f'Preparing to save noise mask: {output_path}')
            save_noise_mask(output_path, noise_mask)

            logging.info(f'Processed image: {image_file}')


create_noise_masks(
    'train_latest_new_256n/images_flowdir_arc',
    'train_latest_new_256n/masks',
    'train_latest_new_256n/masks',
    'train_latest_new_256n/noise_masks'
)


