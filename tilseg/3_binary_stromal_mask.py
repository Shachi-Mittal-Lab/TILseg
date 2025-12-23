import os
import numpy as np
from skimage import io
from multiprocessing import Pool
import time

def process_image(file, patches_dir, masks_directory):
    # Save the modified image with stroma_binary_mask prefix as a tif file
    output_path = os.path.join(masks_directory, f'stroma_binary_mask_{file}')
    if os.path.exists(output_path):
        print(f"Stroma binary mask for {file} is done.")
    else:
        image_path = os.path.join(patches_dir, file)
        patch = io.imread(image_path)
        
        # Create a binary mask for red pixels
        red_mask = (patch[:, :, 0] == 255) & (patch[:, :, 1] == 0) & (patch[:, :, 2] == 0)
        green_mask = (patch[:, :, 0] == 0) & (patch[:, :, 1] == 100) & (patch[:, :, 2] == 0)
        blue_mask = (patch[:, :, 0] == 0) & (patch[:, :, 1] == 0) & (patch[:, :, 2] == 255)
        
        # Convert red pixels to white
        patch[red_mask] = [255, 255, 255]
        
        # Convert green and blue pixels to black
        patch[green_mask | blue_mask] = [0, 0, 0]       

        io.imsave(output_path, patch, plugin='tifffile')

        # Introduce a sleep period to simulate processing time
        time.sleep(1)  # Sleep for 1 second

def convert_red_to_white_and_save_masks(directory_path):
    # Create a new directory to store binary masks
    patches_dir = os.path.join(directory_path, 'filtered_patches')
    if os.path.exists(patches_dir) == False:
        patches_dir = os.path.join(directory_path, '3class')
    masks_directory = os.path.join(directory_path, 'binary_masks')
    os.makedirs(masks_directory, exist_ok=True)

    all_items = os.listdir(patches_dir)

    # Filter out only files (not directories)
    files = [f for f in all_items if os.path.isfile(os.path.join(patches_dir, f)) and not f.endswith('.db')]

    # Create a pool of worker processes
    with Pool() as pool:
        pool.starmap(process_image, [(file, patches_dir, masks_directory) for file in files])

    return None

def binary_stromal_mask(directory_path):
    # directory_path = r"/media/mrl/My Passport/WSI_kaplan_meyer/2015009_H&E"
    convert_red_to_white_and_save_masks(directory_path)
