import os
import cv2
import pandas as pd
import numpy as np

def count_white_pixels(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Count white pixels (pixel value == 255)
    return np.sum(image == 255)

def compute_white_pixel_sum(directory):
    # Initialize variables to store sum of white pixels
    sum_filtered_til = 0
    sum_binary_mask = 0

    # Iterate over files in "filtered_til_mask" directory
    filtered_til_mask_dir = os.path.join(directory, "filtered_til_mask")
    for file in os.listdir(filtered_til_mask_dir):
        if file.endswith('.tif'):
            sum_filtered_til += count_white_pixels(os.path.join(filtered_til_mask_dir, file))

    # Iterate over files in "binary_masks" directory
    binary_masks_dir = os.path.join(directory, "binary_masks")
    for file in os.listdir(binary_masks_dir):
        if file.endswith('.tif'):
            sum_binary_mask += count_white_pixels(os.path.join(binary_masks_dir, file))

    return sum_filtered_til, sum_binary_mask

def til_score_generator(input_folder):
    # Initialize lists to store directory names and sum of white pixels
    directories = []
    sums_filtered_til = []
    sums_binary_mask = []

    # Search for directories containing "binary_masks" and "filtered_til_mask"
    for root, dirs, files in os.walk(input_folder):
        if "binary_masks" in dirs and "filtered_til_mask" in dirs:
            directories.append(os.path.basename(root))
            sum_filtered_til, sum_binary_mask = compute_white_pixel_sum(root)
            sums_filtered_til.append(sum_filtered_til)
            sums_binary_mask.append(sum_binary_mask)

    # Calculate the ratio of sums_filtered_til to sums_binary_mask
    ratios = [sums_filtered_til[i] / sums_binary_mask[i] for i in range(len(sums_filtered_til))]

    # Create DataFrame from the collected data
    df = pd.DataFrame({'Directory': directories,
                       'Sum_White_Pixels_Filtered_Til': sums_filtered_til,
                       'Sum_White_Pixels_Binary_Mask': sums_binary_mask,
                       'Ratio_Filtered_Til_to_Binary_Mask': ratios})

    # Save DataFrame to Excel
    df.to_excel(os.path.join(input_folder, "til_score.xlsx"), index=False)


