import os
import cv2
import pandas as pd
import numpy as np
import skimage.io as io

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

def compute_global_tilseg_score(mainPath, directories):

    # retrieve stitched binary masks (denominator) and eroded til masks (numerator)
    stitching_dir = os.path.join(mainPath, "stitching")
    out_stitched_filtered_tils_dir = os.path.join(stitching_dir, "stitched_filtered_til_mask_eroded")
    out_stitched_filtered_stroma_dir = os.path.join(stitching_dir, "stitched_binary_stroma")

    tilscore_list = []
    tilcount_list = []
    tilarea_list = []
    stromaarea_list = []
    tilarea_contour_list = []
    tilarea_contour_masked_list = []

    for directory in directories:

        # create boolean arrays 
        arr_spatial_stroma = io.imread(os.path.join(out_stitched_filtered_stroma_dir, directory))[:,:,0] == 255 # count purely white pixels (discounting overlap)
        arr_spatial_til = (io.imread(os.path.join(out_stitched_filtered_tils_dir, directory))[:,:,0] > 0).astype(np.uint8) # count any non-black pixels (accounting for overlapped cells)

        # calculate the total number of non-zero pixels in the sTIL and stromal masks
        cur_tilarea = np.count_nonzero(arr_spatial_til)
        cur_stromaarea = np.count_nonzero(arr_spatial_stroma)

        # calculate the sTIL score
        cur_tilscore = cur_tilarea / cur_stromaarea

        # count the number of sTIL contours
        contours, _ = cv2.findContours(arr_spatial_til, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cur_tilcount = len(contours)
        cur_tilarea_contour = sum([cv2.contourArea(c) for c in contours])

        # draw and calculate the area of the masked sTIL contours
        filtered_tils_mask = np.zeros(arr_spatial_stroma.shape, dtype=np.uint8)
        cv2.drawContours(filtered_tils_mask, contours, contourIdx=-1, color=255, thickness=cv2.FILLED)
        cur_tilarea_contour_masked = np.count_nonzero(filtered_tils_mask)

        # append outputs
        tilscore_list.append(cur_tilscore)
        tilcount_list.append(cur_tilcount)
        tilarea_list.append(cur_tilarea)
        stromaarea_list.append(cur_stromaarea)
        tilarea_contour_list.append(cur_tilarea_contour)
        tilarea_contour_masked_list.append(cur_tilarea_contour_masked)

    # delineate csv columns
    df = pd.DataFrame({'Directory': directories, 'tilscore': tilscore_list, 'tilcount': tilcount_list, 'tilarea': tilarea_list, 'tilarea_Contour': tilarea_contour_list, 'tilarea_Contour_Masked': tilarea_contour_masked_list, 'stromaarea': stromaarea_list})
    df['tilarea_percDiff'] = (df['tilarea_Contour_Masked'] - df['tilarea']) / df['tilarea'] * 100

    # save results as a .csv file to mainPath directory
    csv_path = os.path.join(mainPath, 'global_tilseg_scores.csv')
    df.to_csv(csv_path)


