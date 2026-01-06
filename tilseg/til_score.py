import os
import cv2
import pandas as pd
import numpy as np
import skimage.io as io

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
        arr_spatial_stroma = io.imread(os.path.join(out_stitched_filtered_stroma_dir, directory) + '.tif')[:,:,0] == 255 # count purely white pixels (discounting overlap)
        arr_spatial_til = (io.imread(os.path.join(out_stitched_filtered_tils_dir, directory) + '.tif')[:,:,0] > 0).astype(np.uint8) # count any non-black pixels (accounting for overlapped cells)

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


