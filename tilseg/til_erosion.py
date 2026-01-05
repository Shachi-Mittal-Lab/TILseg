## IMPORTS
import numpy as np
import pandas as pd
import cv2
import os
import pickle
import skimage.io as io
import skimage


def contour_to_label_mask(contours, patch_shape=(3000, 4000)):
    """
    Draws contours onto an empty mask to create a binary image of sTIL segmentations.

    Args:
        contours (list): List of contour arrays, where each contour represents 
        the segmentation of a single cell
        Typically obtained from OpenCV contour detection functions.
        patch_shape (tuple): Shape of the output mask as (height, width). Default is (3000, 4000)

    Returns:
        label_mask (np.ndarray): A 2D labeled mask of shape `patch_shape` where 0 is a background pixel
          and 1 is a TIL pixel
    """
    # create an empty mask
    label_mask = np.zeros(patch_shape, dtype=np.int32) # use int32 for opencv compatibility

    # draw contours onto empty mask
    for i, contour in enumerate(contours, start=1):
        cv2.drawContours(label_mask, [contour], -1, color=i, thickness=cv2.FILLED)

    return label_mask.astype(np.uint32) # convert to uint32 for storage


def erode_overlaps(slide_dir, slidename):
    """
    Process segmented contours to create eroded binary masks by removing boundary pixels.
    This function loads pre-processed segmentation contours from a pickle file, converts them
    to labeled masks, identifies and removes boundary pixels to create separation between
    adjacent regions (simulating erosion), and saves the resulting binary masks as image files.

    Args:
        slide_dir (str): path to the output directory for a slide
        slidename (str): corresponding slide name

    Returns:
        N/A
    """
    seg_pkl_path = os.path.join(slide_dir, f"{slidename}_filtered_segmentations.pkl")
    out_dir = os.path.join(slide_dir, 'filtered_til_mask_eroded')
    os.makedirs(out_dir, exist_ok=True)

    with open(seg_pkl_path, 'rb') as f:
        segmentations = pickle.load(f)

    for patch, contours in segmentations.items():
        # generate label mask
        label_mask = contour_to_label_mask(contours)
        boundary_bool = skimage.segmentation.find_boundaries(label_mask, connectivity=label_mask.ndim,
                                                                 mode='outer', background=0)
        
        # converting these pixels to the background value in the label array
        label_mask[boundary_bool] = 0
        
        # converting the label array into a binary mask of foreground (255) and background (0)
        nuclei_mask_final = np.zeros((label_mask.shape[0], label_mask.shape[1]))
        nuclei_mask_final[label_mask != 0] = 255
        nuclei_mask_final = np.uint8(nuclei_mask_final)
        
        # saving the binary mask in the save directory
        filtered_til_mask_eroded_patch_path = os.path.join(out_dir, patch)
        if not os.path.exists(filtered_til_mask_eroded_patch_path):
            io.imsave(filtered_til_mask_eroded_patch_path, nuclei_mask_final, check_contrast=False)

    return None