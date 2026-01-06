# Importing requried packages
import skimage.io as io
import numpy as np
import cv2
import os
import pandas as pd
import pickle

def epith_mask_saver(in_stitched_3CC_path, epithelia_color, out_stitched_epithelium_path):
    """
    Generate and save a binary epithelium mask from a stitched 3-channel image.

    This function loads a stitched RGB image of the 3CC tissue classifier output and identifies pixels
    that exactly match the specified epithelial color. A binary mask is created
    where epithelial pixels are set to 1 and all others to 0, and the mask is
    written to disk as an image.

    Parameters
    ----------
    in_stitched_3CC_path : str
        File path to the stitched 3CC tissue classified image.
    epithelia_color : array-like of length 3
        RGB color value corresponding to epithelium pixels
        (e.g., [R, G, B]).
    out_stitched_epithelium_path : str
        Output file path for the binary epithelium mask image.

    Returns
    -------
    None
        The function saves the epithelium mask to disk and does not return a value.
    """

    stitched_3CC = io.imread(in_stitched_3CC_path)
    epithelium_mask = np.all(stitched_3CC == epithelia_color, axis=-1).astype(np.uint8)
    # np.save(out_stitched_epithelium_path, epithelium_mask)
    cv2.imwrite(out_stitched_epithelium_path, epithelium_mask)
    del stitched_3CC
    del epithelium_mask

    return None


def epith_cluster_filterer(in_stitched_epithelium_path, out_stitched_filtered_epith_clusters_distance_transform_path, max_ep_cluster_area_pixels):
    """
    Filter epithelial clusters by area and compute a distance transform.

    This function loads a binary epithelium mask, extracts connected epithelial
    clusters, and removes clusters smaller than a specified area threshold.
    The remaining clusters are used to generate an inverted mask, from which
    a Euclidean distance transform is computed and saved to disk.

    Parameters
    ----------
    in_stitched_epithelium_path : str
        File path to the binary stitched epithelium mask image.
    out_stitched_filtered_epith_clusters_distance_transform_path : str
        Output file path for the saved distance transform (.npy file).
    max_ep_cluster_area_pixels : int
        Minimum area (in pixels) required for an epithelial cluster to be retained.

    Returns
    -------
    None
        The function saves the distance transform to disk and does not return a value.
    """
    
    epithelium_mask = io.imread(in_stitched_epithelium_path)
    shape = epithelium_mask.shape
    # Connectivey = 1 I think. Need to ponder on this.
    contours, _ = cv2.findContours(epithelium_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    del epithelium_mask
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= max_ep_cluster_area_pixels]
    epithelium_cluster_mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(epithelium_cluster_mask, filtered_contours, contourIdx=-1, color=255, thickness=cv2.FILLED)
    inverted_epithelium_cluster_mask = cv2.bitwise_not(epithelium_cluster_mask)
    del epithelium_cluster_mask
    distance_transform = cv2.distanceTransform(inverted_epithelium_cluster_mask, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
    del inverted_epithelium_cluster_mask
    np.save(out_stitched_filtered_epith_clusters_distance_transform_path, distance_transform)
    del distance_transform

    return None


def spatial_distance_masker(dist_mask_path, dist, out_stitched_spatial_distance_mask_path):
    """
    Create and save a binary spatial distance mask from a distance transform.

    This function loads a precomputed distance transform and thresholds it to
    generate a binary mask where pixels with distance less than or equal to the
    specified threshold are retained.

    Parameters
    ----------
    dist_mask_path : str
        File path to the saved distance transform (.npy file).
    dist : float or int
        Distance threshold (in pixels) for mask generation.
    out_stitched_spatial_distance_mask_path : str
        Output file path for the binary spatial distance mask image.

    Returns
    -------
    None
        The function saves the spatial distance mask to disk and does not return a value.
    """

    dist_mask = (np.load(dist_mask_path) <= dist).astype(np.uint8)
    cv2.imwrite(out_stitched_spatial_distance_mask_path, dist_mask)
    del dist_mask

    return None


def spatial_stroma_filterer(stitched_stroma_mask_path, stitched_spatial_distance_mask_path, out_stitched_filtered_stroma_path):
    """
    Filter stromal regions based on proximity to epithelial clusters.

    This function loads a stitched stroma mask and a spatial distance mask, then
    performs a logical AND operation to retain only stromal pixels within the
    specified spatial distance from epithelial clusters. The filtered stroma
    mask is saved to disk.

    Parameters
    ----------
    stitched_stroma_mask_path : str
        File path to the stitched stroma mask image.
    stitched_spatial_distance_mask_path : str
        File path to the binary spatial distance mask image.
    out_stitched_filtered_stroma_path : str
        Output file path for the filtered stroma mask image.

    Returns
    -------
    None
        The function saves the filtered stroma mask to disk and does not return a value.
    """

    stroma_mask = io.imread(stitched_stroma_mask_path)[:,:,0] == 255
    distance_mask = io.imread(stitched_spatial_distance_mask_path)
    filtered_stroma_mask = stroma_mask & distance_mask
    del stroma_mask
    del distance_mask
    cv2.imwrite(out_stitched_filtered_stroma_path, filtered_stroma_mask)
    del filtered_stroma_mask

    return None


def contours_creator(stitched_tils_path, out_contours_path):
    # convert stitched image to binary
    stitched_mask = io.imread(stitched_tils_path)[:,:,0]
    stitched_binary = (stitched_mask > 0).astype(np.uint8)
    del stitched_mask

    # retrieve contours
    wsi_contours, _ = cv2.findContours(
        stitched_binary,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    )
    del stitched_binary

    # Save contours
    with open(out_contours_path, "wb") as f:
        pickle.dump(wsi_contours, f)

    return None


def tilseg_score_calculator(out_spatial_stroma_path, out_stitched_filtered_tils_contours_path):
    """
    Compute TIL spatial metrics restricted to filtered stromal regions for a single WSI.

    This function loads a binary stromal mask (in a spatially confined region about epithelial clusters) and a set of pre-filtered TIL contours,
    retains only those TIL contours whose centroids fall within the stromal mask,
    and computes quantitative stromal tumor-infiltrating lymphocyte (sTIL) metrics including
    sTIL area, stromal area, sTIL density score, and sTIL count.

    The sTIL score is defined as the ratio of total sTIL pixel area to total stromal
    pixel area within the filtered & spatially confined stroma.

    Parameters
    ----------
    out_spatial_stroma_path : str
        Path to the directory containing the filtered stromal masks.
        The stromal mask is expected to be a binary TIFF image named using the WSI
        identifier (WSI[:-4] + '.tif'), where nonzero pixels denote stromal regions.

    out_stitched_filtered_tils_contours_path : str
        Path to a pickle file containing a list of OpenCV contours representing
        filtered TIL cell segmentations for the WSI.

    Returns
    -------
    tilscore : float
        TIL density score, computed as:
        (total sTIL area in pixels) / (total stromal area in pixels).

    tilcount : int
        Number of sTIL contours whose centroids fall within the filtered & spatially confined stromal mask.

    tilarea : int
        Total area (in pixels) occupied by retained sTIL contours within the stroma.

    stromaarea : int
        Total area (in pixels) of the filtered stromal mask.

    Notes
    -----
    - Contour centroids are computed using image moments; contours with degenerate
      moments (e.g., zero area) are skipped.
    - TIL area is computed by rasterizing retained contours into a filled binary mask.
    - This function assumes consistent coordinate alignment between the stromal mask
      and TIL contours.
    """


    # Load filtered stroma mask and compute its area (number of nonzero pixels)
    arr_spatial_stroma = io.imread(out_spatial_stroma_path)
    stromaarea = np.count_nonzero(arr_spatial_stroma)

    # Load TIL contours for this WSI
    with open(out_stitched_filtered_tils_contours_path, "rb") as f:
        contours_spatial_til = pickle.load(f)

    # -----------------------------------------------------------------
    # Keep only contours whose centroid falls inside the filtered stroma
    # -----------------------------------------------------------------
    filtered_contours = []
    for c in contours_spatial_til:
        M = cv2.moments(c)
        try:
            # Centroid coordinates from image moments
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            # Convert to integer pixel indices (x=col, y=row)
            ix, iy = int(round(cx)), int(round(cy))
            # Retain contour if centroid pixel is inside filtered stroma (nonzero)
            if arr_spatial_stroma[iy, ix]:
                filtered_contours.append(c)
        except:
            # Handles degenerate contours where m00==0 or invalid centroid computation
            print('Not able to find cell centroid for one cell')

    # Create a filled mask from retained contours to compute TIL area in pixels
    shape = arr_spatial_stroma.shape
    del arr_spatial_stroma
    filtered_tils_mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(filtered_tils_mask, filtered_contours, contourIdx=-1, color=255, thickness=cv2.FILLED)

    # Compute metrics
    tilarea = np.count_nonzero(filtered_tils_mask)
    del filtered_tils_mask
    tilscore = tilarea / stromaarea
    tilcount = len(filtered_contours)

    return tilscore, tilcount, tilarea, stromaarea