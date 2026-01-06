# Importing requried packages
import skimage.io as io
import numpy as np
import cv2
import os
import pandas as pd
import pickle

# Main inputs
in_dir_path = None
while in_dir_path is None:
    check_path = input("Path to WSIs and their annotations: ")
    if os.path.exists(check_path):
        in_dir_path = check_path
    else:
        print("The path does not exist or is invalid. Please try again.")

max_ep_cluster_area_pixels_list = None
while max_ep_cluster_area_pixels_list is None:
    check_input_1 = list(map(int, input("Input the minimum epithilial cluster filter sizes (pixels) (e.g.: 7500,12500): ").split(',')))
    def is_list_of_ints(x):
        return isinstance(x, list) and all(isinstance(i, int) for i in x)
    if is_list_of_ints(check_input_1):
        max_ep_cluster_area_pixels_list = check_input_1
    else:
        print("Please enter a list of integer values separated by commas.")

dist_list = None
while dist_list is None:
    check_input_2 = list(map(int, input("Input the maximum distance from epithilial clusters to score sTILs (pixels) (e.g.: 79,198): ").split(',')))
    def is_list_of_ints(x):
        return isinstance(x, list) and all(isinstance(i, int) for i in x)
    if is_list_of_ints(check_input_2):
        dist_list = check_input_2
    else:
        print("Please enter a list of integer values separated by commas.")


# Further paths & parameters used in the code
# Inputs to the code
stitched_3CC_dir_path = os.path.join(in_dir_path, "stitched_3cc_raw")
stitched_stroma_dir_path = os.path.join(in_dir_path, "stitched_binary_stroma")
stitched_tils_path = os.path.join(in_dir_path, "stitched_filtered_til_mask_eroded")
epithelia_color = [0, 0, 255]  # Blue
# Output file paths
out_stitched_epithelium_path = os.path.join(in_dir_path, "stitched_ep_mask")
out_stitched_filtered_epith_clusters_distance_transform_path = os.path.join(in_dir_path, "stitched_distance_transform_from_ep_cluster_mask")
out_spatial_dist_mask_path = os.path.join(in_dir_path, "stitched_spatial_distance_mask")
out_stitched_filtered_stroma_path = os.path.join(in_dir_path, "stitched_spatialFiltered_stroma_mask")
out_spatial_results_path = os.path.join(in_dir_path, "spatial_results")

# List all stitched WSI files in the directory (assumes file names end with .tif/.png/etc.)
WSIs = os.listdir(stitched_3CC_dir_path)
WSIs = [f for f in WSIs if f.endswith('.tif')]

# -----------------------------------------------------------------------------
# Step 1: Build epithelium masks from stitched 3-channel images
# -----------------------------------------------------------------------------
print("Step 1: Build epithelium masks from stitched 3-channel images")
os.makedirs(out_stitched_epithelium_path, exist_ok=True)
for WSI in WSIs:

    print(f"Processing: {WSI}")
    # Input: stitched 3-class tissue classification image for this WSI
    in_stitched_3CC_path = os.path.join(stitched_3CC_dir_path, WSI)
    # Output: binary epithelium mask (same WSI stem, saved as .tif)
    out_stitched_epithelium_path_cur = os.path.join(out_stitched_epithelium_path, WSI[:-4]+'.tif')
    
    from tilseg.spatial import epith_mask_saver
    # Create and save epithelium mask by exact RGB match to epithelia_color
    epith_mask_saver(in_stitched_3CC_path=in_stitched_3CC_path,
                     epithelia_color=epithelia_color,
                     out_stitched_epithelium_path=out_stitched_epithelium_path_cur)


# -----------------------------------------------------------------------------
# Step 2: Generate binary masks of spatially filtered (based on proximity to
#         epithelial clusters) stroma.
# -----------------------------------------------------------------------------
print("Step 2: Generate binary masks of spatially filtered stroma")
for WSI in WSIs:
    print(f"Processing: {WSI}")

    for max_ep_cluster_area_pixels in max_ep_cluster_area_pixels_list:

        # Make a per-WSI folder to store intermediate distance transforms for this threshold
        os.makedirs(os.path.join(out_stitched_filtered_epith_clusters_distance_transform_path, WSI[:-4]), exist_ok=True)
        # Compute distance transform from the retained epithelial clusters.
        # Saved as a .npy array for later thresholding by `dist`.
        from tilseg.spatial import epith_cluster_filterer
        epith_cluster_filterer(in_stitched_epithelium_path=os.path.join(out_stitched_epithelium_path, WSI[:-4]+'.tif'),
                            out_stitched_filtered_epith_clusters_distance_transform_path=os.path.join(out_stitched_filtered_epith_clusters_distance_transform_path,  WSI[:-4],
                                                                        WSI[:-4] + f'_{max_ep_cluster_area_pixels}.npy'),
                            max_ep_cluster_area_pixels=max_ep_cluster_area_pixels)

        # For each distance threshold, create a proximity mask and use it to filter stroma
        for dist in dist_list:

            print(WSI, max_ep_cluster_area_pixels, dist)

            # -----------------------------------------------------------------
            # Step 2b(i): Threshold the distance transform into a binary proximity mask
            # -----------------------------------------------------------------
            # Folder name encodes the parameters (area + dist) so downstream results
            # remain organized and reproducible.
            out_spatial_dist_mask_path_cur = os.path.join(out_spatial_dist_mask_path, f'{max_ep_cluster_area_pixels}area_{dist}dist')
            os.makedirs(out_spatial_dist_mask_path_cur, exist_ok=True)
            # Convert distance transform (.npy) -> binary mask image (.tif):
            #   1 where distance <= dist, else 0
            from tilseg.spatial import spatial_distance_masker
            spatial_distance_masker(dist_mask_path = os.path.join(out_stitched_filtered_epith_clusters_distance_transform_path,  WSI[:-4],
                                                                            WSI[:-4] + f'_{max_ep_cluster_area_pixels}.npy'),
                                    dist=dist,
                                    out_stitched_spatial_distance_mask_path = os.path.join(out_spatial_dist_mask_path, f'{max_ep_cluster_area_pixels}area_{dist}dist', WSI[:-4]+'.tif'))

            # -----------------------------------------------------------------
            # Step 2b(ii): Filter stroma to keep only stromal pixels within dist of epithelium
            # -----------------------------------------------------------------
            out_spatial_stroma_path_cur = os.path.join(out_stitched_filtered_stroma_path, f'{max_ep_cluster_area_pixels}area_{dist}dist')
            os.makedirs(out_spatial_stroma_path_cur, exist_ok=True)
            # stroma_filtered = stroma_mask AND proximity_mask
            from tilseg.spatial import spatial_stroma_filterer
            spatial_stroma_filterer(stitched_stroma_mask_path = os.path.join(stitched_stroma_dir_path, WSI),
                                    stitched_spatial_distance_mask_path = os.path.join(out_spatial_dist_mask_path,
                                                                                       f'{max_ep_cluster_area_pixels}area_{dist}dist',
                                                                                       WSI[:-4]+'.tif'),
                                    out_stitched_filtered_stroma_path = os.path.join(out_spatial_stroma_path_cur, WSI[:-4]+'.tif'))

        # ---------------------------------------------------------------------
        # Step 2c: Cleanup intermediate distance transform for this (WSI, area)
        # ---------------------------------------------------------------------
        # This .npy can be large; remove once all dist thresholds are generated.
        os.remove(os.path.join(out_stitched_filtered_epith_clusters_distance_transform_path,  WSI[:-4], WSI[:-4] + f'_{max_ep_cluster_area_pixels}.npy'))


# -----------------------------------------------------------------------------
# Step 3: Compute spatial TIL metrics for each (area threshold, distance threshold)
# -----------------------------------------------------------------------------
# For each parameter pair:
#   - Load the filtered stroma mask (stroma within dist of epithelial cluster).
#   - Load TIL contours (previously generated by global TILseg code).
#   - Keep only TIL contours whose centroid lies inside the filtered stroma.
#   - Compute:
#       tilarea   = number of pixels in filled TIL mask
#       stromaarea = number of nonzero pixels in filtered stroma
#       tilscore  = tilarea / stromaarea
#       tilcount  = number of retained TIL contours
#   - Save CSV with one row per WSI.
# -----------------------------------------------------------------------------
print("Step 3: Compute spatial sTIL scores")
for max_ep_cluster_area_pixels in max_ep_cluster_area_pixels_list:

    for dist in dist_list:
        print(f"Processing --> Epithelial cluster size threshold: {max_ep_cluster_area_pixels}, Distance threshold: {dist}")

        # Output CSV path encodes parameters for traceability
        os.makedirs(out_spatial_results_path, exist_ok=True)
        out_spatial_results_path_cur = os.path.join(out_spatial_results_path, f'spatial_tilscore_{max_ep_cluster_area_pixels}area_{dist}dist.csv')

        # Accumulators (one entry per WSI)
        tilscore_list = []
        tilcount_list = []
        tilarea_list = []
        stromaarea_list = []

        for WSI in WSIs:

            print(WSI, ".... Done")

            # Folder for filtered stroma masks for this parameter combo
            out_spatial_stroma_path_cur = os.path.join(out_stitched_filtered_stroma_path, f'{max_ep_cluster_area_pixels}area_{dist}dist', WSI[:-4]+'.tif')
            # Pickled global sTIL contours for this WSI
            out_stitched_filtered_tils_contours_path_cur = os.path.join(stitched_tils_path, WSI[:-4]+'.pkl')

            # Filtering out sTIL contours that fall outside of the spatially confined region & calculating the spatial sTIL score
            from tilseg.spatial import tilseg_score_calculator
            cur_tilscore, cur_tilcount, cur_tilarea, cur_stromaarea = tilseg_score_calculator(
                out_spatial_stroma_path_cur, out_stitched_filtered_tils_contours_path_cur)

            # Store metrics for this WSI
            tilscore_list.append(cur_tilscore)
            tilcount_list.append(cur_tilcount)
            tilarea_list.append(cur_tilarea)
            stromaarea_list.append(cur_stromaarea)

        # Save per-WSI spatial metrics table
        df = pd.DataFrame({'Directory': WSIs, 'tilscore': tilscore_list, 'tilcount': tilcount_list, 'tilarea': tilarea_list, 'stromaarea': stromaarea_list})
        df.to_csv(out_spatial_results_path_cur)


print("Spatial TILseg sTIL scores have been saved in the 'spatial_results' folder!")