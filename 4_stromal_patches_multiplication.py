import os
import numpy as np
from skimage import io
from multiprocessing import Pool
import time
from concurrent.futures import ProcessPoolExecutor


def process_stromal_patch(file, in_dir, out_dir):
    wsi_code = os.path.basename(in_dir)
    PatchesPath = os.path.join(in_dir, 'patches')
    StromaBinaryMasksPath = os.path.join(in_dir, 'binary_masks')
    stroma_mask_path = os.path.join(StromaBinaryMasksPath, 'stroma_binary_mask_Classified_' + file)

    output_tif_path = os.path.join(out_dir, f'stromal_{file}')
    if os.path.exists(output_tif_path):
        print(f"Stromal mask for {file} is done.")
    else:
        if file.endswith('.tif') and os.path.exists(stroma_mask_path):
            # Read the patch and stroma binary mask
            patch = io.imread(os.path.join(PatchesPath, file))
            stroma_mask = io.imread(stroma_mask_path)  

            patch_stroma_mask_binary = np.zeros_like(patch)

            # Assuming stroma pixels are white in the binary mask
            patch_mask = np.all(stroma_mask == [255, 255, 255], axis=-1)

            # Create a 3D array with [1, 1, 1] for stroma pixels
            patch_stroma_mask_binary[patch_mask] = [1, 1, 1]

            # Multiply the binary mask and stroma and save as TIFF
            patch_stroma = np.uint8(patch_stroma_mask_binary * patch)
            io.imsave(output_tif_path, patch_stroma)

def multiplying(in_dir):
    # Create the output folder
    out_dir = os.path.join(in_dir, "stromal_patches")
    os.makedirs(out_dir, exist_ok=True)

    with ProcessPoolExecutor() as executor:
        wsi_code = os.path.basename(in_dir)
        PatchesPath = os.path.join(in_dir, 'patches')
        files = [file for file in os.listdir(PatchesPath) if file.endswith('.tif')]
        for file in files:
            executor.submit(process_stromal_patch, file, in_dir, out_dir)
    return None