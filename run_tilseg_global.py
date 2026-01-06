import os
import sys
from multiprocessing import Pool, cpu_count
import pickle

def main():
    mainPath = None
    while mainPath is None:
        check_path = input("Path to WSIs and their annotations: ")
        if os.path.exists(check_path):
            mainPath = check_path
        else:
            print("The path does not exist or is invalid. Please try again.")

    # Input Steps
    steps = None
    while steps is None:
        string = "Input steps number you would like to run:\n1. Extracting Patches from Annotations\n2. Implement: 3 class classifier\n3. Binary Stromal Mask\n4. Stromal Patches Multiplication\n5. Nuclear Segmentation Stardist & Filtering Contours\n6. Stitch WSI\n7. TIL score"
        print(string)
        user_input = input("Input as [1,2,3,4,5,6,7,8]: ")
        try:
            # Split the input by commas and try to convert each element to an integer
            steps_list = list(map(int, user_input.split(',')))
            # Check if the numbers are within a certain range
            if all(1 <= step <= 6 for step in steps_list):
                steps = steps_list  # Assign the valid steps and end the loop
            else:
                print("Please enter numbers between 1 and 6.")
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")

    ### 1. Extracting Patches from Annotations
    if 1 in steps:
        print("1. Extracting Patches from Annotations")
        from tilseg import extracting_patches
        extracting_patches.extracting_patches(mainPath)
        print("Done (1/8)")
    else:
        print("Skip: 1. Extracting Patches from Annotations (1/6)")

    # Get the list of items (files and directories) in the specified path
    items = os.listdir(mainPath)

    # Filter out directories from the list of items
    directories = [item for item in items if os.path.isdir(os.path.join(mainPath, item))]

    # Loop through each WSI directory
    for directory in directories:
        path = os.path.join(mainPath, directory)
        
        print("Running pipeline for WSI ", directory)

        ### 2. Implement: 3 class classifier
        if 2 in steps:
            print(directory, ": 2. Implement: 3 class classifier")
            from tilseg import implement
            implement.implement(path)
            print(directory, ": Done (2/8)")
        else:
            print("Skip: 2. Implement: 3 class classifier (2/8)")


        ### 3. Binary Stromal Mask
        if 3 in steps:            
            print(directory, ": 3. Binary Stromal Mask")
            from tilseg import binary_stromal_mask
            binary_stromal_mask.binary_stromal_mask(path)
            print(directory, ": Done (3/8)")
        

        ### 4. Stromal Patches Multiplication
        if 4 in steps:
            print(directory, ": 4. Stromal Patches Multiplication")
            from tilseg import stromal_patches_multiplication
            stromal_patches_multiplication.multiplying(path)
            print(directory, ": Done (4/8)")
    

    ### 5. Nuclear Segmentation & Filtering
    for directory in directories:
        path = os.path.join(mainPath, directory)
        if 5 in steps:
            # Pickle the dictionary to a file
            seg_path = os.path.join(path, f"{directory}_filtered_segmentations.pkl")
            if os.path.exists(seg_path) is False:
                print(directory, ": 5. Nuclear Segmentation & Filtering")
                from tilseg import nuclearseg
                segmentations, _ = nuclearseg.nuclearseg_he_wrapper(path, save_seg=True, save_image=False)

                from tilseg import filtering_contours
                til_contours = filtering_contours.filtering(path, segmentations)

                with open(seg_path, "wb") as f:
                    pickle.dump(til_contours, f)
                print(directory, ": Done (5/8)")


    ### 6. Erode sTILs 
    for directory in directories:
        path = os.path.join(mainPath, directory)
        if 6 in steps:
            print("6. Eroding sTILs and stitching patches into WSI")
            from tilseg import til_erosion

            # erode sTILs
            print("eroding sTILs")
            til_erosion.erode_overlaps(path, directory)

            print(directory, ": Done (6/8)")


    ### 7. Stitch WSI
    for directory in directories:
        path = os.path.join(mainPath, directory)
        if 7 in steps:
            print("7. Stitching patches into WSI")
            from tilseg import wsi_stitch

            # create parent folder for all stitched images
            stitching_dir = os.path.join(path, "stitching") 

            # stitch together the raw 3CC output
            print("stitching raw 3CC WSIs...")
            wsi_stitch.stitch_wsi(folder_path=mainPath,
                       out_dir=stitching_dir,
                       patch_type='3cc_raw')
            
            # stitch together the binary mask
            print("stitching binary masks...")
            wsi_stitch.stitch_wsi(folder_path=mainPath,
                       out_dir=stitching_dir,
                       patch_type='binary')
            
            # stitch together the final til mask with erosion
            print("stitching eroded til masks..")
            wsi_stitch.stitch_wsi(folder_path=mainPath,
                       out_dir=stitching_dir,
                       patch_type='final_til_mask_eroded')
            
            print(directory, ": Done (7/8)")

    ### 8. Global TILseg Scoring
    if 8 in steps:
        print("8. Global TIL score generating")
        from tilseg import til_score
        til_score.compute_global_tilseg_score(mainPath, directories)
        print("Done! (8/8)")


if __name__ == "__main__":
    main()