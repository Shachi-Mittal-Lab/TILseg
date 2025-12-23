import os
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
        string = "Input steps number you would like to run:\n1. Extracting Patches from Annotations\n2. Implement: 3 class classifier\n3. Binary Stromal Mask\n4. Stromal Patches Multiplication\n5. Nuclear Segmentation Stardist & Filtering Contours\n6. TIL score"
        print(string)
        user_input = input("Input as [1,2,3,4,5,6]: ")
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

    # 1. Extracting Patches from Annotations
    if 1 in steps:
        print("1. Extracting Patches from Annotations")
        from tilseg2 import extracting_patches_from_annotated_folder_wise
        extracting_patches_from_annotated_folder_wise.extracting_patches(mainPath)
        print("Done (1/6)")
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

        # * Reorganize Patches
        if 2 in steps:
            # 2. Implement: 3 class classifier
            print(directory, ": 2. Implement: 3 class classifier")
            from tilseg2 import implement
            implement.implement(path)
            print(directory, ": Done (2/6)")
        else:
            print("Skip: 2. Implement: 3 class classifier (2/7)")

        if 3 in steps:            
            # 7. Binary Stromal Mask
            print(directory, ": 3. Binary Stromal Mask")
            from tilseg2 import binary_stromal_mask
            binary_stromal_mask.binary_stromal_mask(path)
            print(directory, ": Done (3/6)")
        
        if 4 in steps:
            # 4. Stromal Patches Multiplication
            print(directory, ": 4. Stromal Patches Multiplication")
            from tilseg2 import stromal_patches_multiplication
            stromal_patches_multiplication.multiplying(path)
            print(directory, ": Done (4/6)")

    for directory in directories:
        path = os.path.join(mainPath, directory)
        
        print("Running pipeline for WSI ", directory)
            
    for directory in directories:
        path = os.path.join(mainPath, directory)
        if 5 in steps:
            # Pickle the dictionary to a file
            seg_path = os.path.join(path, f"{directory}_filtered_segmentations.pkl")
            if os.path.exists(seg_path) is False:
                # 6. Nuclear Segmentation Stardist
                print(directory, ": 5. Nuclear Segmentation Stardist")
                from tilseg2 import nuclearseg
                segmentations, _ = nuclearseg.nuclearseg_he_wrapper(path, save_seg=True, save_image=False)
                # print(directory, ": Done (5/6)")

                from tilseg2 import filtering_contours
                til_contours = filtering_contours.filtering(path, segmentations)

                with open(seg_path, "wb") as f:
                    pickle.dump(til_contours, f)
                print(directory, ": Done (5/6)")

    if 6 in steps:
        print("6. TIL score generating")
        from tilseg2 import til_score
        til_score.til_score_generator(mainPath)
        print("Done (6/6)")


if __name__ == "__main__":
    main()