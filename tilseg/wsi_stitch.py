## IMPORTS

# Standard library imports
import os
import pickle
import re
import xml.etree.cElementTree as ET

# Third-party imports
import cv2 as cv
import numpy as np
from PIL import Image
import skimage.io as io
from tqdm import tqdm

# import openslide


def parse_xml(anno_path,
              factor):
    """
    Extracting coordinate information from annotations using a .xml file

    Args:
        anno_path (str): path to annotation file (.xml)
        factor (int): 

    Returns:
        annolist (list): list containing the coordinates of each annotation region (len(annolist) = # annotations)
    """
    tree = ET.ElementTree(file=anno_path)
    annolist = {}
    root = tree.getroot()
    
    if root.tag != "ASAP_Annotations": # annotations made using ImageScope
        i = 0
        for annotation in root.findall("Annotation"):
            for region in annotation.findall(".//Region"):
                vasc = []
                for vertex in region.findall(".//Vertex"):
                    try:
                        x = float(vertex.attrib.get("X"))
                        y = float(vertex.attrib.get("Y"))
                        vasc.append((int(x / factor), int(y / factor)))
                    except Exception as e:
                        print(f"Error parsing coordinates: {e}")
                        continue
                annolist[i] = vasc
                i += 1
        return annolist
    
    else: # annotations made using ASAP
        i = 0
        for coords in root.iter('Coordinates'):
            vasc = []
            for coord in coords:
                vasc.append((int(float(coord.attrib.get("X")) / factor), int(float(coord.attrib.get("Y")) / factor)))
            annolist[i] = vasc
            i += 1
        return annolist


def is_mostly_white(image, 
                    threshold: float = 0.7):
    """
    Extracting coordinate information from annotations using a .xml file

    Args:
        image (np.ndarray): an (w, h, 3) array containing the RGB values of each pixel for a patch
        threshold (float): the maximum percentage of white pixels to be considered a tissue patch

    Returns:
        (boolean): True if the patch is above the white_percentage threshold and False otherwise 
    """
    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Apply threshold to identify white pixels
    _, binary = cv.threshold(gray, 230, 255, cv.THRESH_BINARY)
    # Calculate the percentage of white pixels
    white_percentage = np.sum(binary == 255) / binary.size
    
    return white_percentage >= threshold


def get_patches(wsi_folder,
                patch_type):
    """
    Retrieve patches from 'patches' folder for each slide

    Args:
        wsi_path (str): List of valid patches.
        patch_type (str): The key indicating the type of patches that need to be stitched together

    Returns:
        patches (list): An list containing the patches which are stored as np arrays.
    """
    # validate patch_type
    valid_patch_types = {'he', '3cc', 'final_tils', 'final_til_mask_eroded', 'stroma_mask', '3cc_raw', 'tils', 'binary'}
    if patch_type not in valid_patch_types:
        raise ValueError(f"Invalid patch_type '{patch_type}'. Valid options are: {valid_patch_types}")
    
    # retreive the appropriate patches folder
    folder_map = {
        'he': 'patches',
        '3cc': '3class',
        'final_tils': 'final_til_contour',
        'final_til_mask_eroded': 'filtered_til_mask_eroded',
        'stroma_mask': 'binary_masks',
        '3cc_raw': 'raw_3class',
        'tils': 'filtered_til_mask',
        'binary': 'binary_masks'
    }
    patches_folder = os.path.join(wsi_folder, folder_map[patch_type])
    
    # Filter files: Only .tif files that match the 'position_<number>.tif' pattern
    # this is because of hidden files like Thumb.db, DS.store, etc.
    filtered_filenames = [
        filename for filename in os.listdir(patches_folder)
        if filename.endswith('.tif')
    ]

    # sort filenames based on the number in 'position_<number>.tif'
    sorted_filenames = sorted(
        filtered_filenames,
        key=lambda filename: int(re.search(r'position_(\d+)', filename).group(1))  # Sort by the number extracted
    )

    # read patches into a np.array for later usage
    patches = []
    patch_sizes = []  # List to store patch sizes
    for filename in sorted_filenames:
        patch_path = os.path.join(patches_folder, filename)
        patch_bgr = cv.imread(patch_path)
        patch_rgb = cv.cvtColor(patch_bgr, cv.COLOR_BGR2RGB)
        patches.append(patch_rgb)

        # Get the size of the patch (width, height)
        patch_sizes.append((patch_rgb.shape[1], patch_rgb.shape[0]))  # (width, height)
    
    return patches, patch_sizes

class Rectangle:
    '''
    Define rectangle coordinates & Overlapping conditions.
    '''
    def __init__(self, x1, y1, x2, y2):
        self.left = x1
        self.top = y1
        self.right = x2
        self.bottom = y2

    def overlaps(self, other):
        # return TRUE if overlaps, FALSE if not overlaps
        return not (self.right < other.left or  # Other to the right of Self
                    self.left > other.right or  # Other to the left of Self
                    self.bottom < other.top or  # Other to the bottom of Self
                    self.top > other.bottom)    # Other to the top of Self


def find_groups(rectangles):
    '''
    Group overlapping rectangles.
    '''
    n = len(rectangles)
    adjacency_list = {i: [] for i in range(n)}
    
    # For each rect, which other rect is overlapped?
    for i in range(n):
        for j in range(i + 1, n):
            if rectangles[i].overlaps(rectangles[j]):
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)
    
    # Group the rectangles
    visited = [False] * n  
    groups = []
    
    def dfs(i, group):
        visited[i] = True
        group.append(i)
        for neighbor in adjacency_list[i]:
            if not visited[neighbor]:
                dfs(neighbor, group)
    
    for i in range(n):
        if not visited[i]:
            group = []
            dfs(i, group)
            groups.append(group)
    
    return groups


def extract_patches(wsi,
                    annolist, 
                    patch_size, 
                    folder_path,
                    slidename,
                    mlevel=0,
                    padding_percent=5, 
                    overlap_percent=5):
    """
    Retrieves and saves the coordinate information of all patches as well as a boolean mask indicating which
    tissue patches are below the white pixel threshold (as a .pkl file).

    Args:
        wsi (np.ndarray): an array containing the loaded WSI (RGB)
        annolist (list): list containing the coordinates of each annotation region (len(annolist) = # annotations).
        patch_size (tuple): a tuple (w, h) indicating the dimensions of the patch.
        folder_path (str): path to the directory containing all the .svs and .xml files (as well as the results folders if applicable).
        slidename (str): the ID of the .svs file.
        mlevel (int): the desired pyramid level for processing patches. A higher mlevel indicates a more downsampled image.
        padding_percent (int): the area extracted around the patch as a percentage of the original patch dimensions. 
        overlap_percent (int): the area around the patch that should overlap with adjacent patches.
    Returns:
        colored (list): a boolean list of all patches where True=tissue patches and False=mostly white patches.
        coordinates (list): list of the coordinates (upper left corner) for every patch (tissue and white).
    """
    colored = []
    coordinates = []

    patch_width, patch_height = patch_size
    padding = int(min(patch_width, patch_height) * (padding_percent / 100))
    overlap = int(min(patch_width, patch_height) * (overlap_percent / 100))
    factor = 2 ** mlevel
    
    # Create rectangle bounding boxes from XML annotations
    rects = []
    for _, coord in annolist.items():
        # Calculate the bounding rectangle for each set of coordinates
        x, y, w, h = cv.boundingRect(np.array(coord))
        # Save x, y, w, h for each rect into a list
        rects.append(Rectangle(x,y,x+w,y+h))  # (left, top, right, bottom)

    # Group rectangles based on overlapping
    groups = find_groups(rects)  # Save as 

    # Within each group, EXTRACT PATCHES
    for group_id in groups:
        combined_coords = []

        # Combined coordinates within the same group
        for id in group_id:
            combined_coords.extend(annolist[id])
        combined_coords = np.array(combined_coords)
        x, y, w, h = cv.boundingRect(np.array(combined_coords))
        
        x_start, y_start = x, y
        x_end, y_end = x + w, y + h
        
        while y_start < y_end:
            x_curr = x_start
            while x_curr < x_end:
                x_patch_start = max(x_curr - padding, 0)
                y_patch_start = max(y_start - padding, 0)
                x_patch_end = min(x_patch_start + patch_width, wsi.level_dimensions[0][0])
                y_patch_end = min(y_patch_start + patch_height, wsi.level_dimensions[0][1])
                
                #patch_img = wsi.read_region((x_patch_start * factor, y_patch_start * factor), mlevel, (patch_width, patch_height))
                patch_img_rgba = np.asarray(wsi.read_region((x_patch_start * factor, y_patch_start * factor), mlevel, (x_patch_end - x_patch_start, y_patch_end - y_patch_start)))
                patch_img = patch_img_rgba[:, :, :3]
                
                is_colored = not is_mostly_white(patch_img)
                colored.append(is_colored)
                coordinates.append((x_patch_start, y_patch_start))
                
                x_curr += patch_width - overlap
            y_start += patch_height - overlap

    # pickle the colored boolean list + coordinates to a file
    wsi_folder = os.path.join(folder_path, slidename)
    stitch_info_path = os.path.join(wsi_folder, f"stitch_info.pkl")
    stitch_info = [colored, coordinates]
    with open(stitch_info_path, "wb") as f:
        pickle.dump(stitch_info, f)
    
    return colored, coordinates


def get_stitch_info(folder_path, 
                    slidename, 
                    extract_patches_func, 
                    *args, 
                    **kwargs):
    """
    Retrieves and saves the coordinate information and boolean mask of patches above the white threshold. 
    It calls on the extract_patches function and saves that information if the stitch_info.pkl file doesn't 
    already exist. Otherwise, it will retrieve that information if the .pkl already does exist.

    Args:
        folder_path (str): path to the directory containing all the .svs and .xml files (as well as the results folders if applicable)
        slidename (str): the ID of the .svs file 
        extract_patches_func (func): function for extracting patch coordinates + boolean mask
        *args (int): arguments for extract_patches
        **kwargs (str): keyword arguments for extract_patches
    Returns:
        colored (list): a boolean list of all patches where True=tissue patches and False=mostly white patches.
        coordinates (list): list of the coordinates (upper left corner) for every patch (tissue and white).
    """
    # define the path to the stitch_info.pkl file
    wsi_folder = os.path.join(folder_path, slidename)
    stitch_info_path = os.path.join(wsi_folder, "stitch_info.pkl")

    # check if the .pkl file exists
    if os.path.isfile(stitch_info_path):
        print(f"Loading stitch_info from {stitch_info_path}.")
        with open(stitch_info_path, "rb") as f:
            stitch_info = pickle.load(f)
        colored, coordinates = stitch_info
    else:
        print("stitch_info.pkl not found. Extracting patch info...")
        # Remove duplicate arguments from kwargs
        kwargs.pop("folder_path", None)
        kwargs.pop("slidename", None)

        # Call the extract_patches function to generate the data
        colored, coordinates = extract_patches_func(folder_path=folder_path, slidename=slidename, *args, **kwargs)

        # Save the data for future use
        os.makedirs(wsi_folder, exist_ok=True)
        with open(stitch_info_path, "wb") as f:
            pickle.dump([colored, coordinates], f)
        print(f"Stitch info saved to {stitch_info_path}.")

    return colored, coordinates


def pad_patch(patch, 
              target_height, 
              target_width):
    current_height, current_width = patch.shape[:2]
    if current_height < target_height or current_width < target_width:
        padding_height = target_height - current_height
        padding_width = target_width - current_width

        # Pad the patch with zeros (or a specific value if needed)
        patch = np.pad(patch, ((0, padding_height), (0, padding_width), (0, 0)), mode='constant', constant_values=0)
    
    return patch


def stitch_patches(patches,
                   colored_boolean,
                   coordinates,
                   patch_sizes,
                   padding_percent=5,
                   overlap_percent=5):
    """
    Restitch patches into a single WSI while inserting black patches for invalid regions.

    Args:
        patches (list): List of valid patches.
        colored_boolean (list): Boolean list indicating validity of each patch.
        coordinates (list): List of (x, y) coordinates for all patches.
        patch_size (tuple): Size of each patch (width, height).

    Returns:
        wsi (np.ndarray): Restitched image with black patches for invalid regions.
    """
    # get coordinates of only the colored patches
    colored_coordinates = [coord for coord, is_colored in zip(coordinates, colored_boolean) if is_colored]
    if not colored_coordinates:
        raise ValueError("No colored patches found.")
    
    # calculate grid dimensions according to colored patches
    x_coords = [coord[0] for coord in colored_coordinates]
    y_coords = [coord[1] for coord in colored_coordinates]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    print(f"Grid dimensions: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}")

    # create a blank image for the WSI
    max_patch_w = max(patch_sizes, key=lambda size: size[0])[0]  # max patch width
    max_patch_h = max(patch_sizes, key=lambda size: size[1])[1]  # max patch height
    wsi_w = max_x - min_x + max_patch_w  # Accounts for patches on the far right
    wsi_h = max_y - min_y + max_patch_h  # Accounts for patches on the bottom
    wsi = np.zeros((wsi_h, wsi_w, 3), dtype=np.uint8)
    weight_matrix = np.zeros((wsi_h, wsi_w), dtype=np.float32)

    print(f"WSI size: {wsi.shape}")
    print(f"Total patches: {len(colored_boolean)}")

    # stitch patches onto blank WSI
    patch_index = 0  # track colored patches
    for colored, (x, y) in tqdm(zip(colored_boolean, coordinates)):
        if not colored:
            continue  # skip non-colored patches
        
        # don't exceed the number of available patches
        if patch_index >= len(patches):
            print(f'Warning, there are only {len(patches)} patches available but {sum(colored_boolean)} total patches')
            break
        
        patch_w, patch_h = patch_sizes[patch_index]

        # calculate padding
        # padding = int(min(patch_w, patch_h) * (padding_percent / 100))
        padding_w = int(patch_w * (padding_percent / 100))
        padding_h = int(patch_h * (padding_percent / 100))

        # calculate overlap
        overlap_w = int(patch_w * (overlap_percent / 100))
        overlap_h = int(patch_h * (overlap_percent / 100))

        # adjust relative position: add padding but only subtract overlap for shift
        x_rel = x - min_x + padding_w - overlap_w
        y_rel = y - min_y + padding_h - overlap_h

        # ensure x_rel and y_rel are non-negative
        # x_rel = max(x_rel, 0)
        # y_rel = max(y_rel, 0)
        print(f"x={x}, min_x={min_x}, padding={padding_w}, overlap={overlap_w}")
        print(x_rel, y_rel)
        
        # load patch
        patch = patches[patch_index]

        # pad patch if the size is less than maximum patch size (x,y)
        if patch.shape[0] < max_patch_h or patch.shape[1] < max_patch_w:
            print(f"Padding patch {patch_index} to max size ({max_patch_h}, {max_patch_w})")
            patch = pad_patch(patch, max_patch_h, max_patch_w)  # Pad the patch to the max size

        # define region for placing the patch
        x_end = x_rel + patch.shape[1]
        y_end = y_rel + patch.shape[0]

        # ensure that the patch does not go out of bounds
        if x_end > wsi.shape[1] or y_end > wsi.shape[0]:
            raise ValueError(f"Patch placement goes out of bounds. x_end: {x_end}, y_end: {y_end}")

        # place patch onto blank WSI w/blending
        wsi[y_rel:y_end, x_rel:x_end] = (wsi[y_rel:y_end, x_rel:x_end] * weight_matrix[y_rel:y_end, x_rel:x_end, None] + patch) // (weight_matrix[y_rel:y_end, x_rel:x_end, None] + 1)
        weight_matrix[y_rel:y_end, x_rel:x_end] += 1

        patch_index += 1

    return wsi


def resize_stitched_wsi(wsi,
                        resize_factor=1):
    # get dimensions
    original_h, original_w = wsi.shape[:2]
    new_w = int(original_w * resize_factor)
    new_h = int(original_h * resize_factor)

    # Resize using OpenCV
    resized_wsi = cv.resize(wsi, (new_w, new_h), interpolation=cv.INTER_LANCZOS4)

    return resized_wsi


def save_stitched_wsi(wsi, 
                      folder_path,
                      out_dir, # TODO
                      slidename,
                      patch_type):
    
    """
    Saves the WSI to the appropriate output directory depending on if the stitched image is of the H&E,
    3CC output, or final TILs contours. Note that the stitched images of all proccessed WSIs will be saved
    in one directory.

    Args:
        wsi (np.ndarray): array of the stitched WSI to be saved.
        folder_path (str): path to the directory containing all the .svs and .xml files (as well as the results folders if applicable)
        slidename (str): the ID of the .svs file.
        patch_type (str): the key indicating what type of patches are being stitched. The options are: the original H&E,
        3CC output, and final TILs contours patches.
    Returns:
        None
    """
    # Convert the image from BGR to RGB
    wsi_rgb = cv.cvtColor(wsi, cv.COLOR_BGR2RGB)

    # validate patch_type
    valid_patch_types = {'he', '3cc', 'final_tils', 'final_til_mask_eroded', 'stroma_mask', '3cc_raw', 'tils', 'binary'}
    if patch_type not in valid_patch_types:
        raise ValueError(f"Invalid patch_type '{patch_type}'. Valid options are: {valid_patch_types}")
    
    # create the appropriate output folder
    folder_map = {
                'he': 'stitched_he',
                '3cc': 'stitched_3cc',
                'final_tils': 'stitched_final_tils',
                'final_til_mask_eroded': 'stitched_filtered_til_mask_eroded',
                'stroma_mask': 'stitched_stroma_mask',
                '3cc_raw': 'stitched_3cc_raw',
                'tils': 'stitched_tils_binary',
                'binary': 'stitched_binary_stroma'
            }

    out_stitch_dir = os.path.join(out_dir, folder_map[patch_type])
    os.makedirs(out_stitch_dir, exist_ok=True)
    out_path = os.path.join(out_stitch_dir, f"{slidename}.tif")
    
    # Save the image as a .tif file in RGB format
    cv.imwrite(out_path, wsi_rgb)


# wrapper function for stitching patches
def stitch_wsi(folder_path,
               out_dir, 
               patch_type='he'):
    """
    The wrapper function performs the following steps:
    1. Checks if the stitched image already exists (this may be necessary if the kernel times out in the middle of the run and you don't want to create new directories)
    2. Loads the WSI using OpenSlide
    3. Retrieves the annotation list from the .xml file
    4. Extracts the coordinate information for stitching and saves it if the .pkl file doesn't already exist
    5. Stitches the WSI together using the coordinate information and patch boolean mask
    6. Resizing the WSI (default is 0.5)
    7. Saves the stitched WSI

    Args:
        folder_path (str): path to the directory containing all the .svs and .xml files (as well as the results folders if applicable)
        patch_type (str): the key indicating what type of patches are being stitched. The options are: the original H&E,
        3CC output, and final TILs contours patches.
    Returns:
        None
    """
    mlevel = 0
    factor = 2 ** mlevel
    patch_size = (4000, 3000)
    padding_percent = 5
    overlap_percent = 5

    for filename in os.listdir(folder_path):
        if filename.endswith(".svs"):

            # extract slide name (without extension)
            slidename = os.path.splitext(filename)[0]

            # verifying if the stiched image already exists
            # create the appropriate output folder
            folder_map = {
                'he': 'stitched_he',
                '3cc': 'stitched_3cc',
                'final_tils': 'stitched_final_tils',
                'stroma_mask': 'stitched_stroma_mask',
                '3cc_raw': 'stitched_3cc_raw',
                'tils': 'stitched_tils_binary',
                'binary': 'stitched_binary_stroma'
            }
            # TODO redoing patch_path
            # out_dir = os.path.join(folder_path, folder_map[patch_type])
            # out_path = os.path.join(out_dir, f"{slidename}.tif")
            out_stitch_dir = os.path.join(out_dir, folder_map[patch_type])
            out_path = os.path.join(out_stitch_dir, f"{slidename}.tif")
            if os.path.exists(out_path):
                print(f"Stitched WSI already exists for {slidename}, skipping.")
                continue
            
            print(f"Opening {slidename}")
            # load the WSI using OpenSlide
            wsi_path = os.path.join(folder_path, filename)
            wsi = openslide.OpenSlide(wsi_path)
            
            # get the corresponding XML annotation filename + file path
            annotname = f"{slidename}.xml"
            annopath = os.path.join(folder_path, annotname)
            annolist = parse_xml(annopath, factor)

            # retreive patches + coordinate info for stitching
            wsi_folder = os.path.join(folder_path, slidename) # folder containing outputs
            patches, patch_sizes = get_patches(wsi_folder,
                                  patch_type)
            print(f'number of colored patches: {len(patches)}')
            print(f'colored patch sizes: {len(patch_sizes)}')
            
            colored, coordinates = get_stitch_info(folder_path=folder_path,
                                                   slidename=slidename,
                                                   extract_patches_func=extract_patches,
                                                   wsi=wsi,
                                                   annolist=annolist,
                                                   patch_size=patch_size,
                                                   mlevel=mlevel,
                                                   padding_percent=padding_percent,
                                                   overlap_percent=overlap_percent)

            
            # stitch patches together
            print("Stitching WSI now!")
            wsi = stitch_patches(patches,
                                 colored,
                                 coordinates,
                                 patch_sizes,
                                 padding_percent,
                                 overlap_percent)
            
            # resizing the WSI
            resized_wsi = resize_stitched_wsi(wsi)
            
            # save the stitched WSI
            save_stitched_wsi(resized_wsi, folder_path, out_dir, slidename, patch_type) # TODO
    