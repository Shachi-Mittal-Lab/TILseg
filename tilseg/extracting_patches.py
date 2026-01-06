import numpy as np
import skimage.io as io
import cv2 as cv
import os
import xml.etree.cElementTree as ET
import shutil
import pickle
from PIL import Image, ImageDraw

# Importing OpenSlide Package
cwd = os.getcwd()
OPENSLIDE_PATH = os.path.join(cwd,
                              'openslide-bin-4.0.0.6-windows-x64',
                              'bin')
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide  
import openslide

def parse_xml(anno_path, specs):
    '''
    Extracting coordinates of the annotations from an XML file.
    Built to be compatible with XML files annotated with:
        - Aperio ImageScope: Pathology Slide Viewing Software
        - ASAP: Automated Slide Analysis Platform
    '''
    factor = 2**specs['mlevel']
    tree = ET.ElementTree(file=anno_path)
    annolist = {}
    root = tree.getroot()
    i = 0
    if root.find("Annotation") is not None:
        print("Parsing XML from Aperio ImageScope!")
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
    elif root.iter('Coordinates') is not None:
        print("Parsing XML from ASAP")
        for coords in root.iter('Coordinates'):
            vasc = []
            for coord in coords:
                vasc.append((int(float(coord.attrib.get("X")) / factor), int(float(coord.attrib.get("Y")) / factor)))
            annolist[i] = vasc
            i += 1
    return annolist

def is_mostly_white(image, threshold):
    '''
    Calculating white threshold by applying binary ranges.
    REMOVING patches with percentage of white pixels GREATER OR EQUAL TO threshold.
    '''
    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Apply threshold to identify white pixels
    _, binary = cv.threshold(gray, 230, 255, cv.THRESH_BINARY)
    # Calculate the percentage of white pixels
    white_percentage = np.sum(binary == 255) / binary.size
    return white_percentage >= threshold

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


def extract_patches_with_padding(wsi, annolist, specs):
    '''
    Based on annotations from XML file, create rectangle bounding boxes.
        - If any boxes overlap to each other, draw one box over them.
        - If any boxes do not overlap, keep them as they are.
    For each bounding box,
        - Generate 3000x4000 patch with some % of overlap.
        - If patch is NOT MOSTLY WHITE, then save patch.
    '''
    colored = []
    coordinates = []
    # Create empty list to save chosen extracted patches
    patches = []

    # Load in Specifications
    patch_width, patch_height = specs['patch_size']
    padding = int(min(patch_width, patch_height) * (specs['padding_percent'] / 100))
    overlap = int(min(patch_width, patch_height) * (specs['overlap_percent'] / 100))
    factor = 2**specs['mlevel']

    # Create rectangle bounding boxes from XML annotations
    rects = []
    for _, coord in annolist.items():
        # Calculate the bounding rectangle for each set of coordinates
        x, y, w, h = cv.boundingRect(np.array(coord))
        # Save x, y, w, h for each rect into a list
        rects.append(Rectangle(x,y,x+w,y+h))  # (left, top, right, bottom)

    # handle overlap if there are multiple annotatinos
    if len(rects) == 1:
         groups = [[0]] # if there is only one annotation
    else: # TODO
        # group rectangles based on overlapping
        groups = find_groups(rects)  # if there are multiple annotations

    # Within each group, EXTRACT PATCHES
    for group_id in groups:
        combined_coords = []

        # Combined coordinates within the same group
        for id in group_id:
            combined_coords.extend(annolist[id])
        combined_coords = np.array(combined_coords)

        # scale annotations to match mlevel
        scaled_coords = (combined_coords / factor).astype(np.int32)

        # Make a rectangle bounding box
        x, y, w, h = cv.boundingRect(np.array(combined_coords))
        
        x_start, y_start = x, y
        x_end, y_end = x + w, y + h

        # create a mask for the annotation area
        anno_mask = np.zeros((h, w), dtype=np.uint8)
        shifted_coords = scaled_coords - [x, y]
        cv.fillPoly(anno_mask, [shifted_coords], 255)
        
        # Iterate through bounding box and extract patch
        while y_start < y_end:
            x_curr = x_start
            while x_curr < x_end:
                x_patch_start = max(x_curr - padding, 0)
                y_patch_start = max(y_start - padding, 0)
                x_patch_end = min(x_patch_start + patch_width, wsi.level_dimensions[0][0])
                y_patch_end = min(y_patch_start + patch_height, wsi.level_dimensions[0][1])
                
                patch_img_rgba = np.asarray(wsi.read_region((x_patch_start * factor, y_patch_start * factor), specs['mlevel'], (x_patch_end - x_patch_start, y_patch_end - y_patch_start)))
                patch_img = patch_img_rgba[:, :, :3]

                # is_colored = not is_mostly_white(patch_img, specs['threshold'])
                # # Check: If not mostly white, keep and save into 'patches'
                # if is_colored:
                #     patches.append(patch_img)
                
                # colored.append(is_colored)
                # coordinates.append((x_patch_start, y_patch_start))

                # Check if the patch is mostly white
                is_colored = not is_mostly_white(patch_img, specs['threshold'])

                # Compute mask slice coordinates relative to anno_mask
                y1 = y_patch_start - y
                y2 = y_patch_end - y
                x1 = x_patch_start - x
                x2 = x_patch_end - x

                # Clip to fit within bounds of anno_mask
                y1 = max(0, y1)
                y2 = min(anno_mask.shape[0], y2)
                x1 = max(0, x1)
                x2 = min(anno_mask.shape[1], x2)

                # slice the annotation mask 
                if y2 > y1 and x2 > x1:
                    anno_slice = anno_mask[y1:y2, x1:x2]
                    overlap_area = np.sum(anno_slice == 255)
                    total_patch_area = patch_width * patch_height
                    overlap_percentage = overlap_area / total_patch_area

                # Save the patch if it has >70% overlap with the annotation polygon
                if overlap_percentage > 0.7 and is_colored:
                    patches.append(patch_img)
                    coordinates.append((x_patch_start, y_patch_start))
                    colored.append(is_colored)

                x_curr += patch_width - overlap
            y_start += patch_height - overlap

    # pickle the colored boolean list + coordinates to a file
    stitch_info = [colored, coordinates]
    # wsi_folder = os.path.join(folder_path, slidename)
    # stitch_info_path = os.path.join(wsi_folder, f"stitch_info.pkl")
    # with open(stitch_info_path, "wb") as f:
    #     pickle.dump(stitch_info, f)

    return patches, stitch_info

def save_patches(patches, output_dir, slidename):
    '''
    Make output directory if not exist, and save all the patches
    '''
    os.makedirs(output_dir, exist_ok=True)
    for i, patch in enumerate(patches):
        save_as = os.path.join(output_dir, f'{slidename}_patch_position_{i}.tif')
        io.imsave(save_as, patch, check_contrast=False)

def generate_patches(slide_dir, slidename, wsi, output_dir, annopath, specs):
    '''
    For a SINGLE WSI,
        - Parse XML
        - Extract patches
        - Save patches
    '''
    # Parse XML file to get annotation coordinates
    annolist = parse_xml(annopath, specs)
    
    # Extracting patches, padding, overlapping, & eliminating white patches
    patches, stitch_info = extract_patches_with_padding(wsi, annolist, specs)
    
    save_patches(patches, output_dir, slidename)

    stitch_info_path = os.path.join(slide_dir, f"stitch_info.pkl")
    with open(stitch_info_path, "wb") as f:
        pickle.dump(stitch_info, f)

def extracting_patches_from_annotated_folder_wise(folder_path, specs, rerun):
    '''
    Load .svs and .xml files, and generate patches for EACH WSI folder wise.
    Checkpoints:
        - Run if no patches were generated for a WSI.
        - Rerun if indicated.
        - Skip if patches are already generated for a WSI.
    '''
    for wsi_file in os.listdir(folder_path):
        wsi_path = os.path.join(folder_path, wsi_file)
        if os.path.exists(wsi_path) is False:
            print('No WSI (.svs) is found!')
        elif wsi_file.endswith('.svs'):
            # Load WSI (.svs)
            wsi = openslide.OpenSlide(wsi_path)

            # Extract WSI slide name
            slidename = os.path.splitext(wsi_file)[0]
            # Finding corresponding XML file
            annotname = f"{slidename}.xml"
            annopath = os.path.join(folder_path, annotname)
            if os.path.exists(annopath) is False:
                print(f"There is no annotations for {slidename}")
            else:
                # Create output directory for WSI
                slide_dir = os.path.join(folder_path, f"{slidename}")
                output_dir = os.path.join(slide_dir, "patches")
                os.makedirs(output_dir, exist_ok=True)
                
                if os.listdir(output_dir) == []:
                    print(f"Extracting WSI {slidename}")
                    # Call the function to generate patches for the current pair of WSI and XML
                    generate_patches(slide_dir, slidename, wsi, output_dir, annopath, specs)
                elif rerun:
                    print(f"Re-extracting WSI {slidename}")
                    shutil.rmtree(output_dir, ignore_errors=True)
                    os.makedirs(output_dir)
                    generate_patches(slide_dir, slidename, wsi, output_dir, annopath, specs)
                else:
                    print(f"{slidename} has been extracted!")

def extracting_patches(folder_path, rerun=False):
    '''
    General function to start extraction;
    Adjustable specifications.
    '''
    specs = {
        "patch_size": (4000, 3000),
        "padding_percent": 5,
        "overlap_percent": 5,
        "mlevel": 0,
        "threshold": 0.7  # KEEP patches with up to 70% white
    }

    extracting_patches_from_annotated_folder_wise(folder_path, specs, rerun)