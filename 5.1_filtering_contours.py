import os
from skimage import io
import cv2 as cv
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Importing OpenSlide Package
OPENSLIDE_PATH = r'd:\\github_repos\\TILseg2\\openslide-bin-4.0.0.6-windows-x64\\bin'
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide  
import openslide

def get_filtered_contours(orig_path, og_file, segmentations, round_filtered,
                         min_area, max_area, max_roundness):
    
    contours = segmentations[og_file]
    contours = [c.astype(np.int32) for c in contours]
    
    # Calculate areas and perimeters for all contours
    areas = np.array([cv.contourArea(c) for c in contours])
    perimeters = np.array([cv.arcLength(c, True) for c in contours])
    
    # Calculate roundness with protection against division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        roundness = np.where((areas != 0) & (perimeters != 0),
                           (perimeters ** 2) / (4 * np.pi * areas),
                           0)
    
    # Create mask for valid contours
    valid_mask = ((min_area < areas) & (areas < max_area) & 
                 (roundness < max_roundness) & (areas != 0) & (perimeters != 0))
    
    # Filter contours
    contours_filtered = [c for c, valid in zip(contours, valid_mask) if valid]
    
    # Only draw contours if needed for output
    if round_filtered:
        og_img_bgr = cv.imread(os.path.join(orig_path, og_file), cv.IMREAD_COLOR)
        og_img_rgb = cv.cvtColor(og_img_bgr, cv.COLOR_BGR2RGB)
        cv.drawContours(og_img_rgb, contours_filtered, -1, (0, 255, 0), 3)
        io.imsave(os.path.join(round_filtered, og_file), og_img_rgb)
    
    return contours_filtered


def stromal_filtering(contours_filtered, og_file, orig_path, stromal_path,
                     stromal_filtered, filtered_til_mask, final_til_contour):
    # Load images only if needed for output
    need_output = any([stromal_filtered, filtered_til_mask, final_til_contour])
    
    if need_output:
        og_img_bgr = cv.imread(os.path.join(orig_path, og_file))
        og_img_rgb = cv.cvtColor(og_img_bgr, cv.COLOR_BGR2RGB)
        stromal_bgr = cv.imread(os.path.join(stromal_path, f'stromal_{og_file}'), cv.IMREAD_COLOR)
        stromal_gray = cv.cvtColor(stromal_bgr, cv.COLOR_BGR2GRAY)
    else:
        stromal_gray = cv.imread(os.path.join(stromal_path, f'stromal_{og_file}'), cv.IMREAD_GRAYSCALE)
    
    # Pre-allocate mask
    mask = np.zeros(stromal_gray.shape[:2], dtype=np.uint8)
    contours_stromal_b = []
    
    for contour in contours_filtered:
        # Clear mask instead of recreating
        mask.fill(0)
        cv.drawContours(mask, [contour], -1, 255, thickness=cv.FILLED)
        
        total_pixels = cv.countNonZero(mask)
        if total_pixels == 0:
            continue
            
        nonzero_pixels = cv.countNonZero(cv.bitwise_and(stromal_gray, stromal_gray, mask=mask))
        if nonzero_pixels / total_pixels >= 0.5:
            contours_stromal_b.append(contour)
    
    # Only generate outputs if paths are provided
    if final_til_contour:
        og_img_bgr = cv.imread(os.path.join(orig_path, og_file))
        og_img_rgb = cv.cvtColor(og_img_bgr, cv.COLOR_BGR2RGB)
        cv.drawContours(og_img_rgb, contours_stromal_b, -1, (0, 255, 0), 3)
        io.imsave(os.path.join(final_til_contour, og_file), og_img_rgb)
    
    if filtered_til_mask:
        blank_img = np.zeros((3000, 4000, 3), dtype=np.uint8)
        cv.drawContours(blank_img, contours_stromal_b, -1, (255, 255, 255), thickness=cv.FILLED)
        io.imsave(os.path.join(filtered_til_mask, og_file), blank_img)
    
    return contours_stromal_b


def filtering(path, segmentations):
    
    # Getting MPP information
    min_diameter = 2  #µm
    max_diameter = 9  #µm
    
    wsi_path = path + ".svs"
    slide = openslide.OpenSlide(wsi_path)
    # Access MPP values
    mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, '0'))
    mpp_y = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, '0'))

    min_area = (min_diameter/2)**2 * 3.14 / (mpp_x * mpp_y)
    max_area = (max_diameter/2)**2 * 3.14 / (mpp_x * mpp_y)
    print(f'min area: {min_area}')
    print(f'max_area: {max_area}')


    if mpp_x < 0.3:  # Likely 40x
        max_roundness = 1.2
    else:  # Likely 20x
        max_roundness = 1.4
    print(f'roundness threshold: {max_roundness}')

    orig_path = os.path.join(path, "patches")
    # mask_path = os.path.join(path, "kmeans_binary_masks")
    stromal_path = os.path.join(path, "stromal_patches")

    round_filtered = os.path.join(path, "round_filtered")
    os.makedirs(round_filtered, exist_ok=True)

    stromal_filtered = os.path.join(path, "stromal_filtered")
    os.makedirs(stromal_filtered, exist_ok=True)
    filtered_til_mask = os.path.join(path, "filtered_til_mask")
    os.makedirs(filtered_til_mask, exist_ok=True)
    final_til_contour = os.path.join(path, "final_til_contour")
    os.makedirs(final_til_contour, exist_ok=True)
    
    def process_file(og_file):
        if og_file.endswith(".tif"):
            print(f'filtering {og_file}')
            contours_filtered = get_filtered_contours(orig_path, og_file, segmentations, 
                                                    round_filtered, min_area, max_area, max_roundness)
            contours_stromal_b = stromal_filtering(contours_filtered, og_file, orig_path, 
                                                 stromal_path, stromal_filtered, 
                                                 filtered_til_mask, final_til_contour)
            return og_file, contours_stromal_b
        return None
    
    # Process files in parallel
    til_contours = {}
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_file, os.listdir(orig_path))
        for result in results:
            if result:
                og_file, contours = result
                til_contours[og_file] = contours
                
    return til_contours