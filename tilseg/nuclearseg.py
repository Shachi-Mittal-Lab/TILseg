# imports

import cv2
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt
import os
import pickle
from skimage import io
from tqdm import tqdm

from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D

def normalize_predict(folder_path: str):

    detections_raw = {}
    model = StarDist2D.from_pretrained('2D_versatile_he', )

    patches_path = os.path.join(folder_path, "patches")
   
    patch_files = [f for f in os.listdir(patches_path) if f.endswith(".tif")]
    for filename in tqdm(patch_files, desc="Processing Patches", unit="patch"):
        patch_path = os.path.join(patches_path, filename)
        X = imread(patch_path)
        X = np.array(X)
        # print(f"Original image shape: {X.shape}")

        n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
        axis_norm = (0,1)   # normalize channels independently
        # axis_norm = (0,1,2) # normalize channels jointly
        if n_channel > 1:
            print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

        img = normalize(X, 1,99.8, axis=axis_norm)
        # print(f"Normalized image shape: {img.shape}")
        labels, details = model.predict_instances(img)

        detections_raw[filename] = {}
        detections_raw[filename]['labels'] = labels
        detections_raw[filename]['details'] = details

    return detections_raw


def extract_detections(detections_raw: dict, 
                       folder_path: str,
                       out_path: str = None,
                       save_seg: bool = False, 
                       save_image: bool = False):
    
    patches_path = os.path.join(folder_path, "patches")

    segmentations = {}

    for filename, data in detections_raw.items():
        
        patch_path = os.path.join(patches_path, filename) 
        details = data['details']

        contours = []

        # looping over each contour
        for i, coord in enumerate(details['coord']):

            x_coords = np.array(coord[1])
            y_coords = np.array(coord[0])

            # combine x and y coordinates into a single array
            points = np.vstack((x_coords, y_coords)).astype(np.int32).T

            # reshape into format for OpenCV contours
            contour = points.reshape((-1, 1, 2))

            contours.append(contour)
        
        segmentations[filename] = contours # save contours to dictionary

        if save_image:
            
            # draw contours into original image
            patch = cv2.imread(patch_path)
            cv2.drawContours(patch, contours, -1, (0, 255, 0), 3)
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

            # saving image
            # basename = os.path.splitext(os.path.basename(patch_path))[0]
            io.imsave(os.path.join(out_path, f'{filename}'), patch_rgb)
            # plt.imsave(os.path.join(out_path, f'{basename}_nuclearseg.jpg'), patch_rgb)

    if save_seg:
        folder_name = os.path.basename(folder_path)
        seg_path = os.path.join(folder_path, f"{folder_name}_segmentations.pkl")

        # Pickle the dictionary to a file
        with open(seg_path, "wb") as f:
            pickle.dump(segmentations, f)

        print("Segmentations have been saved!")

    return segmentations


def nuclearseg_he_wrapper(folder_path: str, 
                          out_path: str = None,
                          save_seg: bool = True,
                          save_image: bool = True):
    
    out_path = os.path.join(folder_path, "stardist")
    os.makedirs(out_path, exist_ok=True)
    detections_raw = normalize_predict(folder_path)
    segmentations = extract_detections(detections_raw,
                                       folder_path,
                                       out_path,
                                       save_seg,
                                       save_image)
    
    return segmentations, detections_raw