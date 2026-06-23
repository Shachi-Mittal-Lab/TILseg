# TILseg: Automated sTIL Scoring Reveals Prognostic Patterns in TNBC

## QUICK START
1. Install the conda environment: `conda env create -f tilseg_tfGPU.yaml`
2. Activate it: `conda activate tilseg_tfgpu`
3. Place each `.SVS` slide and matching `.XML` annotation in the same folder.
4. Run the global pipeline:
   ```bash
   python run_tilseg_global.py

## CONTENTS
1. [About](#about)
2. [Methodology](#methodology)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Results](#results)

## ABOUT

Tumor infiltrating lymphocytes (TILs) are an increasingly important indicator of patient response to cancer. Higher TIL counts correlate with improved chances of survival in many cancer settings. Stromal TILs (sTILs) are a promising biomarker for patient prognosis. sTILs are conventionally scored manually, often by reporting them as a  a fraction of the total stromal compartment area. Manual scoring of sTILs form H&E-stained tissue sections in clinical practice is difficult to standardize and doesn’t take into account the entire whole slide image, limiting the precision and utility. The TILseg pipeline generates continous and granular patient-level sTIL scores in a standardized and automated manner.

Furthermore, the clinical utility of the spatial heterogenity of sTILs is unclear, and is not taken into account per the International TIL Working Group guidelines. The TILseg pipeline has the functionality to generate sTIL scores within spatially defined regions about cancer epithelia.

### Overall Workflow
![TILseg Pipeline Workflow](figures/methods.jpg)


### Repository Structure

```text
TILseg/
├── figures/
├── models/
├── openslide/
├── tilseg/
├── training/
├── run_tilseg_global.py
├── run_tilseg_spatial.py
└── tilseg_tfGPU.yaml
```

### Pipeline Overview
| Step | Module | Purpose | Input | Output |
|------|------------------|---------|-------|--------|
| 1 | `extracting_patches.py` | Split WSI into patches | `.SVS`, `.XML` | Patch images |
| 2 | `implement.py` | Tissue classification | Patches | 3-class tissue predictions |
| 3 | `binary_stromal_mask.py` | Extract stroma mask | Classified patches | Binary stromal mask |
| 4 | `stromal_patches_multiplication.py` | Mask H&E image to stroma | H&E patch + stroma mask | Stromal-only patch |
| 5 | `nuclearseg.py`, `filtering_contours.py` | Segment and filter sTIL nuclei | H&E patch | Filtered sTIL contours |
| 6 | `til_erosion.py` | Separate adjacent contours | Filtered contours | Eroded sTIL mask |
| 7 | `wsi_stitch.py` | Stitch patch outputs to generate WSI-level outputs | Patch-level outputs | WSI-level outputs |
| 8 | `til_score.py` | Compute global sTIL scores for all WSIs | Stroma + sTIL masks | `global_tilseg_scores.csv` |
| spatial | `spatial.py` | Compute spatial sTIL scores for all WSIs | Global outputs + thresholds | `spatial_results/` |


## METHODOLOGY

### **Global Scoring**

#### 1. Patch Extraction
In `extracting_patches.py`, the annotation files (.XML) are used to parse the WSI into 3000x4000 pixel patches to reduce computational burdens in the pipeline. Patches which are mostly glass, background, or contain sparse tissue are filtered out of the analysis to further increase computational efficiency.

**Input:** `.SVS` WSI + `.XML` annotation file
- **Processing:**
  - Parse annotation coordinates from Aperio ImageScope or ASAP XML files.
  - Group overlapping annotations so connected regions are processed together.
  - Generate patch coordinates from the annotated bounding regions.
  - Read patches from the WSI at the selected pyramid level.
  - Remove patches with a high proportion of white/background pixels.
- **Output:** Patch images saved in `patches/` and a `stitch_info.pkl` file containing patch coordinates and patch-validity information

#### 2. Tissue Classification
Once the WSI is parsed into patches, `implement.py` uses the trained tissue classifier model (3CC) to segment stromal areas, epithelial areas, and other tissue artifacts in a patch-wise manner.

- **Input:** Extracted patch images from `patches/`
- **Processing:**
  - Divide each patch into smaller tiles.
  - Remove empty or nearly empty tiles before inference.
  - Run the pretrained 3-class classifier on each tile.
  - Apply neighborhood-based smoothing to remove isolated mislabeled tiles.
  - Generate a colored class mask for each patch.
- **Output:**
  - `3class/` containing the original patch overlaid with the class prediction
  - `raw_3class/` containing the raw colored classification mask

#### 3. Stromal Tissue Extraction
After segmenting the desired tissue area, `binary_stromal_mask.py` identifies the stromal tissue regions from the classified image and extracts it as a binary mask (patch-wise) for subsequent calculation of the sTIL score. 

- **Input:** Classified patch images from `3class/`
- **Processing:**
  - Identify pixels corresponding to stromal tissue.
  - Convert stromal regions to foreground pixels.
  - Set non-stromal regions to background.
- **Output:** Binary stromal masks saved in `binary_masks/`

#### 4. Binary Mask Multiplication
Then, `stromal_patches_multiplication.py` renders the H&E RGB image only in the stromal regions using the binary mask generated in step 3. This isolates only the stromal region of the breast cancer tissue which will later be used to calculate the denominator of the sTIL score.

- **Input:** Original H&E patch from `patches/` and corresponding stromal mask from `binary_masks/`
- **Processing:**
  - Match each H&E patch to its stromal mask using the patch position index.
  - Multiply the H&E patch by the binary stromal mask.
  - Preserve only stromal pixels for later analysis.
- **Output:** Stromal-only H&E patches saved in `stromal_patches/`

#### 5. sTIL Nuclear Segmentation + Morphological Filtering
On the full H&E patch, `nuclearseg.py` segments nuclei across the entire tissue area using a pretrained [StarDist](https://github.com/stardist/stardist) model. 

- **Input:** H&E patches from `patches/`
- **Processing:**
  - Normalize patch intensities.
  - Run StarDist nucleus segmentation on each patch.
  - Save contour coordinates for each detected nucleus.
- **Output:**
  - Segmentation contours saved as a `.pkl` file
  - Optional visualization images in `stardist/`

Then, `filtering_contours.py` excludes nuclear segmentations that lie outside the stromal regions and filters out nuclei based on size (excludes larger epithelial and stromal cells) and roundness (excludes elongated fibroblast-like cells) filters. As a result, we remain only with sTILs nuclei for scoring.

- **Input:** StarDist contours + stromal patches from `stromal_patches/`
- **Processing:**
  - Compute contour area and perimeter.
  - Exclude nuclei outside biologically plausible size limits.
  - Exclude elongated objects using a roundness threshold.
  - Keep only contours that overlap stromal pixels.
  - Save filtered contour overlays and binary masks.
- **Output:**
  - Filtered contour overlays in `round_filtered/`, `stromal_filtered/`, and `final_til_contour/`
  - Binary filtered sTIL masks in `filtered_til_mask/`
  - Filtered contour dictionary for downstream use

#### 6. sTIL Segmentation Post-processing 
Once the sTILs have been segmented and filtered, `til_erosion.py` adds a 1-pixel gap between any adjacent contours patch-wise (on the binary mask of the filled sTIL contours). 

- **Input:** Filtered contour data from the previous step
- **Processing:**
  - Draw all contours onto a label mask.
  - Detect boundary pixels between adjacent contours.
  - Remove boundary pixels to create a small gap between touching regions.
  - Convert the result back into a binary mask.
- **Output:** Eroded sTIL masks saved in `filtered_til_mask_eroded/`

#### 7. WSI-level Stitching
To generate WSI-level outputs, `wsi_stitch.py` stitches all of patches together using the classified tissue output from step 2, the binary stromal masks from step 3, and the eroded binary sTIL masks. Since patches were originally extracted with padding, stitching them back together accounts for overlapping edges in the final calculations.

- **Input:**
  - Patch-level outputs from `patches/`, `3class/`, `binary_masks/`, `filtered_til_mask/`, or `filtered_til_mask_eroded/`
  - `stitch_info.pkl` containing patch coordinates and validity flags
- **Processing:**
  - Sort patches using their original coordinates.
  - Reassemble patch images into a slide-level canvas.
  - Account for padding and overlap during placement.
  - Resize stitched slides for storage and visualization.
- **Output:** Slide-level stitched images saved in `stitching/` subfolders such as `stitched_he/`, `stitched_3cc/`, `stitched_binary_stroma/`, and `stitched_filtered_til_mask_eroded/`

#### 8. Global sTILs Scoring
The final step, `til_score.py`, produces a whole-slide level sTILs score. This score is the ratio between the total area of sTILs (Step 4 output) divided by the total area of stroma (Step 5 output) aggregated across all patches extracted from the WSI. The scores for the input slide(s) are output as a single .CSV file within the input directory called `global_tilseg_scores.csv`.

- **Input:**
  - Stitched eroded sTIL mask
  - Stitched stromal mask
- **Processing:**
  - Count the number of nonzero sTIL pixels.
  - Count the number of stromal pixels.
  - Compute the global sTIL score as:

  $$\text{sTIL score} = \frac{\text{sTIL area}}{\text{stroma area}}$$

  - Optionally compute contour-based summary statistics for quality control.
- **Output:** `global_tilseg_scores.csv` saved in the input directory

### **Spatial Scoring**
The pipeline is also able to compute sTIL scores using the following parameters to isolate a subset of sTILs that are:
1. within a certain distance from epithelial regions
2. in proximity to epithelial clusters above a specified cluster size threshold

Once the parameters above are defined, spatial scoring is performed via a singular script, `spatial.py` in the following steps:

#### 1. Build epithelium masks
Using the final stitched outputs from the global scoring pipeline, the stitched 3CC prediction is used to generate a binary mask of only the epithelial regions (blue) and is saved for subsequent operations.

#### 2. Generate binary masks of spatially filtered (based on proximity to epithelial clusters) stroma.
To obtain only the stromal regions at a defined distance from epithelial clusters above a certain area threshold (using the epithelial mask output from step 1), the distance transform is calculated for every stromal and epithlial pixel pairing. Only stromal pixels within the defined distance are kept in the stromal binary distance mask.  

#### 3. Compute spatial TIL metrics for each (area threshold, distance threshold) pairing
Once the stromal binary distance mask is generated, then it is used to exclude sTIL contours whose centroids fall outside of the stromal regions. Then, the filtered contours are used to create a binary filtered sTIL mask. To calculate the spatial sTIL score using the defined epithelial area threshold and distance threshold, we divide the sTIL area (binary filtered sTIL mask) by the stromal area (stromal binary distance mask).

## INSTALLATION

Windows OS is supported for running the code. 

1. Install [Anaconda](https://www.anaconda.com/).

2. Clone this repository by typing `git clone https://github.com/Shachi-Mittal-Lab/TILseg.git` in an anaconda/command prompt

3. Create and run a virtual environment for this code:
From the TILseg directory run `conda env create -f tilseg_tfGPU.yaml`. To be able to use GPU for running inference for the CNN model, please install [tensorflow with GPU support](https://neptune.ai/blog/installing-tensorflow-2-gpu-guide).

4. Activate the conda environment with the installed packages and dependencies by typing `conda activate tilseg_tfgpu`.

## USAGE

### **Global Scoring**

#### *Data Preparation*

For every WSI that needs to be scored, the user must provide a single annotation file (`.XML`) for each corresponding image file. We recommend annotating the tissue area that needs to be scored, and each annotation file can contain multiple annotations. 

Currently, TILseg is only able to parse annotation file inputs (`.XML`) from the following image viewing programs:
* [Aperio ImageScope](https://www.leicabiosystems.com/us/digital-pathology/manage/aperio-imagescope/)
* [Automated Slide Analysis Platform](https://computationalpathologygroup.github.io/ASAP) (ASAP).

#### *Model Selection (optional)*
Currently, the default model has been defined as: `TILseg/models/3CC_discovery.h5`. If you plan on using this model, no further changes need to be made to `tilseg`. However, if a different model needs to be used, first upload the desired model to `TILseg/models`. Then, change the name to the desired model in line 61 of `implement.py` (located in `TILseg/tilseg`) and save the file before executing the pipeline.

#### *Execution*
The pipeline can be directly run from the terminal/command line by first executing `run_tilseg_global.py` (On some systems, `python3` may need to be used instead of `python`):
```bash
python run_tilseg_global.py
```

Provide the input path containing the WSIs (.SVS) and their corresponding annotation files (.XML):   
```bash
Path to WSIs and their annotations: 'path/to/your/folder'
```

Select the steps of the pipeline you wish to run by inputting the number of the step(s) needed to run separated by commas. If the pipeline needs to be executed from end-to-end, the user should input the following sequence: `1,2,3,4,5,6,7,8`
```bash
Input steps number you would like to run:
1. Extracting Patches from Annotations
2. Implement: 3 class classifier
3. Binary Stromal Mask
4. Stromal Patches Multiplication
5. Nuclear Segmentation (Stardist) & Filtering Contours
6. sTIL Erosion 
7. Stitching WSIs (raw 3CC output, binary stromal mask, binary eroded sTIL mask)
8. Global sTIL score
```

Once the desired steps have been selected, the TILseg scores will be output in the input directory as a single .CSV file.
```bash
Output file: 'path/to/your/folder/global_tilseg_scores.csv'
```

#### *Required Inputs*
In a single folder (`path/to/your/folder`), please include the following:
* The H&E-stained whole slide images needed to be scored (`.SVS`).
* Corresponding annotations of the tissue area for each WSI (`.XML`). The name of the annotation file should exactly match the name of the WSI file.

Invalid Examples:

❌ `slide1.svs` and `Slide1.xml` (case mismatch)  
❌ `specimen_A.svs` and `specimen_A.json` (wrong annotation format)  
❌ `sample_05.svs` with no corresponding XML file

#### *Recommended Directory Structure for Execution*
```text
path/to/your/folder/
│
├── slide_1.svs       # H&E-stained whole slide image
├── slide_1.xml       # Corresponding tissue annotation
├── slide_2.svs
├── slide_2.xml
└── ...               # Additional slide pairs
```

### **Spatial Scoring**

#### *Data Preparation*

The global scoring needs to be run prior to running the spatial scoring, which will ensure the necessary inputs required for the spatial pipeline.

#### *Execution*

The pipeline can be directly run from the terminal/command line by xecuting `run_tilseg_spatial.py` (On some systems, `python3` may need to be used instead of `python`):
```bash
python run_tilseg_spatial.py
```
Provide the input path containing the WSIs, their corresponding annotation files (.XML), and outputs of running the global TILseg pipeline:   
```bash
Path to WSIs and their annotations: 'path/to/your/folder'
```
Provide the list of epithelial cluster size filters (in pixels). For each iteration, epithelial clusters that are smaller than the respective filter size will be removed from the analyses to counter the effect of small, noisy epithelial predictions.
```bash
Input the list of minimum epithelial cluster size filters (in pixels) (e.g.: 7500,12500):
```

Also, provide the list of distances from the epithlial clusters (in pixels), within which to perform the sTIL scoring. For each iteration, only stroma within the specified distance of epithelial clusters will be scored. 
```bash
Input the list of maximum distances from epithelial clusters to score sTILs (in pixels) (e.g.: 79,198):
```

Spatial TILseg sTIL scores will then be generated and saved in the `spatial_results` folder within the main input directory.
