# TILseg: Segmentation of stromal Tumor Infiltrating Lymphocytes (sTILs) in TNBC

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
![TILseg Pipeline Workflow](images/tilseg_methods.jpg)

### Sample Results
![Visualization](images/tilseg_performance.jpg)

## METHODOLOGY

### 1. Patch Extraction

In `1_extracting_patches.py`, the annotation files (.XML) are used to parse the WSI into 3000x4000 pixel patches to reduce computational burdens in the pipeline. Patches which are mostly glass, background, or contain sparse tissue are filtered out of the analysis to further increase computational efficiency.

### 2. Tissue Classification

Once the WSI is parsed into patches, `2_implement.py` uses the trained tissue classifier model (3CC) to segment stromal areas, epithelial areas, and other tissue artifacts in a patch-wise manner.

### 3. Stromal Tissue Extraction
After segmenting the desired tissue area, `3.1_binary_stromal_mask.py` identifies the stromal tissue regions from the classified image and extracts it as a binary mask (patch-wise) for subsequent calculation of the sTIL score. Then, `3.2_stromal_patches_multiplication.py` renders the H&E RGB image in the stromal regions.


### 4. sTIL Nuclear Segmentation + Morphological Filtering

On the full H&E patch, `4.1_nuclearseg.py` segments nuclei across the entire tissue area using a pretrained [StarDist](https://github.com/stardist/stardist) model. Then, `4.2_filtering_contours.py` excludes nuclear segmentations that lie outside the stromal regions and filters out nuclei based on size (excludes larger epithelial and stromal cells) and roundness (excludes elongated fibroblast-like cells) filters. As a result, we remain only with sTILs nuclei for scoring.

### 5. sTILs Scoring

The final step, `5_til_score.py`, produces a whole-slide level sTILs score. This score is the ratio between the total area of sTILs (Step 4 output) divided by the total area of stroma (Step 5 output) aggregated across all patches extracted from the WSI.

The pipeline is also able to compute sTIL scores using the following parameters to isolate a subset of sTILs that are:

1. within a certain distance from epithelial regions
2. in proximity to epithelial clusters above a specified cluster size threshold

## INSTALLATION
### Create and Activate a Virtual Environment

Download current 3 class model [here](https://uwnetid-my.sharepoint.com/:u:/g/personal/bkha_uw_edu/EZltGcMIdEZGjZC9F6bkqIwBG9ZGyrnKOl0CMH1oIi1m3Q?e=hY63Hn)

Create a virtual environment:  
   ```bash
   python3 -m venv venv_tilseg
   ```

Activate the virtual environment:  
   - On macOS/Linux:  
     ```bash
     source venv_tilseg/bin/activate
     ```
   - On Windows (PowerShell):  
     ```powershell
     .\venv_tilseg\Scripts\activate
     ```

Install Tensorflow-GPU separately [here](https://neptune.ai/blog/installing-tensorflow-2-gpu-guide) and then install the rest of the required python packages using the following command:
```bash
pip install -r requirements.txt
```

### Install OpenSlide Python

Shortcut Instructions: (for more detailed instructions on installing OpenSlide, please click [here](https://openslide.org/api/python/#installing))

- On macOS/Linux:
```bash
  python3 -m pip install openslide-python
```
- On Windows: (adapted from [here](https://openslide.org/api/python/#installing))
    1. Download the [OpenSlide Windows binaries](https://openslide.org/download/#binaries)
    2. Extract them to a known path
    3. Import `openslide` by modifying path in [extracting_patches_from_annotated_folder_wise.py](tilseg2/extracting_patches_from_annotated_folder_wise.py)

    ```bash
    # The path can also be read from a config file, etc.
    OPENSLIDE_PATH = r'c:\path\to\openslide-win64\bin'
    import os
    if hasattr(os, 'add_dll_directory'):
        # Windows
        with os.add_dll_directory(OPENSLIDE_PATH):
            import openslide
    else:
        import openslide
    ```


## USAGE

### Data Preparation

For every WSI that needs to be scored, the user must provide a single annotation file (`.XML`) for each corresponding image file. We recommend annotating the tissue area that needs to be scored, and each annotation file can contain multiple annotations. 

Currently, TILseg is only able to parse annotation file inputs (`.XML`) from the following image viewing programs:
* [Aperio ImageScope](https://www.leicabiosystems.com/us/digital-pathology/manage/aperio-imagescope/)
* [Automated Slide Analysis Platform](https://computationalpathologygroup.github.io/ASAP) (ASAP).

### Execution
The pipeline can be directly run from the terminal/command line by first executing `run_tilseg.py` (On some systems, `python3` may need to be used instead of `python`):
```bash
python run_tilseg.py
```

Provide the input path containing the WSIs and their corresponding annotation files (.XML):   
```bash
Path to WSIs and their annotations: 'path/to/your/folder'
```

Select the steps of the pipeline you wish to run by inputting the number of the step(s) needed to run separated by commas. If the pipeline needs to be executed from end-to-end, the user should input the following sequence: `1,2,3.1,3.2,4,5`
```bash
Input steps number you would like to run:
1.    Extracting Patches from Annotations
2.    Implement: 3 class classifier
3.1.  Binary Stromal Mask
3.2.  Stromal Patches Multiplication
4.    Nuclear Segmentation (Stardist) & Filtering Contours
5.    TIL score
```

Once the desired steps have been selected, the TILseg scores will be output in the input directory as a .CSV file.
```bash
Output file: 'path/to/your/folder/tilscore.csv'
```

### Required Inputs
In a single folder (`path/to/your/folder`), please include the following:
* The H&E-stained whole slide images needed to be scored (`.SVS`).
* Corresponding annotations of the tissue area for each WSI (`.XML`). The name of the annotation file should exactly match the name of the WSI file.

Invalid Examples:

❌ `slide1.svs` and `Slide1.xml` (case mismatch)  
❌ `specimen_A.svs` and `specimen_A.json` (wrong annotation format)  
❌ `sample_05.svs` with no corresponding XML file

### Recommended Directory Structure for Execution
```text
path/to/your/folder/
│
├── slide_1.svs       # H&E-stained whole slide image
├── slide_1.xml       # Corresponding tissue annotation
├── slide_2.svs
├── slide_2.xml
└── ...               # Additional slide pairs
```

## TILseg Results for H&E-stained TNBC Core-needle biopsies Pre-Neoadjuvant Chemotherapy

For a cohort of 51 TNBC patients who received neoadjuvant chemotherapy, we ran their H&E-stained diagnostic biopsy section through the TILseg pipeline. For slide-level stroma & sTILs, we observed statistically significant stratification of good and poor response to therapy, as measured by recurrence-free survival (RFS). We also find that scoring only sTILs < 0.5 mm from epithelia has a poor stratification potential, most likely due to not accounting for cancer-interacting sTILs farther away from epithelia. On the contrary, scoring sTILs 1.0-1.5 mm from epithelia shows the best stratification of this cohort. 

**Global TILseg Patient Scores** 
![TILseg Stratifying TNBC Patient Cohort](images/discovery_biopsy_survival.jpg)
*(A) Kaplan-Meier stratification of RFS in the discovery cohort (n = 51) by TILseg scores. A threshold of ≥ 10% sTILs was used for manual assessment while a cutoff of ≥ 4.0% was used to stratify patients based on TILseg scores. Manual assessment was unable to stratify patient outcomes (p = n.s.) while the TILseg scores improved stratification and were statistically significant (p = 0.0231). (B) Patients in the discovery cohort who had recurred within 3.1 years (poor outcome, n = 13) had significantly lower median TILseg scores than patients who had not recurred at all after 3.1 years (good outcome, n = 38). (C) A multivariate Cox regression model was fitted on WSIs in the training and validation 1 subsets of the biopsy cohort (n = 43), showing TILseg as an independent prognostic factor (HR = 0.74, 95% CI [0.52-1.00], p = 0.0515) after adjusting for clinical tumor stage at the time of diagnosis. A second multivariate Cox regression model was fitted on WSIs in the training, validation 1, and validation 2 samples (n = 51). Consistent directional effects were observed where TILseg remained associated with improved RFS (HR = 0.74, 95% CI [0.53–1.02], p = 0.0640), further supporting its biological and clinical relevance.*

**Spatially Confined TILseg Patient Scores**
![TILseg Spatial Stratification](images/discovery_spatial_analysis_survival.jpg)
*(A) Heat map of log-rank p-values stratifying patients across the biopsy cohort at a threshold of 4% sTILs using spatial TILseg scores (n = 51). The spatial TILseg scores were calculated based on a subset of sTILs at distances up to 0.25–2.0 mm from epithelial cluster sizes equal to or greater than 800–8000 μm2. The total stromal area was derived from the stromal regions within the corresponding distance from an epithelial cluster. sTILs closer in proximity to cancerous areas (i.e., ≤ 0.5 mm) alone are insufficient when stratifying patients (log rank p = 0.057). Consideration of  sTILs at farther distances show greater prognostic value with respect to RFS (log rank p = 0.007), while including sTILs at intermediate distances exhibit the strongest association with RFS (log rank p = 0.004). (B) Kaplan-Meier stratification of recurrence-free survival (RFS) in the discovery cohort (n = 51) at a distance of 1.5 mm from epithelial clusters ≥ 6000 μm2 (log rank p = 0.0039) using a cutoff of 4% sTILs. (C) Boxplots (each with n = 51) showing the distribution of spatial TILseg scores across the biopsy cohort in good (n = 38) and poor outcome (n = 13) patient groups at 0.50, 0.75, 1.50, and 2.0 mm from epithelial clusters ≥ 6000 μm2 where each point represents a single patient. (D) Pairwise Wilcoxon signed-rank test results comparing TILseg scores between distances (0.5–2.0 mm) from epithelial clusters ≥ 6000 μm2 across good- and poor-outcome groups.*
