# TILseg: Segmentation of stromal Tumor Infiltrating Lymphocytes (sTILs) in TNBC

## CONTENTS
1. [About](#about)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Lisence](#license)

## 1. ABOUT


## 2. INSTALLATION
### Step 1: Create and Activate a Virtual Environment

Download current 3 class model [here](https://uwnetid-my.sharepoint.com/:u:/g/personal/bkha_uw_edu/EZltGcMIdEZGjZC9F6bkqIwBG9ZGyrnKOl0CMH1oIi1m3Q?e=hY63Hn)

1. **Create a virtual environment**  
   ```bash
   python3 -m venv venv_tilseg
   ```

2. **Activate the virtual environment**  
   - On macOS/Linux:  
     ```bash
     source venv_tilseg/bin/activate
     ```
   - On Windows (PowerShell):  
     ```powershell
     .\venv_tilseg\Scripts\activate
     ```

### Step 2: Install Tensorflow-GPU
https://neptune.ai/blog/installing-tensorflow-2-gpu-guide

### Step 3: Install Other Dependencies

Install required Python packages:  
```bash
pip install -r requirements.txt
```

### Step 3: Install OpenSlide Python

[**Detailed Download OpenSlide Python Instructions**](https://openslide.org/api/python/#installing)

*Shortcut Instructions:*

(not recommend if have not read the **detailed** [instruction above](https://openslide.org/api/python/#installing))
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

## 3. USAGE


## 4. LICENSE


