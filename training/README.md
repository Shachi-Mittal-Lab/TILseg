# TILseg Model Training

This section describes how to prepare annotation patches and train the 3-class tissue classifier used by TILseg.

## Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Data Preparation](#data-preparation)
4. [Patch Extraction](#patch-extraction)
5. [Dataset Splitting](#dataset-splitting)
6. [Model Training](#model-training)
7. [Outputs](#outputs)

## Overview

The training workflow uses annotated whole-slide images (WSIs) to create patch-level examples for three classes:

- stroma
- epithelium
- other (not including background)

These patches are split into training, validation, and test sets, then used to fine-tune a VGG19-based classifier.

## Requirements

Install the dependencies listed in the main README before running the training notebook or script.

Required packages include:

- Python 3.x
- TensorFlow / Keras
- OpenSlide
- OpenCV
- NumPy
- scikit-learn
- scikit-image
- Pillow
- Matplotlib

Please refer to `TILseg/README.md` for more info on installing the environment.

## Data Preparation

For each slide, provide:

- one whole-slide image: `.svs`
- one matching annotation file: `.xml`

The `.xml` file must have the same base name as the WSI.

Annotations should be exported from **Aperio ImageScope**, and each class should be assigned to a unique annotation ID.

### Recommended folder layout

```text
input_folder/
├── slide_001.svs
├── slide_001.xml
├── slide_002.svs
├── slide_002.xml
└── ...
```

## Training Patch Extraction

The patch extraction step reads each WSI and its matching XML file, then extracts tissue patches from the annotated regions by:

- parsing annotation coordinates from the XML file
- grouping overlapping annotations
- extracting patches at the selected pyramid level
- removing mostly white/background patches
- saving patches into class-specific folders

### Example configuration

```python
run_patch_extraction(
    folder_path=r"path/to/folder",
    id_folders=['id1_S', 'id2_E', 'id3_O'],
    patch_sizes=[(48, 48), (256, 256)],
    mlevel=0,
)
```

### Output structure

```text
input_folder/
├── id1_S/
│   └── slide_name/
│       ├── 48/
│       └── 256/
├── id2_E/
│   └── slide_name/
│       ├── 48/
│       └── 256/
└── id3_O/
    └── slide_name/
        ├── 48/
        └── 256/
```

## Dataset Splitting

After patch extraction, organize the patches into one folder per class, then split them into train/validation/test sets.

### Expected input layout

```text
data_dir/
├── stroma/
├── epithelium/
└── other/
```

### Example split

```python
for class_name in os.listdir(data_dir):
    split_data(class_name, data_dir, train_dir, val_dir, test_dir)
```

## Model Training

The classifier uses a VGG19 backbone with a 3-class softmax output.

### Model setup

- initialize the model with VGG19 weights
- load the pretrained 3-class model weights
- freeze most layers for transfer learning
- compile with Adam and categorical cross-entropy

### Training example

```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=callbacks
)
```

## Outputs

The training workflow produces:

- extracted patch images
- train / validation / test split folders
- a trained `.h5` model file
- training history plots

If you retrain the classifier, update the model path used by the TILseg inference pipeline in `tilseg/implement.py`.
