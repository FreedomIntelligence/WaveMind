# WaveMind Data Processing Documentation

## 📂 File Tree Structure
```bash
/path/to/WaveMind
├── data
│   ├── ImageNetEEG
│   │   ├── eeg_signals_raw_with_mean_std.pth  # Raw data needs to be downloaded
│   │   ├── Image/                              # Raw images need to be downloaded
│   │   └── process.py                          # Preprocessing script
│   ├── SEED
│   │   ├── Preprocessed_EEG/                   # Raw data needs to be downloaded
│   │   └── process.py                          # Preprocessing script
│   ├── THING-EEG
│   │   ├── Data/                               # Raw data needs to be downloaded
│   │   ├── data_config.json
│   │   ├── download.py
│   │   └── process.py                          # Preprocessing script
│   ├── Total
│   │   ├── CLIP_groundTruth/                   # Generated CLIP features
│   │   ├── data_label.h5                       # Generated HDF5 file
│   │   └── dataset_weights.pth                 # Auto-generated during training
│   ├── TUAB
│   │   ├── edf/                                # Raw data needs to be downloaded
│   │   ├── process_refine/                     # Temporary cache directory (auto-deleted after processing)
│   │   └── process.py                          # Preprocessing script
│   ├── TUEV
│   │   ├── edf/                                # Raw data needs to be downloaded
│   │   └── process.py                          # Preprocessing script
│   └── create_dataset_pkl.py                   # CLIP groundtruth generation script
```

## Data Processing Pipeline

### 1. Environment Setup

```bash
# Project root is auto-detected. For backward compatibility, you can set:
# export WaveMind_ROOT_PATH_=/path/to/WaveMind
```

### 2. Download Raw Data

Refer to the download instructions for each dataset (see detailed descriptions below).

### 3. Run Preprocessing Scripts

```bash
cd /path/to/WaveMind/data

# Process each dataset as needed
python ImageNetEEG/process.py
python THING-EEG/process.py
python SEED/process.py
python TUAB/process.py
python TUEV/process.py
```

### 4. Generate CLIP Groundtruth

```bash
# After all datasets are processed, generate CLIP features
python create_dataset_pkl.py
```

## HDF5 Data Specification

### 1. Key Naming Convention

| Dataset | Training Set Key | Test Set Key | Cross-Subject Key |
|---------|-----------------|--------------|-------------------|
| ImageNetEEG | `ImageNetEEG_train` | `ImageNetEEG_test` | `ImageNetEEG_cross` |
| THING-EEG | `thingEEG_train` | `thingEEG_test` | `thingEEG_cross` |
| SEED | `SEED_train` | `SEED_test` | `SEED_cross` |
| TUAB | `TUAB_train` | `TUAB_test` | `TUAB_cross` |
| TUEV | `TUEV_train` | `TUEV_test` | `TUEV_cross` |

**Note**: Different datasets use different naming conventions (maintained for historical compatibility).

### 2. Data Structure

Each sample is a structured array with the following fields:

```python
dtype = [
    ('eeg_data', np.float32, (32, 512)),    # EEG signal: 32 channels × 512 samples
    ('text_feature', np.float16, (768,)),   # CLIP embedding: 768-dimensional vector
    ('caption', 'S192'),                    # Text description: fixed 192 bytes
    ('label', dtype, shape),                # Label: integer or float
    ('image_path', 'S192')                  # Image path: fixed 192 bytes (optional)
]
```

### 3. Validate Data Integrity

```python
import h5py
import os
from data.Utils import get_wavemind_root

# Auto-detected path (or use WaveMind_ROOT_PATH_ env var for backward compatibility)
hdf5_path = os.path.join(get_wavemind_root(), 'data/Total/data_label.h5')

with h5py.File(hdf5_path, 'r') as f:
    print("HDF5 Dataset Keys:")
    for key in f.keys():
        print(f"  - {key}: {f[key].shape}")
```

Expected output:
```
HDF5 Dataset Keys:
  - ImageNetEEG_cross: (N,)
  - SEED_train: (N,)
  - SEED_test: (N,)
  - SEED_cross: (N,)
  - thingEEG_train: (N,)
  - thingEEG_test: (N,)
  - thingEEG_cross: (N,)
  - TUAB_train: (N,)
  - TUAB_test: (N,)
  - TUAB_cross: (N,)
  - TUEV_train: (N,)
  - TUEV_test: (N,)
  - TUEV_cross: (N,)
```

## CLIP Groundtruth Files

Location: `data/Total/CLIP_groundTruth/`

| File | Dimensions | Description |
|------|------------|-------------|
| `ImageNetEEG.pkl` | - | List of 40 class names |
| `thingEEG_closeset.npy` | (1573, 768) | Mean image features for closed-set objects |
| `thingEEG_closeset.pkl` | - | List of 1573 object names |
| `thingEEG.npy` | (200, 768) | Mean image features for zero-shot objects |
| `thingEEG.pkl` | - | List of 200 object names |
| `SEED.npy` | (3, 768) | Text features for 3 emotion classes |
| `SEED.pkl` | - | ['negative', 'neutral', 'positive'] |
| `TUAB.npy` | (2, 768) | Text features for 2 classes |
| `TUAB.pkl` | - | ['abnormal', 'normal'] |
| `TUEV.npy` | (6, 768) | Text features for 6 event types |
| `TUEV.pkl` | - | ['SPSW', 'GPED', 'PLED', 'EYEM', 'ARTF', 'BCKG'] |

## Data Modality Types

WaveMind supports two data modalities:

### Brain Cognition (Image-EEG Pairs)
- **Datasets**: ImageNetEEG, THING-EEG
- **CLIP Feature Source**: CLIP-ViT image encoder
- **Feature Type**: Image embedding (768-dim)
- **Save Method**: `Convert_and_Save.save_to_hdf5_new()`

### Brain State (Text-EEG Pairs)
- **Datasets**: SEED, TUAB, TUEV
- **CLIP Feature Source**: CLIP-BERT text encoder
- **Feature Type**: Text embedding (768-dim)
- **Save Method**: `Convert_and_Save.process_and_save()`

## Dataset Detailed Instructions

### THING-EEG
#### EEG Data
1. Download EEG data from https://huggingface.co/datasets/LidongYang/EEG_Image_decode/tree/main/Preprocessed_data_250Hz
2. Extract to `/path/to/WaveMind/data/THING-EEG/Data/Preprocessed_data_250Hz`

#### Paired Image Data
1. Download `training_images.zip` and `test_images.zip` from https://osf.io/y63gw/files
2. Extract to `/path/to/WaveMind/data/THING-EEG/Data/images_set/training_images` and `test_images`

### ImageNetEEG
Refer to https://github.com/perceivelab/eeg_visual_classification

### SEED
Refer to http://bcmi.sjtu.edu.cn/~seed/

### TUAB
Download from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml

### TUEV
Download from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml

## Data Statistics

We provide `ShowHDF5Statistic.ipynb` for displaying statistics of each dataset.

## Processing Script Notes

All `process.py` files have been optimized to:
- ✅ No redundant comments in code
- ✅ Unified error handling
- ✅ Clear documentation comments
- ✅ Consistent HDF5 save logic
- ✅ Automatic memory management and cleanup

See the `process.py` file in each dataset directory for details.
