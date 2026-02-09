# FOCUS

This repository contains the implementation of **FOCUS**, our noise-aware geospatial
deep learning framework for predicting PFAS contamination.


## Getting Started
### Dependencies
1. `conda create -n <environment-name> python==3.9`
2. `conda activate <environment-name>`
3. Install torch (tested for >=1.7.1 and <=1.11.0) and torchvision (tested for >=0.8.2 and <=0.12). May vary with your system. Please check at: https://pytorch.org/get-started/previous-versions/.
    1. e.g.: `pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115`
4. `pip install -U openmim`
5. `mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/{cuda_version}/{torch_version}/index.html`. Note that pre-built wheels (fast installs without needing to build) only exist for some versions of torch and CUDA. Check compatibilities here: https://mmcv.readthedocs.io/en/v1.6.2/get_started/installation.html
    1. e.g.: `mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11.0/index.html`

6. Follow the required packages as mentioned in requirements.txt 

## Repository Structure

| File | Description |
|------|-------------|
| `pretraining.py` | Script to pretrain the model on the multichannel geospatial features. Includes setup for masked image modeling. |
| `main.py` | Script to finetune the pretrained model using task-specific labels. Supports training routine. |
| `patch_extraction.py` | Utility to extract aligned image and mask patches for model training. Handles multichannel rasters and label masks. |
| `noise_masks.py` | Script to create noise-aware label masks based on distance, flow, land cover, and sampling priors. |

### Dependencies

pillow,
rasterio,
scikit-learn,
scipy,
joblib,
tqdm,
openpyxl,
geopandas,
tensorflow,
pandas,
numpy,
pyproj,
toolz,
dask,
shapely,
nomkl,
matplotlib,
fiona,
keras,

## Data

The original PFAS monitoring data are provided by [NRSA, EPA Great Lakes studies, and MPART].

To run the code, you will need:

- Multi-band geospatial rasters with:
  - Land cover (e.g., NLCD)
  - Distance-to-facility bands
  - Flow direction / hydrological features
- Point-level PFAS measurements converted to raster labels
  
**Data Availability.** The dataset used in this study is large in size and is therefore hosted externally. The data are available on [Google Drive](https://drive.google.com/drive/folders/10g0kCCWZZNVvsV0YTeTV_ZC-eviRbfCV?usp=sharing).


## Status

This repository provides the core implementation used in the FOCUS framework.
We plan to release expanded documentation and usage examples in a future update.

