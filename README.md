<div align="center">
  <img src="GUM.jpg" alt="Example Image" width="200">
  <h1>Global Urban Mapper 2.0</h1>
</div>

## INTRODUCTION

**GUM 2.0** is a robust multimodal urban mapping solution designed to handle incomplete modality observations effectively. Users can leverage this code to map specific urban areas using complete or incomplete Sentinel-2, Sentinel-1, and topographic data.

As a deep learning model, GUM 2.0 has demonstrated its effectiveness in various downstream tasks, including road extraction, building identification, building height estimation, and LCZ (Local Climate Zone) mapping.

**GUM 2.0** supports the monitoring of Sustainable Development Goals (SDGs) by offering researchers and practitioners a flexible, efficient model for urban mapping and the extraction of urban-related elements under varying modality availabilities.

## Repository Overview: Structure and Components
This repository is organized into three primary folders:
- **`gee_code`**: data preprocessing scripts implemented in Google Earth Engine (see `gee_code/README.md`).
- **`algorithm`**: houses the code for the incomplete-learning algorithm, including model training and evaluation (see `algorithm/README.md`; Python 3.11 with a CUDA 11.8 GPU required).
- **`mapping`**: a Colab notebook (`gum2.0_mapping.ipynb`) for running GUM 2.0 predictions without a local GPU (see `mapping/README.md`).

## Trained model
The GUM 2.0 checkpoint is available on
[Google Drive](https://drive.google.com/file/d/1TjpcohwZzzij_Gbm-NhY5-1gyNTlOCan/view?usp=drive_link).
See `mapping/README.md` for the expected file layout and prediction steps, and
`algorithm/README.md` for training and evaluation.

## Citation
If you use GUM 2.0 in your research, please cite:

> Zhou, Y., & Weng, Q. (2026). Advancing global urban mapping with multimodal robustness and versatile applications. *ISPRS Journal of Photogrammetry and Remote Sensing*, 242, 131-155.

## Contact Information

If you have any questions or need further clarification about the project, feel free to reach out by emailing us at 22042458r@connect.polyu.hk or by posting an issue on our repository. We welcome any suggestions!
