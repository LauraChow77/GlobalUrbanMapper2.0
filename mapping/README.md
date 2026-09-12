# `mapping` Directory

This is the prediction code directory for the Global Urban Mapper (GUM) 2.0
project. It provides the resources to perform urban mapping with a trained
GUM 2.0 model. The Colab-compatible notebook lets researchers without
personal GPU resources run predictions on Google Colab (T4 GPU).

## Contents

- `gum2.0_mapping.ipynb`: a step-by-step Colab notebook for global urban
  mapping. It installs the environment, loads a GUM 2.0 checkpoint, and
  predicts urban maps from multimodal image stacks. Large scenes are
  processed in 256x256 patches via the model's `test_step`.

## Download model

[GUM 2.0 checkpoint (Google Drive)](https://drive.google.com/file/d/1TjpcohwZzzij_Gbm-NhY5-1gyNTlOCan/view?usp=drive_link)

## Making predictions

1. Open `gum2.0_mapping.ipynb` in Google Colab and set the runtime to a
   T4 GPU as described in the notebook. Two setup cells require a runtime
   restart; the notebook also documents a one-time mmcv/mmseg version-check
   workaround in its Setup section.
2. Mount Google Drive and arrange your files as follows (paths are set in
   the notebook's "Global Variables" cell):

   ```
   My Drive/
   └── GUM2.0/
       ├── model/GUM_2p0.pth   <- the downloaded checkpoint
       ├── datasets/           <- your input .tif stacks
       └── output/             <- predictions; create this folder beforehand
   ```

3. Inputs are GeoTIFF stacks with the GUM 2.0 channel layout
   `[S2 x4, S1 x4, (unused), slope, aspect, ...]`; the 13-band stacks
   produced by the GUM 2.0 preprocessing pipeline work as-is. Outputs are
   single-band uint8 GeoTIFFs (0 = non-urban, 1 = urban, nodata = 255)
   with the same georeferencing as the input.
4. Run all cells in order. Predictions are written to `output/` with the
   same filenames as the inputs.

Contact: 22042458r@connect.polyu.hk
