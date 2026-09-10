# `algorithm` Directory

This directory contains the code for training the teacher models of GUM 2.0.

Each teacher is a multi-encoder UNet (`UNetResMultiEnc`) with separate encoders
for Sentinel-2, Sentinel-1, and topography. Three teachers are trained, one
per modality availability: full observation (`FO`), S2-missing (`S2_missing`),
and S1-missing (`S1_missing`).

## Dataset configuration

All dataset locations are under `DATASETS.*` — defaults in
`core/configs/defaults.py`, to be overridden for your own machine in
`configs/UNetResMultiEnc_stage1n2Epoch50.yaml` (yacs only accepts keys
already defined in the defaults; commented-out samples for every key are
provided in the yaml itself). Each input sample is a 256×256, 10-channel
stack `[S2 ×4, S1 ×4, slope, aspect]` with a matching binary urban label.

| Key | Description |
| --- | --- |
| `TRAIN_DIRS` / `TRAIN_TXT_PATHS` | Image folders + filename lists for training (4 parallel entries: supervised stage 1 + 3 unsupervised stage-2 subsets) |
| `VAL_DIRS` / `VAL_TXT_PATHS` | Validation images + filename list |
| `TEST_DIRS` / `TEST_TXT_PATHS` | Test images + filename list |
| `LABEL_DIR` | Urban-label rasters, shared by train / val / test |
| `CLOUD_MASK_DIRS.MASK_10` … `MASK_90` | Simulated cloud-mask rasters; `evaluate_scenarios.py` uses all nine levels, the student augmentation uses `MASK_20`–`MASK_80` |

## How to train the teachers

1. Set the dataset paths (image folders, file lists, and the label
   directory) in `configs/UNetResMultiEnc_stage1n2Epoch50.yaml` — all under
   `DATASETS.*`.

2. Train one teacher per modality availability:

```bash
python train_teacher_models.py -cfg configs/UNetResMultiEnc_stage1n2Epoch50.yaml INPUT.MODE FO
python train_teacher_models.py -cfg configs/UNetResMultiEnc_stage1n2Epoch50.yaml INPUT.MODE S2_missing
python train_teacher_models.py -cfg configs/UNetResMultiEnc_stage1n2Epoch50.yaml INPUT.MODE S1_missing
```

3. Each run writes a timestamped experiment folder under `result/`
   containing the config log, TensorBoard events, per-epoch metrics, and
   checkpoints. The file `val_max_miou.pth` is the best-validation
   checkpoint of that teacher.

## How to train the student

After the three teachers are trained:

1. Set their checkpoint paths (`MODEL.TEACHER_FULL`, `MODEL.TEACHER_S1_TOPO`,
   `MODEL.TEACHER_S2_TOPO`) and the augmentation strength
   (`SOLVER.RANDOM_MASK_RATIO`, e.g. `0.2`) in
   `configs/UNetResMultiEnc_stage1n2Epoch50.yaml`.

2. Run with `MODE: 'ST'`:

```bash
python train_student_models.py -cfg configs/UNetResMultiEnc_stage1n2Epoch50.yaml MODE ST
```

During training, each batch is randomly corrupted (no S2, no S1, or a
simulated cloud mask at 20/40/60/80%); the student learns to match the
ensemble of the three frozen teachers. The cloud-mask folders are read from
`DATASETS.CLOUD_MASK_DIRS`. The best-validation checkpoint is saved as
`val_max_miou.pth` in the timestamped experiment folder, ready for
`evaluate_scenarios.py`.

## How to evaluate

`evaluate_scenarios.py` evaluates one checkpoint under every
modality-availability scenario in a single run:

```bash
python evaluate_scenarios.py -cfg configs/UNetResMultiEnc_stage1n2Epoch50.yaml \
    MODEL.WEIGHTS <path_to_checkpoint>/val_max_miou.pth
```

| Scenario | Meaning |
| -------- | ------- |
| `full` | S2 + S1 + topography (full observation) |
| `nos2` | S2 fully masked |
| `nos1` | S1 fully masked |
| `cloud_mask_10` ... `cloud_mask_90` | S2 partially occluded by simulated clouds at 10-90% |

Outputs (under a new `result/.../<timestamp>/` experiment directory):

- `test_results.csv` — OA / mIoU / loss and per-class IoU, precision, recall
  for each scenario.
- `res/`, `res_nos2/`, `res_nos1/`, `res_cloud_mask_XX/` — prediction rasters.

The simulated cloud masks are read from the folders set in
`DATASETS.CLOUD_MASK_DIRS` in `configs/UNetResMultiEnc_stage1n2Epoch50.yaml`.
