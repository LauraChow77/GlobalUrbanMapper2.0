# `algorithm` Directory

This directory contains the code for training the teacher models of GUM 2.0.

Each teacher is a multi-encoder UNet (`UNetResMultiEnc`) with separate encoders
for Sentinel-2, Sentinel-1, and topography. Three teachers are trained, one
per modality availability: full observation (`FO`), S2-missing (`S2_missing`),
and S1-missing (`S1_missing`).

## How to train

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
The checkpoint may also be a student model trained with `MODE: 'ST'`.
