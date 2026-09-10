import torch
from yacs.config import CfgNode as CN


_C = CN()

# General
_C.SEED = 0
_C.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_C.OUTPUT_DIR = ""
_C.resume = ""
_C.MODE = 'normal'
_C.INFERENCE_MODE = 'hard'

# Model
_C.MODEL = CN()
_C.MODEL.NAME = "UNet_PSPHEAD"
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.WEIGHTS = ""

# Teacher checkpoints, used when MODE == 'ST'
_C.MODEL.TEACHER_FULL = './result/model/teachers/s2_s1_topo.pth'
_C.MODEL.TEACHER_S1_TOPO = './result/model/teachers/s1_topo.pth'
_C.MODEL.TEACHER_S2_TOPO = './result/model/teachers/s2_topo.pth'

# Input
_C.INPUT = CN()
_C.INPUT.MODE = 'FO'
_C.INPUT.MASK_VALUE = 0

# Loss
_C.LOSS = CN()
_C.LOSS.TYPE = 'ce'

# Datasets
# NOTE: the paths below are generic placeholders. Override them for your own
# machine in configs/UNetResMultiEnc_stage1n2Epoch50.yaml (commented samples
# are provided there), or copy that file to a gitignored *.local.yaml.
_C.DATASETS = CN()
_C.DATASETS.CLASS_NAMES = ["non-urban", "urban"]
_C.DATASETS.TRAIN_DIRS = [
    '/path/to/train_images',
    '/path/to/train_images',
    '/path/to/train_images',
    '/path/to/train_images',
]
_C.DATASETS.TRAIN_TXT_PATHS = [
    '/path/to/train_files_1.txt',
    '/path/to/train_files_2.txt',
    '/path/to/train_files_3.txt',
    '/path/to/train_files_4.txt',
]
_C.DATASETS.VAL_DIRS = ['/path/to/val_images']
_C.DATASETS.VAL_TXT_PATHS = ['/path/to/val_files.txt']
_C.DATASETS.TEST_DIRS = ['/path/to/test_images']
_C.DATASETS.TEST_TXT_PATHS = ['/path/to/test_files.txt']
_C.DATASETS.LABEL_DIR = '/path/to/urban_labels'

# Simulated cloud-mask rasters for the cloud-occlusion test scenarios
_C.DATASETS.CLOUD_MASK_DIRS = CN()
_C.DATASETS.CLOUD_MASK_DIRS.MASK_10 = '/path/to/cloud_masks_10'
_C.DATASETS.CLOUD_MASK_DIRS.MASK_20 = '/path/to/cloud_masks_20'
_C.DATASETS.CLOUD_MASK_DIRS.MASK_30 = '/path/to/cloud_masks_30'
_C.DATASETS.CLOUD_MASK_DIRS.MASK_40 = '/path/to/cloud_masks_40'
_C.DATASETS.CLOUD_MASK_DIRS.MASK_50 = '/path/to/cloud_masks_50'
_C.DATASETS.CLOUD_MASK_DIRS.MASK_60 = '/path/to/cloud_masks_60'
_C.DATASETS.CLOUD_MASK_DIRS.MASK_70 = '/path/to/cloud_masks_70'
_C.DATASETS.CLOUD_MASK_DIRS.MASK_80 = '/path/to/cloud_masks_80'
_C.DATASETS.CLOUD_MASK_DIRS.MASK_90 = '/path/to/cloud_masks_90'

_C.DATASETS.INFERENCE_DIRS = ['/path/to/inference_images']
_C.DATASETS.INFERENCE_TXT_PATHS = ['None']

# Solver
_C.SOLVER = CN()
_C.SOLVER.EPOCH = 50
_C.SOLVER.LR_METHOD = 'poly'
_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.BASE_LR_D = 0.0005
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.BATCH_SIZE = 16
_C.SOLVER.NUM_WORKERS = 8
_C.SOLVER.RANDOM_MASK_RATIO = 0.2
