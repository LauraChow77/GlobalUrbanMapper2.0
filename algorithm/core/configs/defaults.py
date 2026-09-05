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
_C.DATASETS = CN()
_C.DATASETS.CLASS_NAMES = ["non-urban", "urban"]
_C.DATASETS.TRAIN_DIRS = [
    'D:/code/GUM_multimodal/data/sup/img_stack_with_product',
    'D:/code/GUM_multimodal/data/unsup/img_stack_with_product',
    'D:/code/GUM_multimodal/data/unsup/img_stack_with_product',
    'D:/code/GUM_multimodal/data/unsup/img_stack_with_product',
]
_C.DATASETS.TRAIN_TXT_PATHS = [
    'D:/code/GUM_decoupling/data/stage1_train_13728.txt',
    'D:/code/GUM_decoupling/data/stage2_inner_12721.txt',
    'D:/code/GUM_decoupling/data/stage2_middle_3889.txt',
    'D:/code/GUM_decoupling/data/stage2_outer_2800.txt',
]
_C.DATASETS.VAL_DIRS = ['D:/code/GUM_multimodal/data/sup/img_stack_with_product']
_C.DATASETS.VAL_TXT_PATHS = ['D:/code/GUM_decoupling/data/stage1_val_2463.txt']
_C.DATASETS.TEST_DIRS = ['D:/code/GUM_multimodal/data/sup/img_stack_with_product']
_C.DATASETS.TEST_TXT_PATHS = ['D:/code/GUM_decoupling/data/test_2929.txt']
_C.DATASETS.LABEL_DIR = 'D:/code/GUM_decoupling/data/urban_label_gum2p0'

# Simulated cloud-mask rasters for the cloud-occlusion test scenarios
_C.DATASETS.CLOUD_MASK_DIRS = CN()
_C.DATASETS.CLOUD_MASK_DIRS.MASK_10 = 'D:/code/GUM_multimodal/data/sup/simulate_cloud_mask_0.1'
_C.DATASETS.CLOUD_MASK_DIRS.MASK_30 = 'D:/code/GUM_multimodal/data/sup/simulate_cloud_mask_0.3'
_C.DATASETS.CLOUD_MASK_DIRS.MASK_50 = 'D:/code/GUM_multimodal/data/sup/simulate_cloud_mask_0.5'
_C.DATASETS.CLOUD_MASK_DIRS.MASK_70 = 'D:/code/GUM_multimodal/data/sup/simulate_cloud_mask_0.7'
_C.DATASETS.CLOUD_MASK_DIRS.MASK_90 = 'D:/code/GUM_multimodal/data/sup/simulate_cloud_mask_0.9'

_C.DATASETS.INFERENCE_DIRS = ['C:/mapping/img_clipped']
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
