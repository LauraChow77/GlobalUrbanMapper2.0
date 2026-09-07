from common import parse_arguments, setup_config_logger, save_results
from core.utils.seed import set_random_seed, seed_worker
from core.losses import build_criterions
from core.utils.lr_scheduler import lr_scheduler

from torch.utils.tensorboard import SummaryWriter

import logging
from core.models import build_model
import torch
from torch.utils.data import DataLoader
from core.datasets import GUM
from tqdm import tqdm
import rasterio
import os
import numpy as np
from torch.optim import SGD
from core.utils.evaluator import Evaluator, run_val
import torch.nn.functional as F
from torchvision import transforms
import random
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Required by deterministic cublas ops (must be set before the first CUDA call).
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


class CloudMaskAugmenter:
    """Random corruption augmentation for student training.

    Each corrupted sample loses S2, loses S1, or gets an S2 cloud mask at
    20/40/60/80%. The paired s1_topo_mask / s2_topo_mask record, per sample,
    whether the full-observation teacher remains valid (1) or the
    S1+topography / S2+topography teacher takes over (0).
    """
    _random_mask_choices = ['no_s2', 'no_s1', 'mask_20', 'mask_40', 'mask_60', 'mask_80']

    def __init__(self):
        self._preloaded_masks = None
        self._cloud_mask_dirs = None
        self._mask_value = 0

    def initialize(self, cfg):
        if cfg.SOLVER.RANDOM_MASK_RATIO > 0 and self._preloaded_masks is None:
            self._mask_value = cfg.INPUT.MASK_VALUE
            self._cloud_mask_dirs = {
                20: cfg.DATASETS.CLOUD_MASK_DIRS.MASK_20,
                40: cfg.DATASETS.CLOUD_MASK_DIRS.MASK_40,
                60: cfg.DATASETS.CLOUD_MASK_DIRS.MASK_60,
                80: cfg.DATASETS.CLOUD_MASK_DIRS.MASK_80,
            }
            self._preloaded_masks = self._preload(self._cloud_mask_dirs)

    def _preload(self, mask_dirs):
        """Load all cloud-mask rasters into RAM as uint8 arrays.

        Returns {level: (N, H, W) uint8 numpy array}. The rasters are tiny
        (~250 KB each), so preloading avoids a disk read + decompression
        stall on the training thread for every corrupted sample.
        """
        masks = {}
        for key, directory in mask_dirs.items():
            files = [
                f for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith('.tif')
            ]
            if not files:
                raise FileNotFoundError(
                    f"No cloud mask TIFF files found for mask level {key} in {directory}")
            level_masks = []
            for f in sorted(files):
                with rasterio.open(os.path.join(directory, f)) as src:
                    level_masks.append(src.read(1))
            masks[key] = np.stack(level_masks)
        return masks

    def _handle_no_s2(self, index, s2, s1, s1_topo_mask, s2_topo_mask):
        s1_topo_mask[index, :, :, :] = 0
        s2[index, :, :, :] = self._mask_value
        return s2, s1, s1_topo_mask, s2_topo_mask

    def _handle_no_s1(self, index, s2, s1, s1_topo_mask, s2_topo_mask):
        s2_topo_mask[index, :, :, :] = 0
        s1[index, :, :, :] = self._mask_value
        return s2, s1, s1_topo_mask, s2_topo_mask

    def _handle_mask(self, index, s2, s1, s1_topo_mask, s2_topo_mask, mask_value):
        level_masks = self._preloaded_masks.get(mask_value)
        if level_masks is None or len(level_masks) == 0:
            raise FileNotFoundError(
                f"No preloaded cloud masks for mask level {mask_value} "
                f"({self._cloud_mask_dirs.get(mask_value, 'unknown directory')})"
            )
        cloud_mask = level_masks[random.randrange(len(level_masks))]
        mask = torch.from_numpy(cloud_mask != 1).to(s2.device)
        s1_topo_mask[index, :, :, :] = mask
        s2[index, :, :, :] = torch.where(
            mask, s2[index, :, :, :],
            torch.tensor(self._mask_value, device=s2.device)
        )
        return s2, s1, s1_topo_mask, s2_topo_mask

    def apply(self, index, choice, s2, s1, s1_topo_mask, s2_topo_mask):
        switcher = {
            'no_s2': self._handle_no_s2,
            'no_s1': self._handle_no_s1,
            'mask_20': lambda *args: self._handle_mask(*args, 20),
            'mask_40': lambda *args: self._handle_mask(*args, 40),
            'mask_60': lambda *args: self._handle_mask(*args, 60),
            'mask_80': lambda *args: self._handle_mask(*args, 80),
        }
        func = switcher.get(choice)
        if func is None:
            raise ValueError(f"No handler for mask choice '{choice}'")
        return func(index, s2, s1, s1_topo_mask, s2_topo_mask)


def forward_model(model, inputs, s2, s1, topo, cfg):
    """Student forward for evaluation (eval mode returns a single tensor)."""
    outputs = model(s2, s1, topo)
    if isinstance(outputs, tuple):
        outputs = outputs[-1]
    return outputs, None


def compute_loss(outputs, targets, ce):
    """Compute training loss."""
    return ce(outputs, targets.squeeze(1))


def build_dataloaders(cfg):
    """Build training, validation, and test DataLoaders."""
    data_transforms = transforms.Compose([transforms.ToTensor()])
    training_dataset = GUM(
        cfg.DATASETS.TRAIN_DIRS, cfg.DATASETS.TRAIN_TXT_PATHS,
        transform=data_transforms, inference_mode=False,
        label_dir=cfg.DATASETS.LABEL_DIR)
    val_dataset = GUM(
        cfg.DATASETS.VAL_DIRS, cfg.DATASETS.VAL_TXT_PATHS,
        transform=data_transforms, inference_mode=False,
        label_dir=cfg.DATASETS.LABEL_DIR)
    test_dataset = GUM(
        cfg.DATASETS.TEST_DIRS, cfg.DATASETS.TEST_TXT_PATHS,
        transform=data_transforms, inference_mode=False,
        label_dir=cfg.DATASETS.LABEL_DIR)

    training_loader = DataLoader(
        training_dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
        num_workers=cfg.SOLVER.NUM_WORKERS, worker_init_fn=seed_worker,
        shuffle=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
        num_workers=cfg.SOLVER.NUM_WORKERS, worker_init_fn=seed_worker,
        shuffle=False, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
        num_workers=cfg.SOLVER.NUM_WORKERS, worker_init_fn=seed_worker,
        shuffle=False, pin_memory=True, persistent_workers=True)
    return training_loader, val_loader, test_loader


def train_epochwise_st(epoch,
                       training_loader,
                       model, t_model, s1_topo_model, s2_topo_model,
                       optimizer, scheduler,
                       criterions,
                       cfg,
                       mask_augmenter):
    """Train the student for one epoch with contextual knowledge distillation."""
    random_mask_ratio = cfg.SOLVER.RANDOM_MASK_RATIO

    loss_sum = 0.0
    sup_loss_sum, consistent_loss_sum = 0.0, 0.0

    model.train()
    t_model.eval()
    s1_topo_model.eval()
    s2_topo_model.eval()

    ce, kl = criterions[0], criterions[1]

    with tqdm(range(len(training_loader)), ncols=150) as tbar:
        tbar.set_description("epoch %3d/%-3d" % (epoch + 1, cfg.SOLVER.EPOCH))
        for i, data in enumerate(training_loader):
            inputs, targets = data[0].to(cfg.DEVICE), data[1].to(torch.long).to(cfg.DEVICE)
            s2, s1, topo = inputs[:, :4, :, :], inputs[:, 4:8, :, :], inputs[:, 8:, :, :]
            s2_full, s1_full, topo_full = s2.clone(), s1.clone(), topo.clone()

            batch_size, channels, height, width = s2.shape
            s1_topo_mask = torch.ones(
                (batch_size, 1, height, width), dtype=torch.bool, device=s2.device)
            s2_topo_mask = torch.ones(
                (batch_size, 1, height, width), dtype=torch.bool, device=s2.device)

            if random_mask_ratio > 0:
                random_mask_num = math.ceil(batch_size * random_mask_ratio)
                indices = torch.randperm(batch_size).tolist()[:random_mask_num]
                selected_choices = random.choices(mask_augmenter._random_mask_choices, k=random_mask_num)
                index_choice_pairs = list(zip(indices, selected_choices))

                for index, choice in index_choice_pairs:
                    s2, s1, s1_topo_mask, s2_topo_mask = mask_augmenter.apply(
                        index, choice, s2, s1, s1_topo_mask, s2_topo_mask)

            optimizer.zero_grad()

            kw_outputs, partial_outputs = model(s2, s1, topo)

            no_s2 = torch.full_like(s2_full, cfg.INPUT.MASK_VALUE)
            no_s1 = torch.full_like(s1_full, cfg.INPUT.MASK_VALUE)
            with torch.inference_mode():
                full_outputs = t_model(s2_full, s1_full, topo_full)
                s1_topo_outputs = s1_topo_model(no_s2, s1_full, topo_full)
                s2_topo_outputs = s2_topo_model(s2_full, no_s1, topo_full)

            sup_loss = ce(partial_outputs, targets.squeeze(1))

            contextual_output = full_outputs
            contextual_output = torch.where(s1_topo_mask, contextual_output, s1_topo_outputs)
            contextual_output = torch.where(s2_topo_mask, contextual_output, s2_topo_outputs)

            ensembled_softmax_output = F.softmax(contextual_output, dim=1)
            log_softmax_partial_outputs = F.log_softmax(kw_outputs, dim=1)
            consistent_loss = kl(log_softmax_partial_outputs, ensembled_softmax_output).mean()

            loss = sup_loss + consistent_loss

            loss.backward()
            optimizer.step()
            scheduler(optimizer, i, epoch, 0)

            loss_sum += loss.item()
            sup_loss_sum += sup_loss.item()
            consistent_loss_sum += consistent_loss.item()
            tbar.set_postfix_str(
                "loss_full: {:.3f} | s_sup_loss: {:.3f} | s_consistent_loss: {:.3f}".format(
                    loss_sum / (i + 1), sup_loss_sum / (i + 1), consistent_loss_sum / (i + 1)))
            tbar.update()

    return None


def train(cfg):
    """Train a student model with contextual knowledge distillation.

    The student is supervised by ground-truth labels and by an ensemble of
    three frozen teachers (full observation, S2-missing, S1-missing). During
    training, random cloud/missing-modality augmentation is applied so the
    student learns to match the appropriate teacher for each corrupted input.
    """
    writer = SummaryWriter(os.path.join(cfg.EXPERIMENT_DIR, "log"))
    logger = logging.getLogger("UNet_PSPHEAD.trainer")
    logger.info("Start training student model")
    experiment_result_dir = os.path.join(cfg.EXPERIMENT_DIR, "experiments")
    os.makedirs(experiment_result_dir, exist_ok=True)

    training_loader, val_loader, test_loader = build_dataloaders(cfg)

    built = build_model(cfg)
    if not isinstance(built, tuple):
        raise ValueError(
            "Student training requires MODE: 'ST' so build_model returns the "
            "student plus the three frozen teachers; got a single model "
            "(check MODE in the config or pass MODE ST on the command line).")
    model, t_model, s1_topo_model, s2_topo_model = built
    for p in t_model.parameters():
        p.requires_grad = False
    for p in s1_topo_model.parameters():
        p.requires_grad = False
    for p in s2_topo_model.parameters():
        p.requires_grad = False

    model = model.to(cfg.DEVICE)
    t_model = t_model.to(cfg.DEVICE)
    s1_topo_model = s1_topo_model.to(cfg.DEVICE)
    s2_topo_model = s2_topo_model.to(cfg.DEVICE)

    optimizer = SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR,
                    momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.BASE_LR_D)
    scheduler = lr_scheduler(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR,
                             cfg.SOLVER.EPOCH, len(training_loader))
    criterions = build_criterions(cfg.LOSS.TYPE, 'kl')

    # Random corruption augmenter, created once and shared across epochs.
    mask_augmenter = CloudMaskAugmenter()
    mask_augmenter.initialize(cfg)

    best_indices = {"val_miou": 0., "val_loss": np.inf,
                    "test_miou": 0., "test_loss": np.inf}
    evaluator = Evaluator(cfg.MODEL.NUM_CLASSES)
    start_epoch = 0 if cfg.resume == "" else torch.load(cfg.resume)["epoch"]
    for epoch in range(start_epoch, cfg.SOLVER.EPOCH):
        train_epochwise_st(
            epoch, training_loader, model, t_model, s1_topo_model, s2_topo_model,
            optimizer, scheduler, criterions, cfg, mask_augmenter)

        criterion = criterions[0]
        val_indices = run_val(
            epoch, val_loader, model, evaluator, criterion, cfg, forward_model, writer, ["Val/loss", "Val/mIoU"])

        # Test evaluation is only meaningful when the checkpoint improves
        # (selection is val-based anyway), so skip it on flat epochs.
        if val_indices["mIoU"] >= best_indices["val_miou"]:
            test_indices = run_val(
                epoch, test_loader, model, evaluator, criterion, cfg, forward_model, writer, ["Test/loss", "Test/mIoU"])
        else:
            test_indices = None

        save_results(
            epoch, model, scheduler, "UNet_PSPHead",
            best_indices, val_indices, test_indices,
            experiment_result_dir, cfg)

    writer.close()


def main():
    cfg, args = parse_arguments()
    set_random_seed(cfg.SEED, deterministic=True)
    # Strict deterministic mode is impossible here: cross_entropy_loss has no
    # deterministic CUDA kernel in torch 2.0. warn_only keeps convs/BN
    # deterministic (via cudnn.deterministic=True) without crashing.
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Pin the cudnn TF32 default explicitly: convolutions have always run in
    # TF32 (it is the PyTorch default), so this changes nothing today, but a
    # future default flip must not silently alter numerics. TF32 matmuls
    # (torch.backends.cuda.matmul.allow_tf32) are deliberately NOT enabled:
    # they would change results relative to previously trained checkpoints.
    torch.backends.cudnn.allow_tf32 = True
    setup_config_logger(cfg, args)
    train(cfg)


if __name__ == '__main__':
    main()
