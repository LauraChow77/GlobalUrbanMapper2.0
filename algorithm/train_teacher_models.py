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
import os
from torch.optim import SGD
from core.utils.evaluator import Evaluator, run_val
from torchvision import transforms
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Teacher-model modes: full input, S2 missing, or S1 missing.
TEACHER_MODES = {'FO', 'S2_missing', 'S1_missing'}


def forward_model(model, inputs, s2, s1, topo, cfg):
    """Apply the configured teacher-modality mask and run the model."""
    input_mode = getattr(cfg.INPUT, 'MODE', 'FO')
    if input_mode == 'S2_missing':
        s2 = torch.full_like(s2, cfg.INPUT.MASK_VALUE)
    elif input_mode == 'S1_missing':
        s1 = torch.full_like(s1, cfg.INPUT.MASK_VALUE)

    model_outputs = model(s2, s1, topo)
    if isinstance(model_outputs, tuple):
        outputs = model_outputs[-1]
    else:
        outputs = model_outputs
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
        shuffle=True, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
        num_workers=cfg.SOLVER.NUM_WORKERS, worker_init_fn=seed_worker,
        shuffle=False, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
        num_workers=cfg.SOLVER.NUM_WORKERS, worker_init_fn=seed_worker,
        shuffle=False, pin_memory=True)
    return training_loader, val_loader, test_loader


def train_epochwise(epoch,
                    training_loader, model,
                    optimizer, scheduler,
                    criterions,
                    cfg):
    loss_sum = 0.0

    ce = criterions[0]
    model.train()

    with tqdm(range(len(training_loader)), ncols=150) as tbar:
        tbar.set_description("epoch %3d/%-3d" % (epoch + 1, cfg.SOLVER.EPOCH))
        for i, data in enumerate(training_loader):
            inputs, targets = data[0].to(cfg.DEVICE), data[1].to(torch.long).to(cfg.DEVICE)

            optimizer.zero_grad()

            s2, s1, topo = inputs[:, :4, :, :], inputs[:, 4:8, :, :], inputs[:, 8:, :, :]

            outputs = forward_model(model, inputs, s2, s1, topo, cfg)[0]
            loss = compute_loss(outputs, targets, ce)

            loss.backward()
            optimizer.step()
            scheduler(optimizer, i, epoch, 0)

            loss_sum += loss.item()
            tbar.set_postfix_str("loss_seg: {:.3f}".format(loss_sum / (i + 1)))
            tbar.update()

    return None

def train(cfg):
    """Train a single teacher model (full, S2-missing, or S1-missing input).

    The desired modality availability is selected via cfg.INPUT.MODE:
      - 'FO'            : full observation (S2 + S1 + topo)
      - 'S2_missing'    : S2 masked with MASK_VALUE
      - 'S1_missing'    : S1 masked with MASK_VALUE
    """
    input_mode = getattr(cfg.INPUT, 'MODE', 'FO')
    if input_mode not in TEACHER_MODES:
        raise ValueError(
            f"cfg.INPUT.MODE must be one of {TEACHER_MODES}, got {input_mode}")

    writer = SummaryWriter(os.path.join(cfg.EXPERIMENT_DIR, "log"))
    logger = logging.getLogger("UNet_PSPHEAD.trainer")
    logger.info("Start training teacher model (mode=%s)", input_mode)
    experiment_result_dir = os.path.join(cfg.EXPERIMENT_DIR, "experiments")
    os.makedirs(experiment_result_dir, exist_ok=True)

    training_loader, val_loader, test_loader = build_dataloaders(cfg)

    model = build_model(cfg)
    model.to(cfg.DEVICE)

    optimizer = SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.BASE_LR_D)
    scheduler = lr_scheduler(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, cfg.SOLVER.EPOCH, len(training_loader))
    criterions = build_criterions(cfg.LOSS.TYPE)

    best_indices = {"val_miou": 0., "val_loss": np.inf,
                    "test_miou": 0., "test_loss": np.inf}
    evaluator = Evaluator(cfg.MODEL.NUM_CLASSES)
    start_epoch = 0 if cfg.resume == "" else torch.load(cfg.resume)["epoch"]
    for epoch in range(start_epoch, cfg.SOLVER.EPOCH):
        train_epochwise(
            epoch, training_loader, model, optimizer, scheduler,
            criterions, cfg)

        criterion = criterions[0]
        val_indices = run_val(
            epoch, val_loader, model, evaluator, criterion, cfg, forward_model, writer, ["Val/loss", "Val/mIoU"])

        test_indices = run_val(
            epoch, test_loader, model, evaluator, criterion, cfg, forward_model, writer, ["Test/loss", "Test/mIoU"])

        save_results(
            epoch, model, scheduler, "UNet_PSPHead",
            best_indices, val_indices, test_indices,
            experiment_result_dir, cfg)

    writer.close()

def main():
    cfg, args = parse_arguments()
    set_random_seed(cfg.SEED, deterministic=True)
    setup_config_logger(cfg, args)
    train(cfg)

if __name__ == '__main__':
    main()
