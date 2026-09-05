from common import parse_arguments, setup_config_logger
from core.utils.seed import set_random_seed, seed_worker
from core.losses import build_criterions

from core.models import build_model
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from core.datasets import GUM
from tqdm import tqdm
import rasterio
import os
import numpy as np
from core.utils.evaluator import Evaluator, comprehensive_evaluation
import csv


_cloud_mask_cache = {}


def apply_cloud_mask(filenames, mask_dir, s2_cloud_mask, mask_value):
    for index, filename in enumerate(filenames):
        cache_key = (filename, mask_dir)
        if cache_key not in _cloud_mask_cache:
            mask_path = os.path.join(mask_dir, filename)
            with rasterio.open(mask_path) as src:
                _cloud_mask_cache[cache_key] = src.read(1)
        cloud_mask = _cloud_mask_cache[cache_key]
        mask = torch.tensor(cloud_mask != 1, dtype=torch.bool, device=s2_cloud_mask.device)
        s2_cloud_mask[index] = torch.where(
            mask, s2_cloud_mask[index],
            torch.tensor(mask_value, device=s2_cloud_mask.device))


@torch.no_grad()
def test(model, cfg):
    data_transforms = transforms.Compose([transforms.ToTensor()])
    test_dataset = GUM(
        cfg.DATASETS.TEST_DIRS, cfg.DATASETS.TEST_TXT_PATHS,
        transform=data_transforms, inference_mode=False,
        label_dir=cfg.DATASETS.LABEL_DIR)
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
        num_workers=cfg.SOLVER.NUM_WORKERS, worker_init_fn=seed_worker,
        shuffle=False)

    criterions = build_criterions(cfg.LOSS.TYPE)
    criterion = criterions[0]

    evaluator_names = [
        'evaluator',
        'evaluator_nos2',
        'evaluator_nos1',
        'evaluator_cloud_mask_10',
        'evaluator_cloud_mask_30',
        'evaluator_cloud_mask_50',
        'evaluator_cloud_mask_70',
        'evaluator_cloud_mask_90',
    ]

    evaluators = {name: Evaluator(cfg.MODEL.NUM_CLASSES) for name in evaluator_names}

    model.eval()
    loss_variables = {
        'outputs': 0.0,
        'outputs_nos2': 0.0,
        'outputs_nos1': 0.0,
        'outputs_cloud_mask_10': 0.0,
        'outputs_cloud_mask_30': 0.0,
        'outputs_cloud_mask_50': 0.0,
        'outputs_cloud_mask_70': 0.0,
        'outputs_cloud_mask_90': 0.0,
    }

    with tqdm(range(len(test_loader)), ncols=150) as tbar:
        tbar.set_description("testing")
        for i, data in enumerate(test_loader):
            inputs, targets = data[0].to(cfg.DEVICE), data[1].to(torch.long).to(cfg.DEVICE)
            s2, s1, topo = inputs[:, :4, :, :], inputs[:, 4:8, :, :], inputs[:, 8:, :, :]

            no_s2 = torch.full_like(s2, cfg.INPUT.MASK_VALUE)
            no_s1 = torch.full_like(s1, cfg.INPUT.MASK_VALUE)

            s2_cloud_masks = {
                '10': s2.clone(),
                '30': s2.clone(),
                '50': s2.clone(),
                '70': s2.clone(),
                '90': s2.clone(),
            }
            cloud_mask_dirs = {
                '10': cfg.DATASETS.CLOUD_MASK_DIRS.MASK_10,
                '30': cfg.DATASETS.CLOUD_MASK_DIRS.MASK_30,
                '50': cfg.DATASETS.CLOUD_MASK_DIRS.MASK_50,
                '70': cfg.DATASETS.CLOUD_MASK_DIRS.MASK_70,
                '90': cfg.DATASETS.CLOUD_MASK_DIRS.MASK_90,
            }
            for key in s2_cloud_masks.keys():
                apply_cloud_mask(data[2]['filename'], cloud_mask_dirs[key],
                                 s2_cloud_masks[key], cfg.INPUT.MASK_VALUE)

            outputs_dict = {}
            outputs_dict['outputs'] = model(s2, s1, topo)
            outputs_dict['outputs_nos2'] = model(no_s2, s1, topo)
            outputs_dict['outputs_nos1'] = model(s2, no_s1, topo)
            for key in s2_cloud_masks.keys():
                outputs_dict[f'outputs_cloud_mask_{key}'] = model(s2_cloud_masks[key], s1, topo)

            loss_dict = {}
            squeezed_targets = targets.squeeze(1)
            for key, output in outputs_dict.items():
                loss_dict[f'loss_{key}'] = criterion(output, squeezed_targets)

            for key, loss in loss_dict.items():
                loss_variables[key.replace('loss_', '')] += loss.item()

            tbar.set_postfix_str("test loss: %.3f | test loss (no s2): %.3f | test loss (no s1): %.3f" % (
                loss_variables['outputs'] / (i + 1),
                loss_variables['outputs_nos2'] / (i + 1),
                loss_variables['outputs_nos1'] / (i + 1)))
            tbar.update()

            pred_dict = {}
            for key, output in outputs_dict.items():
                pred_dict[key] = np.argmax(output.cpu().numpy(), axis=1)

            target = squeezed_targets.cpu().numpy()
            for key, evaluator in evaluators.items():
                pred = pred_dict[key.replace('evaluator', 'outputs')]
                evaluators[key].add_batch(target, pred)

            for j in range(target.shape[0]):
                file_name = data[2]['filename'][j]
                input_path = os.path.join(cfg.DATASETS.TEST_DIRS[0], file_name)

                with rasterio.open(input_path) as src:
                    profile = src.profile
                    profile.update(dtype=np.uint8, count=1, nodata=255)

                for key, outputs in pred_dict.items():
                    output_dir = os.path.join(cfg.EXPERIMENT_DIR, key.replace('outputs', 'res'))
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, file_name)
                    output = outputs[j:j+1, :, :]
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(output)

    evaluator_metrics = {}
    for key, evaluator in evaluators.items():
        print("测试结果", key)
        test_indices = comprehensive_evaluation(0, evaluator, cfg.DATASETS.CLASS_NAMES)
        test_indices["loss"] = loss_variables[key.replace('evaluator', 'outputs')] / len(test_loader)
        evaluator_metrics[key] = test_indices

    _export_results_csv(evaluator_metrics, cfg)


def _export_results_csv(evaluator_metrics, cfg):
    """Export per-scenario test metrics to a CSV in the experiment directory."""
    csv_path = os.path.join(cfg.EXPERIMENT_DIR, "test_results.csv")
    rows = []
    for key, metrics in evaluator_metrics.items():
        scenario = key.replace("evaluator", "").strip("_") or "full"

        row = {"scenario": scenario}
        row["OA"] = metrics.get("OA", "")
        row["mIoU"] = metrics.get("mIoU", "")
        row["loss"] = metrics.get("loss", "")

        for cls_name in cfg.DATASETS.CLASS_NAMES:
            iou_dict = metrics.get("IoU", {})
            row[f"IoU_{cls_name}"] = iou_dict.get(cls_name, "")
            precision_dict = metrics.get("class precision", {})
            row[f"precision_{cls_name}"] = precision_dict.get(cls_name, "")
            recall_dict = metrics.get("class recall", {})
            row[f"recall_{cls_name}"] = recall_dict.get(cls_name, "")

        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved test results to {csv_path}")


def main():
    cfg, args = parse_arguments()
    set_random_seed(cfg.SEED, deterministic=False)
    setup_config_logger(cfg, args)

    model = build_model(cfg)
    if isinstance(model, tuple):
        model = model[0]
    model.to(cfg.DEVICE)

    test(model, cfg)


if __name__ == '__main__':
    main()
