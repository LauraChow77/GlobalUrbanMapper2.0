import numpy as np
import torch
from tqdm import tqdm

from core.utils.pretty_format import organize_info


eps = 1e-8


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        """
        Compute the overall accuracy (OA).
        :return: overall accuracy (OA)
        """
        OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + eps)
        return OA

    def Precision_Class(self):
        """
        Compute per-class precision (i.e., among pixels predicted as class a,
        how many truly are a) and the mean precision.
        :return: per-class precision and mean precision
        """
        precison_cls = np.zeros(self.num_class)
        for i in range(self.num_class):
            precison_cls[i] = self.confusion_matrix[i, i] / (self.confusion_matrix[:, i].sum() + eps)
        # Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + eps)
        m_precision = np.nanmean(precison_cls)
        return precison_cls, m_precision

    def Recall_Class(self):
        """
        Compute per-class recall (i.e., for each class a, how many of its
        pixels are predicted as a) and the mean recall.
        :return: per-class recall and mean recall
        """
        recall_cls = np.zeros(self.num_class)
        for i in range(self.num_class):
            recall_cls[i] = self.confusion_matrix[i, i] / (self.confusion_matrix[i].sum() + eps)
        # Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + eps)
        m_recall = np.nanmean(recall_cls)
        return recall_cls, m_recall

    def Mean_Intersection_over_Union(self):
        """
        Compute per-class IoU and mean IoU.
        :return: per-class IoU and mean IoU
        """
        iou_class = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix) + eps)
        MIoU = np.nanmean(iou_class)
        return iou_class, MIoU

    @staticmethod
    def cal_indices(evaluator, class_names):
        overall_accuracy = evaluator.Pixel_Accuracy()

        precision_class, avg_precision = evaluator.Precision_Class()
        recall_class, avg_recall = evaluator.Recall_Class()

        iou_class, miou = evaluator.Mean_Intersection_over_Union()

        precision_cls = {}
        recall_cls = {}
        iou_cls = {}
        for i in range(len(precision_class)):
            name = class_names[i]
            precision_cls[name] = precision_class[i]
            recall_cls[name] = recall_class[i]
            iou_cls[name] = iou_class[i]

        return ["OA",
                "class precision", "class recall",
                "IoU", "mIoU"],\
               [overall_accuracy,
                precision_cls, recall_cls,
                iou_cls, miou]

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class) & (pre_image >= 0) & (pre_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def comprehensive_evaluation(epoch, evaluator, class_names, to_print=True):
    # compute OA, class accuracy, precision, recall, class IoU, mIoU
    eval_keys, eval_values = Evaluator.cal_indices(evaluator, class_names)
    eval_keys = ["epoch"] + eval_keys
    eval_values = [epoch ] + eval_values
    if to_print:
        print(organize_info(eval_keys, eval_values))
    eval_indices = dict(zip(eval_keys, eval_values))

    return eval_indices


@torch.no_grad()
def run_val(epoch, val_loader, model, evaluator, criterion, cfg, forward_fn,
            writer=None, comment=[]):
    """Evaluate a model on a loader and return per-epoch metric indices."""
    model.eval()
    val_loss = 0.0
    evaluator.reset()
    with tqdm(range(len(val_loader)), ncols=150) as tbar:
        tbar.set_description("epoch %3d/%-3d" % (epoch + 1, cfg.SOLVER.EPOCH))
        for i, data in enumerate(val_loader):
            inputs, targets = data[0].to(torch.float32).to(cfg.DEVICE), data[1].to(torch.long).to(cfg.DEVICE)
            s2, s1, topo = inputs[:, :4, :, :], inputs[:, 4:8, :, :], inputs[:, 8:, :, :]

            squeezed_targets = targets.squeeze(1)
            outputs, _ = forward_fn(model, inputs, s2, s1, topo, cfg)

            loss = criterion(outputs, squeezed_targets)

            val_loss += loss.item()
            tbar.set_postfix_str("val loss: %.3f" % (val_loss / (i + 1)))
            tbar.update()

            pred = np.argmax(outputs.cpu().numpy(), axis=1)
            target = squeezed_targets.cpu().numpy()
            evaluator.add_batch(target, pred)

    if comment[0].__contains__('Val'):
        print("Validation results")
    elif comment[0].__contains__('Test'):
        print("Test results")

    val_indices = comprehensive_evaluation(epoch, evaluator, cfg.DATASETS.CLASS_NAMES)
    val_indices["loss"] = val_loss / len(val_loader)

    if writer is not None:
        writer.add_scalar(comment[0], val_indices["loss"], epoch)
        writer.add_scalar(comment[1], val_indices["mIoU"], epoch)

    return val_indices
