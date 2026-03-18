from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.floating] | npt.NDArray[np.integer]
Metrics = tuple[list[float], list[float], list[float], list[float], list[float]]


def _to_bin(a: Array, threshold: float) -> npt.NDArray[np.uint8]:
    return (a >= threshold).astype(np.uint8)


def confusion_counts(gt: Array, pred: Array, threshold: float = 0.5) -> tuple[int, int, int, int]:
    gt_b = (gt > 0).astype(np.uint8)
    pr_b = _to_bin(pred, threshold)

    tp = int(np.sum((gt_b == 1) & (pr_b == 1)))
    fp = int(np.sum((gt_b == 0) & (pr_b == 1)))
    fn = int(np.sum((gt_b == 1) & (pr_b == 0)))
    tn = int(np.sum((gt_b == 0) & (pr_b == 0)))
    return tp, fp, fn, tn


def binary_iou(gt: Array, pred: Array, threshold: float = 0.5) -> float:
    gt_b = (gt > 0).astype(np.uint8)
    pr_b = _to_bin(pred, threshold)

    inter = int(np.sum(gt_b & pr_b))
    union = int(np.sum(gt_b | pr_b))
    return 1.0 if union == 0 else inter / union


def binary_dice(gt: Array, pred: Array, threshold: float = 0.5) -> float:
    gt_b = (gt > 0).astype(np.uint8)
    pr_b = _to_bin(pred, threshold)

    inter = int(np.sum(gt_b & pr_b))
    denom = int(np.sum(gt_b) + np.sum(pr_b))
    return 1.0 if denom == 0 else (2.0 * inter) / denom


def binary_precision_recall_f1(gt: Array, pred: Array, threshold: float = 0.5) -> tuple[float, float, float]:
    tp, fp, fn, _ = confusion_counts(gt, pred, threshold)

    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan

    if not np.isfinite(precision) or not np.isfinite(recall) or (precision + recall) == 0:
        f1 = np.nan
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return float(precision), float(recall), float(f1)


def compute_metric_curves(
    test_lbls: Sequence[Array],
    test_preds: Sequence[Array],
    thresholds: Sequence[float] | npt.NDArray[np.floating],
) -> Metrics:
    dice_vals, iou_vals, precision_vals, recall_vals, f1_vals = [], [], [], [], []

    for thr in thresholds:
        dices, ious, precisions, recalls, f1s = [], [], [], [], []

        for gt, pred in zip(test_lbls, test_preds):
            dices.append(binary_dice(gt, pred, float(thr)))
            ious.append(binary_iou(gt, pred, float(thr)))

            p, r, f1 = binary_precision_recall_f1(gt, pred, float(thr))
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

        dice_vals.append(float(np.nanmean(dices)))
        iou_vals.append(float(np.nanmean(ious)))
        precision_vals.append(float(np.nanmean(precisions)))
        recall_vals.append(float(np.nanmean(recalls)))
        f1_vals.append(float(np.nanmean(f1s)))

    return dice_vals, iou_vals, precision_vals, recall_vals, f1_vals


def save_metric_curves(
    path: Path,
    thresholds: Sequence[float] | npt.NDArray[np.floating],
    dice: Sequence[float],
    iou: Sequence[float],
    precision: Sequence[float],
    recall: Sequence[float],
    f1: Sequence[float],
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("threshold\tdice\tiou\tprecision\trecall\tf1\n")
        for row in zip(thresholds, dice, iou, precision, recall, f1):
            thr, d, i, p, r, f1_val = row
            f.write(f"{float(thr):.4f}\t{float(d):.6f}\t{float(i):.6f}\t{float(p):.6f}\t{float(r):.6f}\t{float(f1_val):.6f}\n")
