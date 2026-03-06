import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.floating] | npt.NDArray[np.integer]


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
