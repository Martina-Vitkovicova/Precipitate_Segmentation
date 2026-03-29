from typing import Sequence, Dict
import itertools
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

Array = npt.NDArray[np.generic]
WeightMap = Dict[int, Dict[int, float]]
IndexMap = Dict[int, int]


def _to_bin(a: Array, threshold: float) -> npt.NDArray[np.uint8]:
    return (a >= threshold).astype(np.uint8)


def _get_gt_instances(test_lbls: Sequence[Array], component_limit: int):
    gt_instances = []
    for gt in test_lbls:
        _, labels = cv2.connectedComponents((gt > 0).astype(np.uint8))
        labels = labels.copy()
        if labels.max() > component_limit:
            labels[labels >= component_limit] = 0

        areas = np.bincount(labels.ravel())
        gt_instances.append((labels, areas))

    return gt_instances


def compute_metric_curves(
    test_lbls: Sequence[Array],
    test_preds: Sequence[Array],
    thresholds: Sequence[float] | npt.NDArray[np.floating],
    component_limit: int = 1000,
    min_iou: float = 0.0
) -> tuple:
    dice_vals, iou_vals, precision_vals, recall_vals, f1_vals = [], [], [], [], []

    # pre-calculate Ground Truth instances
    gt_instances = _get_gt_instances(test_lbls, component_limit)

    for thr in tqdm(thresholds, desc="Computing metrics across thresholds"):
        dices, ious = [], []
        # Lists to store per-image metrics for averaging at the end
        img_precisions, img_recalls, img_f1s = [], [], []

        for idx, pred in enumerate(test_preds):
            pr_b = _to_bin(pred, float(thr))
            gt_label_map, gt_areas = gt_instances[idx]

            # semantic metrics (pixel-wise)
            gt_b = (gt_label_map > 0).astype(np.uint8)
            inter = int(np.sum(gt_b & pr_b))
            union = int(np.sum(gt_b | pr_b))
            denom = int(np.sum(gt_b) + np.sum(pr_b))

            ious.append(1.0 if union == 0 else inter / union)
            dices.append(1.0 if denom == 0 else (2.0 * inter) / denom)

            # instance metrics
            df = match_instances_optimised(pr_b, gt_label_map, gt_areas, component_limit, min_iou)

            tp = int((df["pred_id"].notna() & df["label_id"].notna()).sum())
            fp = int((df["pred_id"].notna() & df["label_id"].isna()).sum())
            fn = int((df["pred_id"].isna() & df["label_id"].notna()).sum())

            # calculate per-image stats
            p = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            r = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            f = 2.0 * p * r / (p + r) if (p + r) > 0 else (0.0 if (p + r) == 0 else np.nan)

            img_precisions.append(p)
            img_recalls.append(r)
            img_f1s.append(f)

        # average the per-image scores = macro-average
        m_prec = float(np.nanmean(img_precisions))
        m_rec = float(np.nanmean(img_recalls))

        if (m_prec + m_rec) > 0:
            m_f1 = 2.0 * (m_prec * m_rec) / (m_prec + m_rec)
        else:
            m_f1 = np.nan

        precision_vals.append(m_prec)
        recall_vals.append(m_rec)
        f1_vals.append(m_f1)

        dice_vals.append(float(np.nanmean(dices)))
        iou_vals.append(float(np.nanmean(ious)))

    return dice_vals, iou_vals, precision_vals, recall_vals, f1_vals


def match_instances_optimised(
    bin_prediction: Array,
    l_grains: Array,
    l_areas: Array,
    component_limit: int,
    min_iou: float
) -> pd.DataFrame:
    p_n, p_grains = cv2.connectedComponents(bin_prediction)

    if p_n > component_limit:
        p_grains[p_grains >= component_limit] = 0
        p_n = component_limit

    p_areas = np.bincount(p_grains.ravel())
    l_n = len(l_areas)

    # get weights only for candidates that actually overlap
    weights_dict = _collect_pairing_weights_optimised(p_grains, l_grains, l_areas, p_areas, min_iou)

    if not weights_dict:
        data = [(i, None) for i in range(1, p_n)] + [(None, i) for i in range(1, l_n)]
        return pd.DataFrame(data, columns=["pred_id", "label_id"])

    # solve assignment only for overlapping pairs
    weights, p_map, l_map = _construct_weight_map(weights_dict)
    p_idx_array, l_idx_array = linear_sum_assignment(weights)

    inv_p = {v: k for k, v in p_map.items()}
    inv_l = {v: k for k, v in l_map.items()}

    matched_p = set()
    matched_l = set()
    data = []

    for pi, li in zip(p_idx_array, l_idx_array):
        actual_p_id = inv_p[pi]
        actual_l_id = inv_l[li]

        if actual_l_id in weights_dict.get(actual_p_id, {}):
            data.append((actual_p_id, actual_l_id))
            matched_p.add(actual_p_id)
            matched_l.add(actual_l_id)

    data += [(i, None) for i in range(1, p_n) if i not in matched_p]
    data += [(None, i) for i in range(1, l_n) if i not in matched_l]

    return pd.DataFrame(data, columns=["pred_id", "label_id"])


def _collect_pairing_weights_optimised(
    p_grains: Array,
    l_grains: Array,
    l_areas: Array,
    p_areas: Array,
    min_iou: float
) -> WeightMap:
    max_l = len(l_areas)
    combined = p_grains.astype(np.uint64) * max_l + l_grains.astype(np.uint64)

    mask = (p_grains > 0) & (l_grains > 0)
    if not np.any(mask):
        return {}

    ids, intersections = np.unique(combined[mask], return_counts=True)

    weights_dict = {}
    for joint_id, inter in zip(ids, intersections):
        p_id = int(joint_id // max_l)
        l_id = int(joint_id % max_l)

        iou = inter / (p_areas[p_id] + l_areas[l_id] - inter)
        if iou >= min_iou:
            weights_dict.setdefault(p_id, {})[l_id] = 1.0 - float(iou)

    return weights_dict


def _construct_weight_map(weights_dict: WeightMap) -> tuple[Array, IndexMap, IndexMap]:
    p_keys = sorted(weights_dict.keys())
    p_map = {v: i for i, v in enumerate(p_keys)}

    l_unique = sorted(list(set(itertools.chain(*(v.keys() for v in weights_dict.values())))))
    l_map = {v: i for i, v in enumerate(l_unique)}

    weights = np.ones((len(p_map), len(l_map)), dtype=np.float32)
    for p, pv in weights_dict.items():
        for ll, lv in pv.items():
            weights[p_map[p], l_map[ll]] = lv

    return weights, p_map, l_map


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
            f.write(f"{float(thr):.4f}\t{float(d):.6f}\t{float(i):.6f}\t{float(p):.6f}\t{float(r):.6f}\t"
                    f"{float(f1_val):.6f}\n")
