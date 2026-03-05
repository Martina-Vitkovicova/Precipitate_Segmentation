# Adapted from semseg/instance_matching.py in jaroslavknotek/semantic-segmentation.
# Source: https://github.com/jaroslavknotek/semantic-segmentation/blob/main/src/semseg
# Changes: corrected cost-matrix initialization, empty-overlap guard, tqdm-based warnings.

import itertools
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from src.precipseg.metrics import binary_iou

Array = npt.NDArray
WeightMap = dict[int, dict[int, float]]
IndexMap = dict[int, int]


@dataclass
class SegmentationInstance:
    """Instance shape"""

    mask: Array
    image: Array
    top_left_x: int
    top_left_y: int
    width: int
    height: int


@dataclass
class SegmentationInstanceFeatures:
    """Precipitate features"""

    ellipse_width_px: float
    ellipse_height_px: float
    ellipse_center_x: float
    ellipse_center_y: float
    ellipse_angle_deg: float
    circle_x: float
    circle_y: float
    circle_radius: float
    area_px: float
    shape: str


def measure_precision_recall_f1(
    bin_prediction: Array,
    label: Array,
    component_limit: int = 500,
) -> Tuple[float, float, float]:
    df, _, _ = match_instances(bin_prediction, label, component_limit=component_limit,)

    grains_pred = len(df[~df["pred_id"].isna()])
    grains_label = len(df[~df["label_id"].isna()])

    tp = df[~df["label_id"].isna() & ~df["pred_id"].isna()]

    precision = len(tp) / grains_pred if grains_pred != 0 else np.nan
    recall = len(tp) / grains_label if grains_label != 0 else np.nan

    if precision + recall != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def match_instances(
    bin_prediction: Array,
    label: Array,
    component_limit: int = 500,
) -> Tuple[pd.DataFrame, Array, Array]:
    bin_prediction = (bin_prediction > 0).astype(np.uint8)
    label = (label > 0).astype(np.uint8)

    # p_n / l_n include background label 0
    p_n, p_grains = cv2.connectedComponents(bin_prediction)
    l_n, l_grains = cv2.connectedComponents(label)

    if p_n > component_limit or l_n > component_limit:
        tqdm.write(f"Too many components found {component_limit=} "
                   f"#predictions:{p_n} #labels:{l_n}. Cropping")
        p_n = min(p_n, component_limit)
        p_grains[p_grains > component_limit] = 0
        l_n = min(l_n, component_limit)
        l_grains[l_grains > component_limit] = 0

    # pairs only #TP candidates
    pred_items, label_items = _pair_using_linear_sum_assignment(p_n, p_grains, l_grains)
    data = list(zip(pred_items, label_items))

    # FP
    p_set = set(pred_items.tolist() if isinstance(pred_items, np.ndarray) else pred_items)
    false_positives = [i for i in range(1, p_n) if i not in p_set]
    for i in false_positives:
        data.append((i, None))

    # FN
    l_set = set(label_items.tolist() if isinstance(label_items, np.ndarray) else label_items)
    label_positives = [i for i in range(1, l_n) if i not in l_set]
    for i in label_positives:
        data.append((None, i))

    df = pd.DataFrame(data, columns=["pred_id", "label_id"])
    return df, p_grains, l_grains


def _pair_using_linear_sum_assignment(
    p_n: int,
    p_grains: Array,
    l_grains: Array,
    cap: Optional[int] = 500,
) -> Tuple[Array, Array]:
    if cap is not None:
        p_n = min(cap, p_n)
        p_grains[p_grains > cap] = 0
        l_grains[l_grains > cap] = 0

    weights_dict = _collect_pairing_weights(p_n, p_grains, l_grains)
    if not weights_dict:
        return np.array([], dtype=int), np.array([], dtype=int)

    weights, p_map, l_map = _construct_weight_map(weights_dict)
    p_item_id, l_item_id = linear_sum_assignment(weights)

    inverse_p_map = {v: k for k, v in p_map.items()}
    p_item = np.array([inverse_p_map[idx] for idx in p_item_id], dtype=int)

    inverse_l_map = {v: k for k, v in l_map.items()}
    l_item = np.array([inverse_l_map[idx] for idx in l_item_id], dtype=int)

    return p_item, l_item


def _construct_weight_map(weights_dict: WeightMap) -> tuple[Array, IndexMap, IndexMap]:
    p_map = {v: i for i, v in enumerate(weights_dict.keys())}

    l_keys = itertools.chain(*(list(k for k in v.keys()) for v in weights_dict.values()))
    l_unique = np.unique(list(l_keys))
    l_map = {v: i for i, v in enumerate(l_unique)}

    weights = np.ones((len(p_map), len(l_map)), dtype=np.float32)
    for p, pv in weights_dict.items():
        for ll, lv in pv.items():
            weights[p_map[p], l_map[ll]] = lv

    return weights, p_map, l_map


def _collect_pairing_weights(p_n: int, p_grains: Array, l_grains: Array,) -> WeightMap:
    weights_dict = {}

    for p_grain_id in range(1, p_n):
        p_grain_mask = np.uint8(p_grains == p_grain_id)

        intersecting_ids = np.unique(l_grains * p_grain_mask)
        intersecting_ids = intersecting_ids[intersecting_ids > 0]

        for l_grain_id in intersecting_ids:
            l_grain_mask = np.uint8(l_grains == l_grain_id)

            # Hungarian solves a minimum-cost assignment, so weight = 1 - IoU.
            weight = 1.0 - binary_iou(l_grain_mask, p_grain_mask, threshold=0.5)
            weights_dict.setdefault(p_grain_id, {}).setdefault(l_grain_id, float(weight))

    return weights_dict
