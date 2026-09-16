"""Shared AMtown02 label encoding and segmentation metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


AMTOWN02_MASK_VALUES = (0, 1, 2, 3, 5, 6, 13, 14, 15, 16, 17, 19, 20, 24, 25)
AMTOWN02_NAMES = {
    0: "unlabeled",
    1: "rooftop",
    2: "guard_rail",
    3: "road",
    5: "dynamic",
    6: "ground",
    13: "vegetation",
    14: "terrain",
    15: "traffic_sign",
    16: "pole",
    17: "caravan",
    19: "building_facade",
    20: "car",
    24: "truck",
    25: "rider",
}


def encode_mask(mask: np.ndarray, mask_values: Sequence, *, source: str = "mask") -> np.ndarray:
    """Map raw mask labels to class indices and reject undeclared values."""
    array = np.asarray(mask)
    if array.ndim not in (2, 3):
        raise ValueError(f"{source}: masks must have 2 or 3 dimensions, found {array.ndim}")

    values = list(mask_values)
    if not values:
        raise ValueError("mask_values must not be empty")

    result = np.full(array.shape[:2], -1, dtype=np.int64)
    for class_index, value in enumerate(values):
        if array.ndim == 2:
            matches = array == value
        else:
            expected = np.asarray(value)
            if expected.ndim == 0 or expected.shape != (array.shape[-1],):
                raise ValueError(
                    f"{source}: RGB/RGBA masks require one {array.shape[-1]}-value label per class"
                )
            matches = (array == expected).all(axis=-1)
        result[matches] = class_index

    unknown = result < 0
    if unknown.any():
        if array.ndim == 2:
            labels = np.unique(array[unknown]).tolist()
        else:
            labels = np.unique(array[unknown], axis=0).tolist()
        raise ValueError(f"{source}: unknown mask label(s): {labels}")
    return result


def metrics_from_confusion(confusion: np.ndarray, mask_values: Sequence) -> dict:
    """Compute percentage metrics, averaging every class with non-zero union."""
    matrix = np.asarray(confusion)
    values = list(mask_values)
    expected_shape = (len(values), len(values))
    if matrix.shape != expected_shape:
        raise ValueError(f"confusion matrix shape must be {expected_shape}, got {matrix.shape}")
    if (matrix < 0).any():
        raise ValueError("confusion matrix cannot contain negative counts")

    total_pixels = int(matrix.sum())
    per_class = {}
    included_ious = []
    included_dices = []
    weighted_iou = 0.0

    for class_index, raw_value in enumerate(values):
        tp = int(matrix[class_index, class_index])
        gt_count = int(matrix[class_index, :].sum())
        pred_count = int(matrix[:, class_index].sum())
        fp = pred_count - tp
        fn = gt_count - tp
        union = tp + fp + fn
        included = union > 0
        iou = (tp / union * 100.0) if included else 0.0
        dice_denominator = 2 * tp + fp + fn
        dice = (2 * tp / dice_denominator * 100.0) if dice_denominator else 0.0
        frequency = gt_count / total_pixels if total_pixels else 0.0

        if included:
            included_ious.append(iou)
            included_dices.append(dice)
        weighted_iou += frequency * iou

        try:
            numeric_value = int(raw_value)
            class_name = AMTOWN02_NAMES.get(numeric_value, f"class_{numeric_value}")
        except (TypeError, ValueError):
            numeric_value = raw_value
            class_name = f"class_{class_index}"
        per_class[class_name] = {
            "mask_value": numeric_value,
            "iou": round(iou, 2),
            "dice": round(dice, 2),
            "frequency": round(frequency * 100.0, 2),
            "included_in_macro": included,
        }

    return {
        "miou": round(float(np.mean(included_ious)) if included_ious else 0.0, 2),
        "dice_score": round(float(np.mean(included_dices)) if included_dices else 0.0, 2),
        "fwiou": round(weighted_iou, 2),
        "pixel_accuracy": round(float(np.trace(matrix)) / total_pixels * 100.0, 2)
        if total_pixels else 0.0,
        "per_class": per_class,
        "macro_class_count": len(included_ious),
    }
