import numpy as np

from utils.segmentation import metrics_from_confusion


def test_absent_ground_truth_class_with_false_positive_counts_in_macro_average():
    # Class 1 has no GT pixels but receives one false-positive prediction.
    confusion = np.array([[1, 1], [0, 0]], dtype=np.int64)

    metrics = metrics_from_confusion(confusion, mask_values=[0, 1])

    assert metrics["miou"] == 25.0
    assert metrics["dice_score"] == pytest.approx(100 / 3, abs=0.01)
    assert metrics["per_class"]["rooftop"]["included_in_macro"] is True


def test_class_absent_from_ground_truth_and_predictions_is_excluded():
    confusion = np.array([[2, 0], [0, 0]], dtype=np.int64)

    metrics = metrics_from_confusion(confusion, mask_values=[0, 1])

    assert metrics["miou"] == 100.0
    assert metrics["dice_score"] == 100.0
    assert metrics["per_class"]["rooftop"]["included_in_macro"] is False


def test_hand_calculated_multiclass_metrics():
    confusion = np.array([[3, 1], [2, 4]], dtype=np.int64)

    metrics = metrics_from_confusion(confusion, mask_values=[0, 1])

    assert metrics["miou"] == pytest.approx((50 + 4 / 7 * 100) / 2, abs=0.01)
    assert metrics["dice_score"] == pytest.approx((2 * 3 / 9 + 2 * 4 / 11) * 50, abs=0.01)
    assert metrics["fwiou"] == pytest.approx(4 / 10 * 50 + 6 / 10 * (4 / 7 * 100), abs=0.01)
    assert metrics["pixel_accuracy"] == 70.0


import pytest
