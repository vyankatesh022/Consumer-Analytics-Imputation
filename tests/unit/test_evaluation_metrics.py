"""Unit tests for evaluation metrics (MAE, RMSE, NRMSE, Accuracy) and validation filters."""

import numpy as np
import pandas as pd
import pytest

from missing_data_platform.evaluation.metrics import (
    calculate_accuracy,
    calculate_mae,
    calculate_nrmse,
    calculate_rmse,
    validate_and_filter_predictions,
)
from missing_data_platform.exceptions import EvaluationError


def test_calculate_mae_exact() -> None:
    """Verify MAE calculation on exact numerical values."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 35.0, 40.0])  # diffs: 2, 2, 5, 0 -> sum=9 -> mean=2.25
    mae = calculate_mae(y_true, y_pred)
    assert mae == 2.25


def test_calculate_rmse_exact() -> None:
    """Verify RMSE calculation on exact numerical values."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array(
        [14.0, 20.0, 27.0]
    )  # diffs: 4, 0, -3 -> sq: 16, 0, 9 -> mean=25/3 -> sqrt=2.88675
    rmse = calculate_rmse(y_true, y_pred)
    assert round(rmse, 4) == round(np.sqrt(25.0 / 3.0), 4)


def test_calculate_nrmse() -> None:
    """Verify NRMSE calculation with non-zero standard deviation."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([11.0, 19.0, 31.0, 39.0])
    nrmse = calculate_nrmse(y_true, y_pred)
    assert nrmse > 0.0


def test_calculate_accuracy_categorical() -> None:
    """Verify categorical accuracy calculation."""
    y_true = pd.Series(["Gold", "Silver", "Platinum", "Bronze"])
    y_pred = pd.Series(["Gold", "Silver", "Gold", "Bronze"])
    acc = calculate_accuracy(y_true, y_pred)
    assert acc == 0.75


def test_metrics_handle_missing_and_invalid_predictions() -> None:
    """Verify metrics filter missing and infinite predictions while tracking counts."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([10.0, np.nan, 30.0, np.inf])

    clean_true, clean_pred, missing_count, invalid_count = validate_and_filter_predictions(
        y_true, y_pred
    )
    assert len(clean_true) == 2
    assert len(clean_pred) == 2
    assert missing_count == 1
    assert invalid_count == 1

    # MAE should be computed on valid pairs ([10, 30] vs [10, 30]) -> MAE = 0.0
    mae = calculate_mae(y_true, y_pred)
    assert mae == 0.0


def test_metrics_shape_mismatch_raises_error() -> None:
    """Verify shape mismatch raises EvaluationError."""
    with pytest.raises(EvaluationError):
        calculate_mae(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_metrics_empty_input_raises_error() -> None:
    """Verify empty input raises EvaluationError."""
    with pytest.raises(EvaluationError):
        calculate_mae(np.array([]), np.array([]))
