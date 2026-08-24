"""Unit tests for downstream mathematical and statistical evaluation metrics."""

import numpy as np
import pandas as pd

from missing_data_platform.downstream.config import DownstreamTaskType
from missing_data_platform.downstream.metrics import (
    calculate_classification_metrics,
    calculate_group_disparity,
    calculate_group_downstream_metrics,
    calculate_imputation_downstream_correlation,
    calculate_performance_delta,
    calculate_performance_recovery,
    calculate_regression_metrics,
)


def test_classification_metrics_computation() -> None:
    """Assert accuracy, precision, recall, f1, and roc_auc calculation."""
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 0, 1, 1])
    y_prob = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.3, 0.85, 0.6])

    metrics = calculate_classification_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 0.75
    assert metrics["f1"] is not None
    assert metrics["precision"] is not None
    assert metrics["recall"] is not None
    assert metrics["roc_auc"] is not None
    assert metrics["pr_auc"] is not None


def test_regression_metrics_computation() -> None:
    """Assert mae, rmse, r2, and nrmse calculation."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 19.0, 29.0, 42.0])

    metrics = calculate_regression_metrics(y_true, y_pred)

    assert metrics["mae"] == 1.5
    assert metrics["rmse"] is not None
    assert metrics["r2"] is not None
    assert metrics["nrmse"] is not None


def test_performance_delta_directionality() -> None:
    """Assert delta computation preserves directionality."""
    complete = {"f1": 0.85, "rmse": 10.0}
    imputed = {"f1": 0.80, "rmse": 12.5}

    delta = calculate_performance_delta(complete, imputed)

    # For higher-is-better (f1): 0.80 - 0.85 = -0.05
    assert delta["f1"] == -0.05
    # For lower-is-better (rmse): 12.5 - 10.0 = +2.5
    assert delta["rmse"] == 2.5


def test_performance_recovery_formula() -> None:
    """Assert formal recovery calculation for higher-is-better and lower-is-better metrics."""
    # Higher is better: complete = 0.90, baseline_missing = 0.50, candidate = 0.80
    # Recovery = (0.80 - 0.50) / (0.90 - 0.50) = 0.30 / 0.40 = 75.0%
    rec_f1 = calculate_performance_recovery(
        complete_val=0.90,
        baseline_missing_val=0.50,
        imputed_val=0.80,
        metric_name="f1",
    )
    assert rec_f1 == 75.0

    # Lower is better: complete = 2.0, baseline_missing = 10.0, candidate = 4.0
    # Recovery = (10.0 - 4.0) / (10.0 - 2.0) = 6.0 / 8.0 = 75.0%
    rec_rmse = calculate_performance_recovery(
        complete_val=2.0,
        baseline_missing_val=10.0,
        imputed_val=4.0,
        metric_name="rmse",
    )
    assert rec_rmse == 75.0


def test_group_downstream_metrics_and_disparity() -> None:
    """Assert demographic slicing and disparity computation."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    groups = pd.Series(["Gold", "Gold", "Gold", "Gold", "Bronze", "Bronze", "Bronze", "Bronze"])

    group_metrics = calculate_group_downstream_metrics(
        y_true=y_true,
        y_pred=y_pred,
        group_series=groups,
        task_type=DownstreamTaskType.CLASSIFICATION,
        minimum_group_size=2,
    )

    assert "Gold" in group_metrics
    assert "Bronze" in group_metrics
    assert group_metrics["Gold"]["sample_count"] == 4
    assert group_metrics["Bronze"]["sample_count"] == 4

    disp = calculate_group_disparity(group_metrics, metric_name="accuracy")
    assert disp["max_disparity"] is not None
    assert disp["disparity_ratio"] is not None


def test_imputation_downstream_correlation() -> None:
    """Assert Spearman rank correlation between imputation MAE and downstream F1."""
    imputation_maes = [0.1, 0.2, 0.4, 0.8, 1.2]
    downstream_f1s = [0.95, 0.88, 0.82, 0.65, 0.50]

    corr = calculate_imputation_downstream_correlation(imputation_maes, downstream_f1s)
    assert corr["spearman_rho"] is not None
    assert corr["spearman_rho"] < 0  # Negative correlation: lower MAE -> higher F1
