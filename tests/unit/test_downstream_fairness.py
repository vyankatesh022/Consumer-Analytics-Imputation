"""Fairness and demographic disparity validation tests for downstream ML impact."""

import numpy as np
import pandas as pd

from missing_data_platform.downstream.config import DownstreamTaskType
from missing_data_platform.downstream.metrics import (
    calculate_group_disparity,
    calculate_group_downstream_metrics,
)


def test_synthetic_group_differential_imputation_downstream_disparity() -> None:
    """Verify downstream group metrics detect performance disparities when Group A & B experience unequal quality."""
    # Synthetic scenario: Group A has clean accurate predictions, Group B has noise-degraded predictions
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    # Group A (first 8): perfect predictions
    # Group B (last 8): inverted/noisy predictions
    y_pred = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1])
    groups = pd.Series(
        [
            "Group_A",
            "Group_A",
            "Group_A",
            "Group_A",
            "Group_A",
            "Group_A",
            "Group_A",
            "Group_A",
            "Group_B",
            "Group_B",
            "Group_B",
            "Group_B",
            "Group_B",
            "Group_B",
            "Group_B",
            "Group_B",
        ]
    )

    group_metrics = calculate_group_downstream_metrics(
        y_true=y_true,
        y_pred=y_pred,
        group_series=groups,
        task_type=DownstreamTaskType.CLASSIFICATION,
        minimum_group_size=4,
    )

    assert "Group_A" in group_metrics
    assert "Group_B" in group_metrics

    acc_a = group_metrics["Group_A"]["metrics"]["accuracy"]
    acc_b = group_metrics["Group_B"]["metrics"]["accuracy"]

    assert acc_a == 1.0
    assert acc_b == 0.0

    disp = calculate_group_disparity(group_metrics, metric_name="accuracy")
    assert disp["max_disparity"] == 1.0
    assert disp["min_value"] == 0.0
    assert disp["max_value"] == 1.0
    assert disp["disparity_ratio"] == 0.0
