"""Unit tests for sample weighting calculations and clipping."""

import numpy as np
import pandas as pd

from missing_data_platform.mitigation.weighting import calculate_group_sample_weights


def test_balanced_sample_weights_calculation() -> None:
    """Verify sample weights balance imbalanced demographic groups."""
    # 80 items GroupA, 20 items GroupB (total 100, K=2)
    # Target w_A = 100 / (2 * 80) = 0.625
    # Target w_B = 100 / (2 * 20) = 2.500
    group_series = pd.Series(["GroupA"] * 80 + ["GroupB"] * 20)
    weights, was_clipped = calculate_group_sample_weights(group_series, max_weight=5.0)

    assert len(weights) == 100
    assert was_clipped is False
    assert round(weights[0], 3) == 0.625
    assert round(weights[80], 3) == 2.500


def test_sample_weight_clipping() -> None:
    """Verify extreme imbalanced groups are clipped to max_sample_weight."""
    # 99 items GroupA, 1 item GroupB (total 100, K=2)
    # Raw w_B = 100 / (2 * 1) = 50.0
    group_series = pd.Series(["GroupA"] * 99 + ["GroupB"] * 1)
    weights, was_clipped = calculate_group_sample_weights(group_series, max_weight=4.0)

    assert was_clipped is True
    assert weights[99] == 4.0  # Clipped at 4.0


def test_single_group_sample_weights() -> None:
    """Verify single group returns all ones."""
    group_series = pd.Series(["GroupA"] * 10)
    weights, was_clipped = calculate_group_sample_weights(group_series)
    assert np.all(weights == 1.0)
    assert was_clipped is False
