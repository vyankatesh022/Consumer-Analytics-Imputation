"""Balanced inverse-frequency sample weighting utilities for fairness-aware training."""

import numpy as np
import pandas as pd


def calculate_group_sample_weights(
    group_series: pd.Series,
    max_weight: float = 5.0,
) -> tuple[np.ndarray, bool]:
    """Calculate balanced inverse-frequency sample weights for demographic cohorts.

    Formula:
        w_g = N / (K * N_g)
    where N is total samples, K is number of unique groups, and N_g is group sample count.
    Weights are clipped at max_weight to maintain numerical stability.

    Args:
        group_series: Series containing group labels for each row.
        max_weight: Maximum allowable sample weight before clipping.

    Returns:
        tuple[np.ndarray, bool]:
            - weights: 1D numpy array of sample weights aligned with group_series.
            - was_clipped: Boolean indicating whether any weight exceeded max_weight and was clipped.
    """
    total_n = len(group_series)
    if total_n == 0:
        return np.array([]), False

    clean_groups = group_series.fillna("Unknown").astype(str)
    counts = clean_groups.value_counts()
    k_groups = len(counts)

    if k_groups <= 1:
        return np.ones(total_n, dtype=float), False

    # Compute balanced weights per group
    group_weights: dict[str, float] = {}
    was_clipped = False

    for g_val, n_g in counts.items():
        raw_w = total_n / (k_groups * n_g)
        if raw_w > max_weight:
            w = max_weight
            was_clipped = True
        else:
            w = raw_w
        group_weights[g_val] = float(w)

    sample_weights = clean_groups.map(group_weights).to_numpy(dtype=float)
    return sample_weights, was_clipped
