"""Masking sampling algorithms and strategies for artificial missingness generation."""

import numpy as np
import pandas as pd


def mask_uniform_random(
    series: pd.Series,
    mask_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly select a fraction of currently observed cells for masking (MCAR simulation)."""
    observed_indices = np.where(series.notna())[0]
    n_eligible = len(observed_indices)

    if n_eligible == 0 or mask_rate <= 0.0:
        return np.zeros(len(series), dtype=bool)

    n_to_mask = int(round(n_eligible * mask_rate))
    n_to_mask = min(n_to_mask, n_eligible)

    selected_observed = rng.choice(observed_indices, size=n_to_mask, replace=False)
    mask = np.zeros(len(series), dtype=bool)
    mask[selected_observed] = True
    return mask


def mask_mar_covariate_conditioned(
    series: pd.Series,
    covariate_series: pd.Series,
    base_mask_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select cells for masking with probability conditioned on an observed auxiliary covariate (MAR simulation)."""
    observed_mask = series.notna().to_numpy()
    n_eligible = int(observed_mask.sum())

    if n_eligible == 0 or base_mask_rate <= 0.0:
        return np.zeros(len(series), dtype=bool)

    # Compute probability weights based on covariate values
    if pd.api.types.is_numeric_dtype(covariate_series):
        clean_cov = pd.to_numeric(covariate_series, errors="coerce").fillna(
            covariate_series.median()
        )
        # Min-max scale covariate to [0.1, 0.9] to avoid 0 probability
        min_v, max_v = clean_cov.min(), clean_cov.max()
        if max_v > min_v:
            norm_weights = (clean_cov - min_v) / (max_v - min_v)
            probs = 0.2 + 0.8 * norm_weights.to_numpy()
        else:
            probs = np.ones(len(series))
    else:
        # Categorical covariate: assign deterministic hash weights per category
        cats = covariate_series.astype(str)
        unique_cats = sorted(cats.unique())
        cat_weights = {cat: (idx + 1) / len(unique_cats) for idx, cat in enumerate(unique_cats)}
        probs = np.array([cat_weights.get(c, 0.5) for c in cats])

    # Zero out probabilities for cells that are already missing
    probs[~observed_mask] = 0.0
    sum_probs = probs.sum()

    if sum_probs == 0:
        return np.zeros(len(series), dtype=bool)

    normalized_probs = probs / sum_probs
    n_to_mask = int(round(n_eligible * base_mask_rate))
    n_to_mask = min(n_to_mask, n_eligible)

    eligible_indices = np.where(observed_mask)[0]
    eligible_probs = normalized_probs[eligible_indices]
    eligible_probs /= eligible_probs.sum()

    selected = rng.choice(eligible_indices, size=n_to_mask, replace=False, p=eligible_probs)
    mask = np.zeros(len(series), dtype=bool)
    mask[selected] = True
    return mask


def mask_stratified_by_group(
    series: pd.Series,
    group_series: pd.Series,
    mask_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample cells for masking proportionally within each stratum of a grouping variable."""
    observed_mask = series.notna()
    mask = np.zeros(len(series), dtype=bool)

    groups = group_series.fillna("Unknown").astype(str)
    for _, group_indices in groups.groupby(groups).groups.items():
        idx_array = np.array(group_indices)
        eligible_in_group = idx_array[observed_mask.iloc[idx_array]]
        n_eligible_grp = len(eligible_in_group)

        if n_eligible_grp > 0 and mask_rate > 0.0:
            n_to_mask = int(round(n_eligible_grp * mask_rate))
            n_to_mask = min(n_to_mask, n_eligible_grp)
            selected_grp = rng.choice(eligible_in_group, size=n_to_mask, replace=False)
            mask[selected_grp] = True

    return mask
