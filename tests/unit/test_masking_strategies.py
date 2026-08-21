"""Unit tests for artificial masking strategy sampling algorithms."""

import numpy as np
import pandas as pd

from missing_data_platform.masking.strategies import (
    mask_mar_covariate_conditioned,
    mask_stratified_by_group,
    mask_uniform_random,
)


def test_mask_uniform_random_rate() -> None:
    """Verify uniform random masking satisfies configured mask rate."""
    rng = np.random.default_rng(42)
    series = pd.Series([10.0] * 100)

    mask = mask_uniform_random(series, mask_rate=0.20, rng=rng)
    assert mask.sum() == 20
    assert len(mask) == 100


def test_mask_uniform_random_preserves_natural_missingness() -> None:
    """Verify that naturally missing cells are not eligible for artificial masking."""
    rng = np.random.default_rng(42)
    series = pd.Series([10.0] * 50 + [np.nan] * 50)  # 50 eligible cells

    mask = mask_uniform_random(series, mask_rate=0.20, rng=rng)
    assert mask.sum() == 10  # 20% of 50 eligible
    # Ensure all masked cells are from the observed first 50
    assert (mask[50:]).sum() == 0


def test_mask_mar_covariate_conditioned() -> None:
    """Verify MAR covariate-conditioned masking produces target mask count."""
    rng = np.random.default_rng(42)
    series = pd.Series([100.0] * 100)
    covariate = pd.Series(range(100))

    mask = mask_mar_covariate_conditioned(
        series=series,
        covariate_series=covariate,
        base_mask_rate=0.30,
        rng=rng,
    )
    assert mask.sum() == 30


def test_mask_stratified_by_group() -> None:
    """Verify stratified masking samples proportionally across demographic cohorts."""
    rng = np.random.default_rng(42)
    series = pd.Series([10.0] * 100)
    groups = pd.Series(["A"] * 50 + ["B"] * 50)

    mask = mask_stratified_by_group(
        series=series,
        group_series=groups,
        mask_rate=0.20,
        rng=rng,
    )
    assert mask[:50].sum() == 10  # 20% of Group A
    assert mask[50:].sum() == 10  # 20% of Group B
    assert mask.sum() == 20
