"""Unit tests for missingness indicator generation and correlation matrix."""

import numpy as np
import pandas as pd

from missing_data_platform.missingness.indicators import (
    compute_missingness_correlation,
    create_missingness_indicators,
)


def test_create_missingness_indicators() -> None:
    """Verify conversion of DataFrame into binary indicator matrix."""
    df = pd.DataFrame(
        {
            "age": [25.0, np.nan, 30.0],
            "income": [np.nan, np.nan, 50000.0],
            "city": ["NYC", "LA", "CHI"],
        }
    )

    indicators = create_missingness_indicators(df)
    assert "is_missing_age" in indicators.columns
    assert "is_missing_income" in indicators.columns
    assert "is_missing_city" in indicators.columns

    assert list(indicators["is_missing_age"]) == [0, 1, 0]
    assert list(indicators["is_missing_income"]) == [1, 1, 0]
    assert list(indicators["is_missing_city"]) == [0, 0, 0]


def test_compute_missingness_correlation() -> None:
    """Verify calculation of inter-feature missingness correlation matrix."""
    indicators = pd.DataFrame(
        {
            "is_missing_age": [0, 1, 0, 1],
            "is_missing_income": [0, 1, 0, 1],  # Perfectly correlated with age
            "is_missing_city": [0, 0, 0, 0],  # Invariant 0
        }
    )

    corr = compute_missingness_correlation(indicators)
    assert corr.shape == (3, 3)
    assert np.isclose(corr.loc["is_missing_age", "is_missing_income"], 1.0)
    assert corr.loc["is_missing_age", "is_missing_city"] == 0.0
