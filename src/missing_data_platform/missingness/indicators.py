"""Missingness Indicator transformation and correlation analysis.

Provides functions to construct binary missingness indicators R_ij in {0, 1}
and compute inter-feature missingness correlation matrices without mutating the source dataset.
"""

import pandas as pd


def create_missingness_indicators(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    prefix: str = "is_missing_",
) -> pd.DataFrame:
    """Create a DataFrame of binary missingness indicators (1 if null, 0 if observed)."""
    target_cols = columns or list(df.columns)
    indicators = pd.DataFrame(index=df.index)

    for col in target_cols:
        if col in df.columns:
            indicators[f"{prefix}{col}"] = df[col].isna().astype(int)

    return indicators


def compute_missingness_correlation(
    indicator_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Pearson/phi correlation matrix between binary missingness indicators.

    Safely handles invariant columns (0% or 100% missing) by filling NaN correlations with 0.0.
    """
    if indicator_df.empty or len(indicator_df.columns) == 0:
        return pd.DataFrame()

    corr_matrix = indicator_df.corr(method="pearson").fillna(0.0)
    return corr_matrix
