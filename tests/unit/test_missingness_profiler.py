"""Unit tests for feature-level, row-level, and combinatorial pattern profilers."""

import numpy as np
import pandas as pd

from missing_data_platform.missingness.profiler import (
    profile_feature_missingness,
    profile_missingness_patterns,
    profile_row_missingness,
)


def test_profile_feature_missingness_ranking() -> None:
    """Verify feature missingness profiling and descending ranking."""
    df = pd.DataFrame(
        {
            "complete_col": [1, 2, 3, 4],
            "low_missing": [1, np.nan, 3, 4],  # 25%
            "high_missing": [np.nan, np.nan, np.nan, 4],  # 75%
        }
    )

    profiles = profile_feature_missingness(df)
    assert len(profiles) == 3
    assert profiles[0].column_name == "high_missing"
    assert profiles[0].missing_percentage == 75.0
    assert profiles[0].rank == 1

    assert profiles[1].column_name == "low_missing"
    assert profiles[1].missing_percentage == 25.0
    assert profiles[1].rank == 2

    assert profiles[2].column_name == "complete_col"
    assert profiles[2].missing_percentage == 0.0
    assert profiles[2].rank == 3


def test_profile_row_missingness_distribution() -> None:
    """Verify row-level missingness calculation."""
    df = pd.DataFrame(
        {
            "a": [1, np.nan, np.nan, 4],
            "b": [1, 2, np.nan, 4],
        }
    )

    row_prof = profile_row_missingness(df)
    assert row_prof.total_rows == 4
    assert row_prof.completely_observed_rows == 2  # rows 0 and 3
    assert row_prof.completely_observed_percentage == 50.0
    assert row_prof.rows_with_any_missing == 2
    assert row_prof.max_missing_columns_in_a_row == 2  # row 2 has both a and b missing


def test_profile_missingness_patterns() -> None:
    """Verify identification of combinatorial missing patterns."""
    df = pd.DataFrame(
        {
            "a": [1, np.nan, np.nan],
            "b": [1, np.nan, 3],
        }
    )

    patterns = profile_missingness_patterns(df)
    assert len(patterns) >= 2
    # Check that pattern with both a and b missing is identified
    both_missing_pat = next(
        (p for p in patterns if "a" in p.missing_columns and "b" in p.missing_columns), None
    )
    assert both_missing_pat is not None
    assert both_missing_pat.row_count == 1
