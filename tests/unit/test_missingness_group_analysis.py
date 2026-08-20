"""Unit tests for demographic and group-level missingness disparity analysis."""

import numpy as np
import pandas as pd

from missing_data_platform.missingness.group_analysis import (
    analyze_feature_missingness_by_group,
    create_age_bands,
)


def test_create_age_bands() -> None:
    """Verify age binning into standard cohorts."""
    ages = pd.Series([15, 20, 30, 40, 50, 65, np.nan])
    bands = create_age_bands(ages)

    assert bands.iloc[0] == "<18"
    assert bands.iloc[1] == "18-24"
    assert bands.iloc[2] == "25-34"
    assert bands.iloc[3] == "35-44"
    assert bands.iloc[4] == "45-54"
    assert bands.iloc[5] == "55+"
    assert bands.iloc[6] == "Unknown"


def test_analyze_feature_missingness_by_group_disparity() -> None:
    """Verify calculation of group-level missingness and disparity ratio."""
    df = pd.DataFrame(
        {
            "region": ["West", "West", "East", "East"],
            "income": [50000.0, 60000.0, np.nan, 80000.0],  # West: 0% missing, East: 50% missing
        }
    )

    disparity = analyze_feature_missingness_by_group(
        df=df,
        target_feature="income",
        grouping_column="region",
    )

    assert disparity.feature_analyzed == "income"
    assert disparity.grouping_variable == "region"
    assert len(disparity.groups) == 2

    west_stat = next(g for g in disparity.groups if g.group_value == "West")
    east_stat = next(g for g in disparity.groups if g.group_value == "East")

    assert west_stat.missing_percentage == 0.0
    assert east_stat.missing_percentage == 50.0
    assert disparity.disparity_ratio > 1.0
