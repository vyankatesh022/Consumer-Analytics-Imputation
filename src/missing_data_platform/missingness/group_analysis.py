"""Customer demographic and segment group-level missingness disparity analysis."""

import pandas as pd

from missing_data_platform.missingness.report import (
    GroupDisparitySummary,
    GroupMissingnessStat,
)


def create_age_bands(age_series: pd.Series) -> pd.Series:
    """Discretize continuous age values into standard demographic cohorts."""
    numeric_age = pd.to_numeric(age_series, errors="coerce")

    bins = [-float("inf"), 17, 24, 34, 44, 54, float("inf")]
    labels = ["<18", "18-24", "25-34", "35-44", "45-54", "55+"]

    binned = pd.cut(numeric_age, bins=bins, labels=labels, right=True)
    # Convert Categorical to object to allow filling missing with 'Unknown'
    return binned.astype(object).fillna("Unknown")


def analyze_feature_missingness_by_group(
    df: pd.DataFrame,
    target_feature: str,
    grouping_column: str,
) -> GroupDisparitySummary:
    """Analyze missingness rates for a target feature across distinct groups of a grouping variable."""
    if target_feature not in df.columns or grouping_column not in df.columns:
        return GroupDisparitySummary(
            feature_analyzed=target_feature,
            grouping_variable=grouping_column,
            min_missing_group="N/A",
            min_missing_percentage=0.0,
            max_missing_group="N/A",
            max_missing_percentage=0.0,
            disparity_ratio=1.0,
            groups=[],
        )

    # Derive grouping series (handling age bands if grouping by age)
    if grouping_column == "age":
        group_series = create_age_bands(df["age"])
        group_name = "age_band"
    else:
        group_series = df[grouping_column].fillna("Missing/Unknown").astype(str)
        group_name = grouping_column

    is_missing_target = df[target_feature].isna()
    temp_df = pd.DataFrame(
        {
            "group": group_series,
            "is_missing": is_missing_target,
        }
    )

    grouped = temp_df.groupby("group", observed=False)
    group_stats: list[GroupMissingnessStat] = []

    for group_val, group_data in grouped:
        pop = len(group_data)
        if pop == 0:
            continue
        miss_count = int(group_data["is_missing"].sum())
        miss_pct = miss_count / pop * 100.0

        group_stats.append(
            GroupMissingnessStat(
                feature_analyzed=target_feature,
                grouping_variable=group_name,
                group_value=str(group_val),
                group_population=pop,
                missing_count=miss_count,
                missing_percentage=round(miss_pct, 2),
            )
        )

    if not group_stats:
        return GroupDisparitySummary(
            feature_analyzed=target_feature,
            grouping_variable=group_name,
            min_missing_group="None",
            min_missing_percentage=0.0,
            max_missing_group="None",
            max_missing_percentage=0.0,
            disparity_ratio=1.0,
            groups=[],
        )

    # Calculate min, max, and disparity ratio
    min_stat = min(group_stats, key=lambda s: s.missing_percentage)
    max_stat = max(group_stats, key=lambda s: s.missing_percentage)

    if min_stat.missing_percentage > 0:
        ratio = max_stat.missing_percentage / min_stat.missing_percentage
    else:
        # If min group has 0% missing, disparity ratio is 1.0 + max_pct difference
        ratio = 1.0 + (max_stat.missing_percentage / 100.0)

    return GroupDisparitySummary(
        feature_analyzed=target_feature,
        grouping_variable=group_name,
        min_missing_group=min_stat.group_value,
        min_missing_percentage=min_stat.missing_percentage,
        max_missing_group=max_stat.group_value,
        max_missing_percentage=max_stat.missing_percentage,
        disparity_ratio=round(ratio, 2),
        groups=group_stats,
    )
