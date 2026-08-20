"""Feature-level, row-level, and combinatorial pattern missingness profiling."""

import pandas as pd

from missing_data_platform.missingness.report import (
    FeatureMissingnessProfile,
    MissingnessPattern,
    RowMissingnessProfile,
)


def profile_feature_missingness(df: pd.DataFrame) -> list[FeatureMissingnessProfile]:
    """Calculate feature-level missingness counts and percentages, ranked by missingness rate."""
    total_records = len(df)
    profiles: list[FeatureMissingnessProfile] = []

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        obs_count = total_records - null_count
        null_pct = (null_count / total_records * 100.0) if total_records > 0 else 0.0

        profiles.append(
            FeatureMissingnessProfile(
                column_name=col,
                total_records=total_records,
                missing_count=null_count,
                observed_count=obs_count,
                missing_percentage=round(null_pct, 2),
                rank=0,  # Assigned after sorting
            )
        )

    # Sort descending by missing percentage
    profiles.sort(key=lambda p: (p.missing_percentage, p.missing_count), reverse=True)
    for rank_idx, prof in enumerate(profiles, start=1):
        prof.rank = rank_idx

    return profiles


def profile_row_missingness(df: pd.DataFrame) -> RowMissingnessProfile:
    """Analyze row-level missingness distributions across all observations."""
    total_rows = len(df)
    if total_rows == 0:
        return RowMissingnessProfile(
            total_rows=0,
            completely_observed_rows=0,
            completely_observed_percentage=0.0,
            rows_with_any_missing=0,
            rows_with_any_missing_percentage=0.0,
            max_missing_columns_in_a_row=0,
            missing_counts_distribution={},
        )

    # Count nulls per row
    nulls_per_row = df.isna().sum(axis=1)
    counts_dist = nulls_per_row.value_counts().to_dict()
    int_dist = {int(k): int(v) for k, v in sorted(counts_dist.items())}

    complete_rows = int_dist.get(0, 0)
    complete_pct = complete_rows / total_rows * 100.0
    any_missing_rows = total_rows - complete_rows
    any_missing_pct = any_missing_rows / total_rows * 100.0
    max_missing = int(nulls_per_row.max()) if not nulls_per_row.empty else 0

    return RowMissingnessProfile(
        total_rows=total_rows,
        completely_observed_rows=complete_rows,
        completely_observed_percentage=round(complete_pct, 2),
        rows_with_any_missing=any_missing_rows,
        rows_with_any_missing_percentage=round(any_missing_pct, 2),
        max_missing_columns_in_a_row=max_missing,
        missing_counts_distribution=int_dist,
    )


def profile_missingness_patterns(
    df: pd.DataFrame,
    max_patterns: int = 20,
) -> list[MissingnessPattern]:
    """Identify combinatorial patterns of missing features and calculate their observed frequencies."""
    total_rows = len(df)
    if total_rows == 0:
        return []

    # Map each row to tuple of missing column names
    is_null_df = df.isna()
    columns = list(df.columns)

    # Fast pattern extraction via string representation of boolean indicator masks
    patterns_series = is_null_df.apply(
        lambda row: tuple(col for col, is_na in zip(columns, row, strict=True) if is_na),
        axis=1,
    )

    pattern_counts = patterns_series.value_counts().head(max_patterns)
    patterns: list[MissingnessPattern] = []

    for pattern_id, (cols_tuple, count) in enumerate(pattern_counts.items(), start=1):
        pct = count / total_rows * 100.0
        patterns.append(
            MissingnessPattern(
                pattern_id=pattern_id,
                missing_columns=list(cols_tuple),
                row_count=int(count),
                percentage=round(pct, 2),
            )
        )

    return patterns
