"""Modular, non-mutating data quality validation checks.

Implements pure inspection functions for schema conformance, missingness rates,
duplicate occurrences, numerical boundary checks, categorical validity, target integrity,
and statistical distribution summaries.
"""

import pandas as pd

from missing_data_platform.ingestion.contract import DataType, RawDataContract
from missing_data_platform.quality.report import (
    CheckDetail,
    DistributionSummary,
    DuplicateMetric,
    MissingnessMetric,
)
from missing_data_platform.quality.rules import QualitySeverity, QualityStatus


def check_schema_conformance(
    df: pd.DataFrame,
    contract: RawDataContract,
    strict: bool = False,
) -> CheckDetail:
    """Verify that DataFrame columns match the RawDataContract schema."""
    missing_cols = [col for col in contract.column_names if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in contract.column_names]

    if missing_cols:
        return CheckDetail(
            check_name="schema_conformance",
            status=QualityStatus.FAIL,
            severity=QualitySeverity.ERROR,
            message=f"Missing required contract columns: {missing_cols}",
            details={"missing_columns": missing_cols, "extra_columns": extra_cols},
        )

    if extra_cols and strict:
        return CheckDetail(
            check_name="schema_conformance",
            status=QualityStatus.WARN,
            severity=QualitySeverity.WARNING,
            message=f"Dataset contains unexpected extra columns: {extra_cols}",
            details={"missing_columns": [], "extra_columns": extra_cols},
        )

    return CheckDetail(
        check_name="schema_conformance",
        status=QualityStatus.PASS,
        severity=QualitySeverity.INFO,
        message="All expected contract columns are present.",
        details={"total_columns": len(df.columns), "extra_columns": extra_cols},
    )


def measure_missingness(
    df: pd.DataFrame,
    contract: RawDataContract,
    warning_threshold_pct: float = 30.0,
    error_threshold_pct: float = 80.0,
) -> tuple[CheckDetail, dict[str, MissingnessMetric]]:
    """Measure missingness counts and percentages across all columns without imputation."""
    total_records = len(df)
    metrics: dict[str, MissingnessMetric] = {}
    severely_missing_cols: list[str] = []
    moderately_missing_cols: list[str] = []
    required_missing_violations: list[str] = []

    for col in df.columns:
        col_defn = contract.get_column(col)
        is_nullable = col_defn.nullable if col_defn else True
        null_count = int(df[col].isna().sum())
        null_pct = (null_count / total_records * 100.0) if total_records > 0 else 0.0

        metrics[col] = MissingnessMetric(
            column_name=col,
            total_records=total_records,
            missing_count=null_count,
            missing_percentage=round(null_pct, 2),
            is_nullable=is_nullable,
        )

        if not is_nullable and null_count > 0:
            required_missing_violations.append(f"{col} ({null_count} nulls)")
        elif null_pct >= error_threshold_pct:
            severely_missing_cols.append(f"{col} ({null_pct:.1f}%)")
        elif null_pct >= warning_threshold_pct:
            moderately_missing_cols.append(f"{col} ({null_pct:.1f}%)")

    if required_missing_violations:
        return (
            CheckDetail(
                check_name="missingness_evaluation",
                status=QualityStatus.FAIL,
                severity=QualitySeverity.ERROR,
                message=f"Non-nullable columns contain missing values: {required_missing_violations}",
                details={
                    "required_missing": required_missing_violations,
                    "severe_missing": severely_missing_cols,
                    "moderate_missing": moderately_missing_cols,
                },
            ),
            metrics,
        )

    if severely_missing_cols:
        return (
            CheckDetail(
                check_name="missingness_evaluation",
                status=QualityStatus.WARN,
                severity=QualitySeverity.WARNING,
                message=f"Columns with severe missingness (>{error_threshold_pct}%): {severely_missing_cols}",
                details={
                    "severe_missing": severely_missing_cols,
                    "moderate_missing": moderately_missing_cols,
                },
            ),
            metrics,
        )

    if moderately_missing_cols:
        return (
            CheckDetail(
                check_name="missingness_evaluation",
                status=QualityStatus.WARN,
                severity=QualitySeverity.WARNING,
                message=f"Columns with moderate missingness (>{warning_threshold_pct}%): {moderately_missing_cols}",
                details={"moderate_missing": moderately_missing_cols},
            ),
            metrics,
        )

    return (
        CheckDetail(
            check_name="missingness_evaluation",
            status=QualityStatus.PASS,
            severity=QualitySeverity.INFO,
            message="Missingness levels across all columns are within acceptable bounds.",
            details={"total_columns_evaluated": len(df.columns)},
        ),
        metrics,
    )


def check_duplicates(
    df: pd.DataFrame,
    id_column: str = "customer_id",
) -> tuple[CheckDetail, DuplicateMetric]:
    """Detect full row duplicates and identifier uniqueness violations."""
    total_records = len(df)
    full_dup_count = int(df.duplicated().sum())
    full_dup_pct = (full_dup_count / total_records * 100.0) if total_records > 0 else 0.0

    id_dup_count = 0
    if id_column in df.columns:
        id_dup_count = int(df[id_column].dropna().duplicated().sum())
    id_dup_pct = (id_dup_count / total_records * 100.0) if total_records > 0 else 0.0

    metric = DuplicateMetric(
        total_records=total_records,
        full_row_duplicates=full_dup_count,
        identifier_duplicates=id_dup_count,
        duplicate_row_percentage=round(full_dup_pct, 2),
        duplicate_id_percentage=round(id_dup_pct, 2),
    )

    if id_dup_count > 0:
        return (
            CheckDetail(
                check_name="duplicate_detection",
                status=QualityStatus.FAIL,
                severity=QualitySeverity.ERROR,
                message=f"Found {id_dup_count} duplicate '{id_column}' identifiers.",
                details={
                    "full_row_duplicates": full_dup_count,
                    "identifier_duplicates": id_dup_count,
                },
            ),
            metric,
        )

    if full_dup_count > 0:
        return (
            CheckDetail(
                check_name="duplicate_detection",
                status=QualityStatus.WARN,
                severity=QualitySeverity.WARNING,
                message=f"Found {full_dup_count} full duplicate rows in dataset.",
                details={"full_row_duplicates": full_dup_count, "identifier_duplicates": 0},
            ),
            metric,
        )

    return (
        CheckDetail(
            check_name="duplicate_detection",
            status=QualityStatus.PASS,
            severity=QualitySeverity.INFO,
            message="No duplicate rows or identifier collisions detected.",
            details={"total_records": total_records},
        ),
        metric,
    )


def check_numerical_boundaries(
    df: pd.DataFrame,
    contract: RawDataContract,
) -> CheckDetail:
    """Inspect numerical columns for impossible values and range violations."""
    violations: list[str] = []

    for col_name, col_defn in contract.columns.items():
        if col_name not in df.columns:
            continue
        if col_defn.data_type not in (DataType.FLOAT, DataType.INTEGER):
            continue

        numeric_series = pd.to_numeric(df[col_name], errors="coerce").dropna()
        if numeric_series.empty:
            continue

        if col_defn.min_value is not None:
            below_count = int((numeric_series < col_defn.min_value).sum())
            if below_count > 0:
                violations.append(
                    f"'{col_name}': {below_count} values < min ({col_defn.min_value})"
                )

        if col_defn.max_value is not None:
            above_count = int((numeric_series > col_defn.max_value).sum())
            if above_count > 0:
                violations.append(
                    f"'{col_name}': {above_count} values > max ({col_defn.max_value})"
                )

    if violations:
        return CheckDetail(
            check_name="numerical_boundaries",
            status=QualityStatus.FAIL,
            severity=QualitySeverity.ERROR,
            message=f"Numerical boundary violations detected: {violations}",
            details={"violations": violations},
        )

    return CheckDetail(
        check_name="numerical_boundaries",
        status=QualityStatus.PASS,
        severity=QualitySeverity.INFO,
        message="All numerical values are within expected mathematical bounds.",
        details={},
    )


def check_categorical_validity(
    df: pd.DataFrame,
    contract: RawDataContract,
) -> CheckDetail:
    """Validate categorical columns against declared controlled vocabularies."""
    violations: list[str] = []
    observed_vocabularies: dict[str, int] = {}

    for col_name, col_defn in contract.columns.items():
        if col_name not in df.columns or col_defn.data_type != DataType.STRING:
            continue

        series = df[col_name].dropna().astype(str).str.strip()
        observed_vocabularies[col_name] = int(series.nunique())

        if col_defn.allowed_categories is not None:
            invalid_cats = set(series.unique()) - set(col_defn.allowed_categories)
            if invalid_cats:
                violations.append(
                    f"'{col_name}' contains unauthorized categories: {list(invalid_cats)}"
                )

    if violations:
        return CheckDetail(
            check_name="categorical_validity",
            status=QualityStatus.FAIL,
            severity=QualitySeverity.ERROR,
            message=f"Categorical vocabulary violations detected: {violations}",
            details={"violations": violations, "cardinalities": observed_vocabularies},
        )

    return CheckDetail(
        check_name="categorical_validity",
        status=QualityStatus.PASS,
        severity=QualitySeverity.INFO,
        message="Categorical features conform to domain vocabularies.",
        details={"cardinalities": observed_vocabularies},
    )


def check_target_integrity(
    df: pd.DataFrame,
    target_column: str = "purchase_next_month",
) -> CheckDetail:
    """Verify target binary classification column integrity."""
    if target_column not in df.columns:
        return CheckDetail(
            check_name="target_integrity",
            status=QualityStatus.FAIL,
            severity=QualitySeverity.ERROR,
            message=f"Target column '{target_column}' is missing from dataset.",
            details={"target_column": target_column},
        )

    target_series = df[target_column]
    null_count = int(target_series.isna().sum())

    if null_count > 0:
        return CheckDetail(
            check_name="target_integrity",
            status=QualityStatus.FAIL,
            severity=QualitySeverity.ERROR,
            message=f"Target column '{target_column}' contains {null_count} missing values.",
            details={"null_count": null_count},
        )

    # Check that non-null values are exclusively {0, 1}
    unique_vals = set(pd.to_numeric(target_series, errors="coerce").dropna().unique())
    if not unique_vals.issubset({0, 1}):
        return CheckDetail(
            check_name="target_integrity",
            status=QualityStatus.FAIL,
            severity=QualitySeverity.ERROR,
            message=f"Target column '{target_column}' contains invalid non-binary values: {unique_vals}",
            details={"unique_values": [float(v) for v in unique_vals]},
        )

    # Compute class distribution
    class_counts = target_series.value_counts().to_dict()
    return CheckDetail(
        check_name="target_integrity",
        status=QualityStatus.PASS,
        severity=QualitySeverity.INFO,
        message=f"Target column '{target_column}' is valid binary indicator.",
        details={"class_distribution": {str(k): int(v) for k, v in class_counts.items()}},
    )


def compute_distribution_summaries(
    df: pd.DataFrame,
    contract: RawDataContract | None = None,
) -> dict[str, DistributionSummary]:
    """Compute lightweight statistical summaries for inspection without mutating data."""
    summaries: dict[str, DistributionSummary] = {}
    total_records = len(df)

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        non_null_count = total_records - missing_count

        # Numeric summary
        if pd.api.types.is_numeric_dtype(series):
            clean_num = pd.to_numeric(series, errors="coerce").dropna()
            min_v = float(clean_num.min()) if not clean_num.empty else None
            max_v = float(clean_num.max()) if not clean_num.empty else None
            mean_v = float(clean_num.mean()) if not clean_num.empty else None
            std_v = float(clean_num.std()) if len(clean_num) > 1 else None
            med_v = float(clean_num.median()) if not clean_num.empty else None

            summaries[col] = DistributionSummary(
                column_name=col,
                count=non_null_count,
                missing_count=missing_count,
                distinct_count=int(clean_num.nunique()),
                min_value=round(min_v, 2) if min_v is not None else None,
                max_value=round(max_v, 2) if max_v is not None else None,
                mean_value=round(mean_v, 2) if mean_v is not None else None,
                std_value=round(std_v, 2) if std_v is not None else None,
                median_value=round(med_v, 2) if med_v is not None else None,
            )
        else:
            # Categorical top categories
            clean_str = series.dropna().astype(str)
            top_cats = clean_str.value_counts().head(5).to_dict()
            summaries[col] = DistributionSummary(
                column_name=col,
                count=non_null_count,
                missing_count=missing_count,
                distinct_count=int(clean_str.nunique()),
                top_categories={str(k): int(v) for k, v in top_cats.items()},
            )

    return summaries
