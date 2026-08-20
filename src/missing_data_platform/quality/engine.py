"""Data Quality Engine Orchestrator.

Executes all modular checks in sequence, gathers telemetry, and produces a
comprehensive QualityReport without modifying the source dataset.
"""

import pandas as pd

from missing_data_platform.exceptions import DataQualityError
from missing_data_platform.ingestion.contract import RawDataContract
from missing_data_platform.logging import get_logger
from missing_data_platform.quality.checks import (
    check_categorical_validity,
    check_duplicates,
    check_numerical_boundaries,
    check_schema_conformance,
    check_target_integrity,
    compute_distribution_summaries,
    measure_missingness,
)
from missing_data_platform.quality.report import CheckDetail, QualityReport
from missing_data_platform.quality.rules import DataQualityConfig, QualityStatus

logger = get_logger("quality.engine")


class DataQualityEngine:
    """Orchestrator for executing non-mutating data quality audits."""

    def __init__(
        self,
        contract: RawDataContract | None = None,
        config: DataQualityConfig | None = None,
    ) -> None:
        self.contract = contract or RawDataContract.default_consumer_contract()
        self.config = config or DataQualityConfig()

    def audit_dataset(
        self,
        df: pd.DataFrame,
        dataset_id: str = "dataset_audit",
    ) -> QualityReport:
        """Execute all data quality checks on DataFrame and return structured report.

        Raises:
            DataQualityError: If input dataset is empty.
        """
        if df.empty:
            raise DataQualityError(
                "Cannot perform data quality audit on an empty DataFrame.",
                context={"dataset_id": dataset_id},
            )

        logger.info(
            "Starting data quality audit",
            dataset_id=dataset_id,
            total_records=len(df),
            total_columns=len(df.columns),
        )

        checks: list[CheckDetail] = []

        # 1. Schema Conformance
        schema_chk = check_schema_conformance(
            df,
            contract=self.contract,
            strict=self.config.strict_schema_matching,
        )
        checks.append(schema_chk)

        # 2. Missingness Measurement (pure measurement, zero imputation)
        miss_chk, missingness_summary = measure_missingness(
            df,
            contract=self.contract,
            warning_threshold_pct=self.config.max_missing_percentage_warning,
            error_threshold_pct=self.config.max_missing_percentage_error,
        )
        checks.append(miss_chk)

        # 3. Duplicate Detection
        dup_chk, dup_metric = check_duplicates(
            df,
            id_column=self.contract.id_column,
        )
        checks.append(dup_chk)

        # 4. Numerical Boundaries
        num_chk = check_numerical_boundaries(df, contract=self.contract)
        checks.append(num_chk)

        # 5. Categorical Vocabularies
        cat_chk = check_categorical_validity(df, contract=self.contract)
        checks.append(cat_chk)

        # 6. Target Variable Integrity
        target_chk = check_target_integrity(
            df,
            target_column=self.contract.target_column,
        )
        checks.append(target_chk)

        # 7. Distribution Summaries
        dist_summaries = compute_distribution_summaries(df, contract=self.contract)

        # Compute aggregate counts and overall status
        passed_count = sum(1 for c in checks if c.status == QualityStatus.PASS)
        warn_count = sum(1 for c in checks if c.status == QualityStatus.WARN)
        fail_count = sum(1 for c in checks if c.status == QualityStatus.FAIL)

        if fail_count > 0:
            overall_status = QualityStatus.FAIL
        elif warn_count > 0:
            overall_status = QualityStatus.WARN
        else:
            overall_status = QualityStatus.PASS

        report = QualityReport(
            dataset_id=dataset_id,
            total_records=len(df),
            total_columns=len(df.columns),
            overall_status=overall_status,
            passed_checks=passed_count,
            warning_checks=warn_count,
            failed_checks=fail_count,
            checks=checks,
            missingness_summary=missingness_summary,
            duplicate_summary=dup_metric,
            distribution_summaries=dist_summaries,
        )

        logger.info(
            "Data quality audit completed",
            dataset_id=dataset_id,
            overall_status=overall_status.value,
            passed=passed_count,
            warnings=warn_count,
            failures=fail_count,
        )

        return report
