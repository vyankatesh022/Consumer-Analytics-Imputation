"""Schema and structural validator for raw consumer data ingestion.

Enforces contract constraints, validates data types and ranges, checks required fields,
and routes malformed records to a quarantine structure while preserving legitimate missingness.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from missing_data_platform.exceptions import DataQualityError
from missing_data_platform.ingestion.contract import DataType, RawDataContract


class ExtraColumnsAction(StrEnum):
    """Policy for handling unexpected columns in raw data."""

    PRESERVE = "preserve"
    DROP = "drop"
    FAIL = "fail"


@dataclass
class ValidationResult:
    """Outcome of schema and structural validation."""

    is_valid: bool
    valid_df: pd.DataFrame
    quarantined_df: pd.DataFrame
    total_records: int
    valid_records_count: int
    quarantined_records_count: int
    missingness_summary: dict[str, int] = field(default_factory=dict)
    validation_errors: list[str] = field(default_factory=list)


class SchemaValidator:
    """Validates raw parsed dataframes against a RawDataContract."""

    def __init__(
        self,
        contract: RawDataContract | None = None,
        extra_columns_action: ExtraColumnsAction = ExtraColumnsAction.PRESERVE,
    ) -> None:
        self.contract = contract or RawDataContract.default_consumer_contract()
        self.extra_columns_action = extra_columns_action

    def validate(self, raw_df: pd.DataFrame) -> ValidationResult:
        """Execute complete schema and record validation on raw input DataFrame.

        Raises:
            DataQualityError: On fatal schema mismatch (e.g. missing required columns, empty dataframe).
        """
        if raw_df.empty:
            raise DataQualityError("Cannot validate an empty DataFrame")

        df = raw_df.copy()
        errors: list[str] = []

        # 1. Fatal Check: Expected column presence
        missing_columns = [col for col in self.contract.column_names if col not in df.columns]
        if missing_columns:
            msg = f"Fatal schema error: Missing expected columns in raw data: {missing_columns}"
            errors.append(msg)
            raise DataQualityError(msg, context={"missing_columns": missing_columns})

        # 2. Unexpected columns policy
        extra_columns = [col for col in df.columns if col not in self.contract.column_names]
        if extra_columns:
            if self.extra_columns_action == ExtraColumnsAction.FAIL:
                msg = f"Fatal schema error: Unexpected columns found: {extra_columns}"
                errors.append(msg)
                raise DataQualityError(msg, context={"extra_columns": extra_columns})
            elif self.extra_columns_action == ExtraColumnsAction.DROP:
                df = df.drop(columns=extra_columns)

        # 3. Initialize quarantine tracking mask and reason column
        quarantine_reasons: list[list[str]] = [[] for _ in range(len(df))]

        # 4. Check Non-Nullable Required Fields (e.g. customer_id, target)
        for req_col in self.contract.required_columns:
            null_mask = df[req_col].isna() | (df[req_col].astype(str).str.strip() == "")
            for idx in np.where(null_mask)[0]:
                quarantine_reasons[idx].append(f"Required column '{req_col}' is null or empty")

        # 5. Check Identifier Uniqueness (customer_id)
        id_col = self.contract.id_column
        if id_col in df.columns:
            duplicated_ids = df[id_col].dropna().duplicated(keep=False)
            for idx in np.where(duplicated_ids)[0]:
                quarantine_reasons[idx].append(
                    f"Duplicate customer identifier: '{df[id_col].iloc[idx]}'"
                )

        # 6. Type Coercion and Range Validation
        canonical_data: dict[str, Any] = {}
        for col_name, col_defn in self.contract.columns.items():
            if col_name not in df.columns:
                continue

            series = df[col_name]
            coerced_series: Any = series.copy()

            if col_defn.data_type in (DataType.FLOAT, DataType.INTEGER):
                # Convert numeric values safely; invalid strings become NaN and flagged
                numeric_series = pd.to_numeric(series, errors="coerce")
                # Identify non-null entries that failed conversion
                originally_not_null = series.notna() & (series.astype(str).str.strip() != "")
                conversion_failed = originally_not_null & numeric_series.isna()

                for idx in np.where(conversion_failed)[0]:
                    quarantine_reasons[idx].append(
                        f"Non-numeric value '{series.iloc[idx]}' in numeric column '{col_name}'"
                    )

                # Check Min / Max Range Violations on valid numeric values
                if col_defn.min_value is not None:
                    below_min = numeric_series < col_defn.min_value
                    for idx in np.where(below_min.fillna(False))[0]:
                        quarantine_reasons[idx].append(
                            f"Value {numeric_series.iloc[idx]} in '{col_name}' below minimum allowed ({col_defn.min_value})"
                        )

                if col_defn.max_value is not None:
                    above_max = numeric_series > col_defn.max_value
                    for idx in np.where(above_max.fillna(False))[0]:
                        quarantine_reasons[idx].append(
                            f"Value {numeric_series.iloc[idx]} in '{col_name}' above maximum allowed ({col_defn.max_value})"
                        )

                coerced_series = numeric_series

            elif col_defn.data_type == DataType.STRING:
                # Keep string representations, preserving genuine NaN
                coerced_series = series.where(series.notna(), None).astype(object)

            canonical_data[col_name] = coerced_series

        canonical_df = pd.DataFrame(canonical_data, index=df.index)

        # Retain any preserved extra columns
        if self.extra_columns_action == ExtraColumnsAction.PRESERVE:
            for extra_col in extra_columns:
                canonical_df[extra_col] = df[extra_col]

        # 7. Segregate Valid vs Quarantined records
        is_quarantined = np.array([len(reasons) > 0 for reasons in quarantine_reasons])
        quarantine_reason_strings = ["; ".join(reasons) for reasons in quarantine_reasons]

        valid_df = canonical_df[~is_quarantined].copy().reset_index(drop=True)
        quarantined_df = canonical_df[is_quarantined].copy()
        quarantined_df["_quarantine_reason"] = [
            r for r, is_q in zip(quarantine_reason_strings, is_quarantined, strict=True) if is_q
        ]
        quarantined_df = quarantined_df.reset_index(drop=True)

        # Ensure target column is integer in valid dataset
        target_col = self.contract.target_column
        if target_col in valid_df.columns and not valid_df.empty:
            valid_df[target_col] = valid_df[target_col].astype(int)

        # Calculate missingness distribution on valid records (strictly preserved)
        missingness = {col: int(valid_df[col].isna().sum()) for col in valid_df.columns}

        return ValidationResult(
            is_valid=(len(quarantined_df) == 0),
            valid_df=valid_df,
            quarantined_df=quarantined_df,
            total_records=len(df),
            valid_records_count=len(valid_df),
            quarantined_records_count=len(quarantined_df),
            missingness_summary=missingness,
            validation_errors=errors,
        )
