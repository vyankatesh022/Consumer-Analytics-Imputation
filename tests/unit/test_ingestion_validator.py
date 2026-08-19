"""Unit tests for SchemaValidator, contract enforcement, and quarantine routing."""

import numpy as np
import pandas as pd
import pytest

from missing_data_platform.exceptions import DataQualityError
from missing_data_platform.ingestion.validator import ExtraColumnsAction, SchemaValidator


@pytest.fixture
def base_valid_df() -> pd.DataFrame:
    """Fixture providing a clean dataframe meeting the default raw consumer contract."""
    data = {
        "customer_id": ["C001", "C002", "C003"],
        "age": [25.0, np.nan, 42.0],  # Legitimate null preserved
        "gender": ["Male", "Female", None],
        "income": [50000.0, 75000.0, np.nan],
        "education": ["Bachelor", "Master", "PhD"],
        "occupation": ["Engineer", "Teacher", "Doctor"],
        "city": ["New York", "Chicago", "Boston"],
        "region": ["East", "Midwest", "East"],
        "purchase_frequency": [3.0, 1.5, 5.0],
        "average_purchase_value": [50.0, 120.0, 80.0],
        "total_spend": [150.0, 180.0, 400.0],
        "discount_usage": [0.1, 0.25, np.nan],
        "website_visits": [10.0, 5.0, 20.0],
        "campaign_exposure": [2.0, 1.0, 4.0],
        "product_category": ["Electronics", "Apparel", "Home"],
        "customer_segment": ["Gold", "Silver", "Platinum"],
        "purchase_next_month": [1, 0, 1],
    }
    return pd.DataFrame(data)


def test_validate_clean_data_preserves_missingness(base_valid_df: pd.DataFrame) -> None:
    """Verify that validation passes on valid data and preserves legitimate NaN values."""
    validator = SchemaValidator()
    result = validator.validate(base_valid_df)

    assert result.is_valid is True
    assert result.valid_records_count == 3
    assert result.quarantined_records_count == 0
    assert len(result.quarantined_df) == 0

    # Assert missingness was NOT filled or altered
    assert pd.isna(result.valid_df.loc[1, "age"])
    assert pd.isna(result.valid_df.loc[2, "income"])
    assert pd.isna(result.valid_df.loc[2, "discount_usage"])
    assert result.missingness_summary["age"] == 1
    assert result.missingness_summary["income"] == 1


def test_validate_missing_required_column_raises_error(base_valid_df: pd.DataFrame) -> None:
    """Verify that missing a required contract column causes fatal DataQualityError."""
    validator = SchemaValidator()
    incomplete_df = base_valid_df.drop(columns=["customer_id"])

    with pytest.raises(DataQualityError) as exc_info:
        validator.validate(incomplete_df)
    assert "Missing expected columns" in str(exc_info.value)


def test_validate_quarantines_null_identifier(base_valid_df: pd.DataFrame) -> None:
    """Verify that rows with null customer_id are routed to quarantine with reason."""
    validator = SchemaValidator()
    df = base_valid_df.copy()
    df.loc[1, "customer_id"] = None  # Make ID null on row 1

    result = validator.validate(df)
    assert result.is_valid is False
    assert result.valid_records_count == 2
    assert result.quarantined_records_count == 1
    assert (
        "Required column 'customer_id' is null"
        in result.quarantined_df.loc[0, "_quarantine_reason"]
    )


def test_validate_quarantines_duplicate_identifiers(base_valid_df: pd.DataFrame) -> None:
    """Verify that duplicate customer IDs are flagged and quarantined."""
    validator = SchemaValidator()
    df = base_valid_df.copy()
    df.loc[1, "customer_id"] = "C001"  # Duplicate C001

    result = validator.validate(df)
    assert result.is_valid is False
    assert result.quarantined_records_count == 2  # Both duplicates quarantined
    assert "Duplicate customer identifier" in result.quarantined_df.loc[0, "_quarantine_reason"]


def test_validate_quarantines_invalid_numeric_conversion(base_valid_df: pd.DataFrame) -> None:
    """Verify that non-numeric strings in numeric columns are quarantined."""
    validator = SchemaValidator()
    df = base_valid_df.copy()
    df["age"] = df["age"].astype(object)
    df.loc[0, "age"] = "invalid_age_str"

    result = validator.validate(df)
    assert result.valid_records_count == 2
    assert result.quarantined_records_count == 1
    assert "Non-numeric value" in result.quarantined_df.loc[0, "_quarantine_reason"]


def test_validate_quarantines_range_violations(base_valid_df: pd.DataFrame) -> None:
    """Verify that values outside allowed min/max ranges are quarantined."""
    validator = SchemaValidator()
    df = base_valid_df.copy()
    df.loc[0, "discount_usage"] = 1.85  # Max is 1.0

    result = validator.validate(df)
    assert result.valid_records_count == 2
    assert result.quarantined_records_count == 1
    assert "above maximum allowed" in result.quarantined_df.loc[0, "_quarantine_reason"]


def test_unexpected_columns_policies(base_valid_df: pd.DataFrame) -> None:
    """Verify extra columns behavior under PRESERVE, DROP, and FAIL actions."""
    df = base_valid_df.copy()
    df["unexpected_metadata"] = "test_tag"

    # 1. Preserve
    val_preserve = SchemaValidator(extra_columns_action=ExtraColumnsAction.PRESERVE)
    res_preserve = val_preserve.validate(df)
    assert "unexpected_metadata" in res_preserve.valid_df.columns

    # 2. Drop
    val_drop = SchemaValidator(extra_columns_action=ExtraColumnsAction.DROP)
    res_drop = val_drop.validate(df)
    assert "unexpected_metadata" not in res_drop.valid_df.columns

    # 3. Fail
    val_fail = SchemaValidator(extra_columns_action=ExtraColumnsAction.FAIL)
    with pytest.raises(DataQualityError) as exc_info:
        val_fail.validate(df)
    assert "Unexpected columns found" in str(exc_info.value)
