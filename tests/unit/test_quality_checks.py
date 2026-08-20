"""Unit tests for modular data quality checks."""

import numpy as np
import pandas as pd
import pytest

from missing_data_platform.ingestion.contract import (
    ColumnDefinition,
    DataType,
    RawDataContract,
)
from missing_data_platform.quality.checks import (
    check_categorical_validity,
    check_duplicates,
    check_numerical_boundaries,
    check_schema_conformance,
    check_target_integrity,
    compute_distribution_summaries,
    measure_missingness,
)
from missing_data_platform.quality.rules import QualityStatus


@pytest.fixture
def clean_consumer_df() -> pd.DataFrame:
    """Fixture providing a valid dataframe for quality checks."""
    data = {
        "customer_id": ["C101", "C102", "C103", "C104"],
        "age": [25.0, 34.0, 45.0, 52.0],
        "gender": ["Male", "Female", "Female", "Male"],
        "income": [50000.0, 75000.0, 95000.0, 62000.0],
        "education": ["Bachelor", "Master", "PhD", "Bachelor"],
        "occupation": ["Engineer", "Teacher", "Doctor", "Analyst"],
        "city": ["New York", "Chicago", "Boston", "Seattle"],
        "region": ["East", "Midwest", "East", "West"],
        "purchase_frequency": [3.0, 1.5, 5.0, 2.0],
        "average_purchase_value": [50.0, 120.0, 80.0, 45.0],
        "total_spend": [150.0, 180.0, 400.0, 90.0],
        "discount_usage": [0.1, 0.25, 0.0, 0.5],
        "website_visits": [10.0, 5.0, 20.0, 12.0],
        "campaign_exposure": [2.0, 1.0, 4.0, 3.0],
        "product_category": ["Electronics", "Apparel", "Home", "Beauty"],
        "customer_segment": ["Gold", "Silver", "Platinum", "Bronze"],
        "purchase_next_month": [1, 0, 1, 0],
    }
    return pd.DataFrame(data)


def test_check_schema_conformance_pass(clean_consumer_df: pd.DataFrame) -> None:
    """Verify that schema conformance passes on complete contract."""
    contract = RawDataContract.default_consumer_contract()
    result = check_schema_conformance(clean_consumer_df, contract)
    assert result.status == QualityStatus.PASS


def test_check_schema_conformance_missing_column(clean_consumer_df: pd.DataFrame) -> None:
    """Verify schema conformance fails when a contract column is missing."""
    contract = RawDataContract.default_consumer_contract()
    incomplete_df = clean_consumer_df.drop(columns=["income"])
    result = check_schema_conformance(incomplete_df, contract)
    assert result.status == QualityStatus.FAIL
    assert "income" in str(result.details)


def test_measure_missingness_exact_counts(clean_consumer_df: pd.DataFrame) -> None:
    """Verify exact calculation of null counts and percentages."""
    contract = RawDataContract.default_consumer_contract()
    df = clean_consumer_df.copy()
    df.loc[0, "income"] = np.nan
    df.loc[1, "income"] = np.nan  # 2 out of 4 = 50%

    check_res, metrics = measure_missingness(df, contract, warning_threshold_pct=30.0)
    assert metrics["income"].missing_count == 2
    assert metrics["income"].missing_percentage == 50.0
    assert metrics["age"].missing_count == 0
    assert check_res.status == QualityStatus.WARN  # 50% is >= 30% warning threshold


def test_check_duplicates_detection(clean_consumer_df: pd.DataFrame) -> None:
    """Verify detection of duplicate rows and identifier collisions."""
    df = clean_consumer_df.copy()
    # Add an identical row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    check_res, metric = check_duplicates(df, id_column="customer_id")
    assert check_res.status == QualityStatus.FAIL  # Duplicate ID is fatal
    assert metric.full_row_duplicates == 1
    assert metric.identifier_duplicates == 1


def test_check_numerical_boundaries_violations(clean_consumer_df: pd.DataFrame) -> None:
    """Verify boundary checks on impossible negative values."""
    contract = RawDataContract.default_consumer_contract()
    df = clean_consumer_df.copy()
    df.loc[0, "age"] = -5.0  # Impossible age
    df.loc[1, "discount_usage"] = 2.5  # Max is 1.0

    result = check_numerical_boundaries(df, contract)
    assert result.status == QualityStatus.FAIL
    assert "age" in result.message
    assert "discount_usage" in result.message


def test_check_categorical_validity(clean_consumer_df: pd.DataFrame) -> None:
    """Verify categorical vocabulary validation."""
    contract = RawDataContract.default_consumer_contract()
    # Replace gender column definition with allowed categories
    contract.columns["gender"] = ColumnDefinition(
        name="gender",
        data_type=DataType.STRING,
        nullable=True,
        allowed_categories=["Male", "Female", "Non-Binary"],
    )

    res_valid = check_categorical_validity(clean_consumer_df, contract)
    assert res_valid.status == QualityStatus.PASS

    # Add unauthorized category
    df_invalid = clean_consumer_df.copy()
    df_invalid.loc[0, "gender"] = "UnauthorizedCategory123"
    res_invalid = check_categorical_validity(df_invalid, contract)
    assert res_invalid.status == QualityStatus.FAIL
    assert "UnauthorizedCategory123" in str(res_invalid.details)


def test_check_target_integrity(clean_consumer_df: pd.DataFrame) -> None:
    """Verify target validation passes on valid {0, 1} and fails on invalid classes."""
    # Valid
    res_valid = check_target_integrity(clean_consumer_df, "purchase_next_month")
    assert res_valid.status == QualityStatus.PASS

    # Invalid: Contains class 2
    df_invalid = clean_consumer_df.copy()
    df_invalid.loc[0, "purchase_next_month"] = 2
    res_invalid = check_target_integrity(df_invalid, "purchase_next_month")
    assert res_invalid.status == QualityStatus.FAIL


def test_compute_distribution_summaries(clean_consumer_df: pd.DataFrame) -> None:
    """Verify computation of statistical summaries."""
    summaries = compute_distribution_summaries(clean_consumer_df)
    assert "age" in summaries
    assert summaries["age"].min_value == 25.0
    assert summaries["age"].max_value == 52.0
    assert summaries["age"].count == 4
    assert summaries["age"].missing_count == 0

    assert "gender" in summaries
    assert summaries["gender"].distinct_count == 2
    assert summaries["gender"].top_categories is not None
    assert summaries["gender"].top_categories["Male"] == 2
