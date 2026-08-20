"""Unit tests for QualityReport model and DataQualityEngine orchestrator."""

import numpy as np
import pandas as pd
import pytest

from missing_data_platform.exceptions import DataQualityError
from missing_data_platform.quality.engine import DataQualityEngine
from missing_data_platform.quality.rules import QualityStatus


@pytest.fixture
def sample_quality_df() -> pd.DataFrame:
    """Fixture providing a dataset with known missingness for quality engine test."""
    data = {
        "customer_id": ["C1", "C2", "C3"],
        "age": [20.0, np.nan, 40.0],
        "gender": ["F", "M", "F"],
        "income": [40000.0, 50000.0, 60000.0],
        "education": ["B", "M", "B"],
        "occupation": ["E", "T", "D"],
        "city": ["NYC", "CHI", "LA"],
        "region": ["East", "Midwest", "West"],
        "purchase_frequency": [1.0, 2.0, 3.0],
        "average_purchase_value": [10.0, 20.0, 30.0],
        "total_spend": [10.0, 40.0, 90.0],
        "discount_usage": [0.1, 0.2, 0.3],
        "website_visits": [1.0, 2.0, 3.0],
        "campaign_exposure": [1.0, 1.0, 2.0],
        "product_category": ["A", "B", "C"],
        "customer_segment": ["Gold", "Silver", "Bronze"],
        "purchase_next_month": [1, 0, 1],
    }
    return pd.DataFrame(data)


def test_data_quality_engine_audit(sample_quality_df: pd.DataFrame) -> None:
    """Verify that DataQualityEngine generates a complete, valid QualityReport."""
    engine = DataQualityEngine()
    report = engine.audit_dataset(sample_quality_df, dataset_id="unit_test_audit")

    assert report.dataset_id == "unit_test_audit"
    assert report.total_records == 3
    assert report.total_columns == 17
    assert report.overall_status in (QualityStatus.PASS, QualityStatus.WARN)
    assert report.passed_checks > 0
    assert len(report.checks) == 6
    assert "age" in report.missingness_summary
    assert report.missingness_summary["age"].missing_count == 1

    # Verify JSON serialization
    json_str = report.to_json()
    assert "unit_test_audit" in json_str
    assert "missingness_summary" in json_str


def test_empty_dataframe_audit_raises_error() -> None:
    """Verify that auditing an empty DataFrame raises DataQualityError."""
    engine = DataQualityEngine()
    with pytest.raises(DataQualityError) as exc_info:
        engine.audit_dataset(pd.DataFrame(), dataset_id="empty_df")
    assert "empty" in str(exc_info.value)


def test_input_dataframe_immutability(sample_quality_df: pd.DataFrame) -> None:
    """Verify that data quality audit strictly preserves the original DataFrame without mutations."""
    original_copy = sample_quality_df.copy(deep=True)
    engine = DataQualityEngine()

    _ = engine.audit_dataset(sample_quality_df, dataset_id="immutability_test")

    # Assert exact equality with original before audit
    pd.testing.assert_frame_equal(sample_quality_df, original_copy)
