"""Unit tests for MissingnessAnalysisEngine orchestrator."""

import numpy as np
import pandas as pd
import pytest

from missing_data_platform.exceptions import DataQualityError
from missing_data_platform.missingness.engine import MissingnessAnalysisEngine


@pytest.fixture
def consumer_missingness_df() -> pd.DataFrame:
    """Fixture with missingness across demographics and continuous features."""
    data = {
        "customer_id": ["C1", "C2", "C3", "C4", "C5", "C6"],
        "age": [20.0, 25.0, 30.0, np.nan, 45.0, 50.0],
        "gender": ["Male", "Female", "Female", "Male", "Male", "Female"],
        "income": [40000.0, 50000.0, np.nan, np.nan, 80000.0, 90000.0],
        "region": ["West", "West", "East", "East", "South", "South"],
        "purchase_frequency": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "average_purchase_value": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "total_spend": [10.0, 40.0, 90.0, 160.0, 250.0, 360.0],
        "discount_usage": [0.1, 0.2, 0.3, 0.4, 0.5, 0.0],
        "website_visits": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "campaign_exposure": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        "product_category": ["A", "B", "C", "D", "E", "F"],
        "customer_segment": ["Gold", "Silver", "Bronze", "Gold", "Silver", "Bronze"],
        "purchase_next_month": [1, 0, 1, 0, 1, 0],
    }
    return pd.DataFrame(data)


def test_missingness_engine_orchestration(consumer_missingness_df: pd.DataFrame) -> None:
    """Verify that MissingnessAnalysisEngine executes all profilers and diagnostics."""
    engine = MissingnessAnalysisEngine()
    report = engine.analyze(consumer_missingness_df, dataset_id="engine_test_run")

    assert report.dataset_id == "engine_test_run"
    assert report.total_records == 6
    assert report.features_with_missingness_count == 2  # age and income
    assert len(report.feature_profiles) == 14
    assert report.row_profile is not None
    assert len(report.top_patterns) > 0
    assert report.mcar_diagnostics is not None
    assert report.mar_diagnostics is not None
    assert len(report.mnar_limitation_statement) > 0

    # JSON export
    json_str = report.to_json()
    assert "engine_test_run" in json_str
    assert "feature_profiles" in json_str


def test_missingness_engine_empty_df_raises_error() -> None:
    """Verify that analyzing an empty DataFrame raises DataQualityError."""
    engine = MissingnessAnalysisEngine()
    with pytest.raises(DataQualityError) as exc_info:
        engine.analyze(pd.DataFrame())
    assert "empty" in str(exc_info.value)


def test_missingness_analysis_preserves_dataset_immutability(
    consumer_missingness_df: pd.DataFrame,
) -> None:
    """Verify that missingness analysis never alters or imputes the source dataset."""
    copy_df = consumer_missingness_df.copy(deep=True)
    engine = MissingnessAnalysisEngine()

    _ = engine.analyze(consumer_missingness_df)

    pd.testing.assert_frame_equal(consumer_missingness_df, copy_df)
