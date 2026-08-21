"""Unit tests for MaskingEngine orchestrator and benchmark dataset creation."""

import pandas as pd
import pytest

from missing_data_platform.exceptions import ConfigurationError, DataQualityError
from missing_data_platform.masking.config import MaskingConfig, MaskingStrategy
from missing_data_platform.masking.engine import MaskingEngine


@pytest.fixture
def synthetic_benchmark_df() -> pd.DataFrame:
    """Fixture providing a clean reference dataframe for masking."""
    data = {
        "customer_id": [f"C{i:03d}" for i in range(100)],
        "age": [float(20 + (i % 50)) for i in range(100)],
        "gender": ["Male" if i % 2 == 0 else "Female" for i in range(100)],
        "income": [float(30000 + i * 500) for i in range(100)],
        "region": ["East", "West", "Midwest", "South"] * 25,
        "purchase_frequency": [float(1 + (i % 5)) for i in range(100)],
        "average_purchase_value": [float(20 + (i % 30)) for i in range(100)],
        "total_spend": [float(100 + i * 10) for i in range(100)],
        "discount_usage": [float((i % 10) / 10.0) for i in range(100)],
        "website_visits": [float(5 + (i % 15)) for i in range(100)],
        "campaign_exposure": [float(i % 4) for i in range(100)],
        "product_category": ["Electronics", "Apparel", "Home", "Beauty"] * 25,
        "customer_segment": ["Gold", "Silver", "Platinum", "Bronze"] * 25,
        "purchase_next_month": [i % 2 for i in range(100)],
    }
    return pd.DataFrame(data)


def test_masking_engine_uniform_random(synthetic_benchmark_df: pd.DataFrame) -> None:
    """Verify that MaskingEngine creates a valid benchmark dataset without altering protected features."""
    config = MaskingConfig(
        experiment_id="exp_bench_01",
        mask_rate=0.20,
        random_seed=42,
        strategy=MaskingStrategy.UNIFORM_RANDOM,
        target_features=["age", "income"],
    )
    engine = MaskingEngine()
    result = engine.generate_benchmark_dataset(synthetic_benchmark_df, config)

    assert result.experiment_id == "exp_bench_01"
    assert result.total_records == 100
    assert result.total_artificially_masked_cells == 40  # 20 from age, 20 from income

    # Protected columns must have ZERO missing values
    assert result.masked_dataset["customer_id"].isna().sum() == 0
    assert result.masked_dataset["purchase_next_month"].isna().sum() == 0

    # Masked dataset contains NaNs for the masked cells
    assert result.masked_dataset["age"].isna().sum() == 20
    assert result.masked_dataset["income"].isna().sum() == 20

    # Ground truth values are captured
    assert len(result.ground_truth_store.get_ground_truth("age")) == 20
    assert len(result.ground_truth_store.get_ground_truth("income")) == 20


def test_reference_dataset_is_not_mutated(synthetic_benchmark_df: pd.DataFrame) -> None:
    """Verify that the original reference dataset is not modified in-place during masking."""
    original_copy = synthetic_benchmark_df.copy(deep=True)
    config = MaskingConfig(
        experiment_id="exp_immutability",
        mask_rate=0.30,
        random_seed=42,
    )
    engine = MaskingEngine()
    _ = engine.generate_benchmark_dataset(synthetic_benchmark_df, config)

    # Reference DataFrame must remain identical
    pd.testing.assert_frame_equal(synthetic_benchmark_df, original_copy)


def test_masking_empty_df_raises_error() -> None:
    """Verify that masking an empty DataFrame raises DataQualityError."""
    engine = MaskingEngine()
    config = MaskingConfig(experiment_id="exp_empty")
    with pytest.raises(DataQualityError):
        engine.generate_benchmark_dataset(pd.DataFrame(), config)


def test_targeting_nonexistent_column_raises_error(synthetic_benchmark_df: pd.DataFrame) -> None:
    """Verify that targeting nonexistent columns raises ConfigurationError."""
    engine = MaskingEngine()
    config = MaskingConfig(
        experiment_id="exp_bad_col",
        target_features=["non_existent_column_xyz"],
    )
    with pytest.raises(ConfigurationError):
        engine.generate_benchmark_dataset(synthetic_benchmark_df, config)


def test_masking_engine_mar_covariate_strategy(synthetic_benchmark_df: pd.DataFrame) -> None:
    """Verify MaskingEngine execution with MAR_COVARIATE strategy."""
    config = MaskingConfig(
        experiment_id="exp_mar_engine",
        mask_rate=0.20,
        strategy=MaskingStrategy.MAR_COVARIATE,
        conditioning_covariate="age",
        target_features=["income"],
    )
    engine = MaskingEngine()
    result = engine.generate_benchmark_dataset(synthetic_benchmark_df, config)
    assert result.total_artificially_masked_cells == 20
    assert result.strategy == MaskingStrategy.MAR_COVARIATE


def test_masking_engine_group_stratified_strategy(synthetic_benchmark_df: pd.DataFrame) -> None:
    """Verify MaskingEngine execution with GROUP_STRATIFIED strategy."""
    config = MaskingConfig(
        experiment_id="exp_strat_engine",
        mask_rate=0.20,
        strategy=MaskingStrategy.GROUP_STRATIFIED,
        conditioning_covariate="region",
        target_features=["income"],
    )
    engine = MaskingEngine()
    result = engine.generate_benchmark_dataset(synthetic_benchmark_df, config)
    assert result.total_artificially_masked_cells == 20
    assert result.strategy == MaskingStrategy.GROUP_STRATIFIED
