"""Unit tests for RandomForestImputerModel and RandomForestImputationConfig."""

import numpy as np
import pandas as pd
import pytest

from missing_data_platform.exceptions import ConfigurationError, DataQualityError, ImputationError
from missing_data_platform.imputation.engine import BaselineImputationEngine
from missing_data_platform.imputation.rf import (
    RandomForestImputationConfig,
    RandomForestImputerModel,
)


@pytest.fixture
def nonlinear_numeric_df() -> pd.DataFrame:
    """Fixture with nonlinear relationship (y = 0.5 * x^2 + 2) and missing values."""
    np.random.seed(42)
    x = [float(i) for i in range(1, 26)]
    y = [float(0.5 * (xi**2) + 2.0) for xi in x]
    z = [float(xi * 3.0 + 1.0) for xi in x]

    # Insert missing values
    y[4] = np.nan  # x=5, expected y=14.5
    y[10] = np.nan  # x=11, expected y=62.5
    x[8] = np.nan  # row 8 has missing x

    return pd.DataFrame(
        {
            "customer_id": [f"C{i:02d}" for i in range(25)],
            "feat_x": x,
            "feat_y": y,
            "feat_z": z,
            "purchase_next_month": [i % 2 for i in range(25)],
        }
    )


def test_rf_config_validation() -> None:
    """Verify RandomForestImputationConfig validation invariants and resource bounds."""
    config = RandomForestImputationConfig(
        n_estimators=50,
        max_depth=10,
        min_samples_leaf=2,
        max_features="sqrt",
        random_seed=123,
        n_jobs=2,
    )
    assert config.n_estimators == 50
    assert config.max_depth == 10
    assert config.min_samples_leaf == 2
    assert config.max_features == "sqrt"
    assert config.random_seed == 123
    assert config.n_jobs == 2

    # Bounds validation
    with pytest.raises(ConfigurationError):
        RandomForestImputationConfig(n_estimators=0)

    with pytest.raises(ConfigurationError):
        RandomForestImputationConfig(n_estimators=600)

    with pytest.raises(ConfigurationError):
        RandomForestImputationConfig(max_depth=0)

    with pytest.raises(ConfigurationError):
        RandomForestImputationConfig(max_depth=100)

    with pytest.raises(ConfigurationError):
        RandomForestImputationConfig(min_samples_leaf=0)

    with pytest.raises(ConfigurationError):
        RandomForestImputationConfig(n_jobs=0)

    with pytest.raises(ConfigurationError):
        RandomForestImputationConfig(n_jobs=-2)

    with pytest.raises(ConfigurationError):
        RandomForestImputationConfig(target_features=["feat_x", "purchase_next_month"])


def test_rf_imputer_nonlinear_reconstruction(nonlinear_numeric_df: pd.DataFrame) -> None:
    """Verify Random Forest models reconstruct nonlinear relationships accurately."""
    config = RandomForestImputationConfig(
        n_estimators=100,
        max_depth=10,
        random_seed=42,
        target_features=["feat_x", "feat_y", "feat_z"],
    )
    imputer = RandomForestImputerModel(config=config)
    imputed_df, metrics = imputer.fit(nonlinear_numeric_df).transform_with_metrics(
        nonlinear_numeric_df
    )

    # Missing values should be completely imputed
    assert imputed_df["feat_x"].isna().sum() == 0
    assert imputed_df["feat_y"].isna().sum() == 0
    assert imputed_df["feat_z"].isna().sum() == 0

    # The reconstructed y[4] (where x=5, z=16, expected y=14.5) should be close to 14.5
    imputed_y4 = imputed_df.loc[4, "feat_y"]
    assert 10.0 <= imputed_y4 <= 20.0

    # Protected columns untouched
    assert imputed_df["customer_id"].tolist() == nonlinear_numeric_df["customer_id"].tolist()
    assert (
        imputed_df["purchase_next_month"].tolist()
        == nonlinear_numeric_df["purchase_next_month"].tolist()
    )
    assert len(metrics) == 3


def test_rf_imputer_unfitted_transform_raises_error() -> None:
    """Verify calling transform on unfitted RandomForestImputerModel raises ImputationError."""
    imputer = RandomForestImputerModel()
    with pytest.raises(ImputationError):
        imputer.transform(pd.DataFrame({"feat_x": [1.0, np.nan]}))


def test_rf_imputer_empty_df_raises_error() -> None:
    """Verify fitting on empty DataFrame raises DataQualityError."""
    imputer = RandomForestImputerModel()
    with pytest.raises(DataQualityError):
        imputer.fit(pd.DataFrame())


def test_rf_imputer_all_target_missing_raises_error() -> None:
    """Verify fitting when a feature has 0 observed values raises ImputationError."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [np.nan, np.nan, np.nan],
        }
    )
    imputer = RandomForestImputerModel()
    with pytest.raises(ImputationError) as exc_info:
        imputer.fit(df)
    assert "0 observed values" in str(exc_info.value)


def test_rf_imputer_source_immutability(nonlinear_numeric_df: pd.DataFrame) -> None:
    """Verify that source DataFrame is not mutated during transform."""
    original_copy = nonlinear_numeric_df.copy(deep=True)
    imputer = RandomForestImputerModel(config=RandomForestImputationConfig(n_estimators=10))
    imputer.fit(nonlinear_numeric_df)
    _ = imputer.transform(nonlinear_numeric_df)

    pd.testing.assert_frame_equal(nonlinear_numeric_df, original_copy)


def test_rf_imputer_no_missing_values() -> None:
    """Verify DataFrame with 0 missing values passes through correctly."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        }
    )
    imputer = RandomForestImputerModel(config=RandomForestImputationConfig(n_estimators=10))
    imputed_df, metrics = imputer.fit(df).transform_with_metrics(df)

    pd.testing.assert_frame_equal(imputed_df, df)
    assert all(m.imputed_count == 0 for m in metrics)


def test_rf_imputer_single_numeric_column_fallback() -> None:
    """Verify dataset with single numeric column uses fallback safely."""
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "income": [50000.0, np.nan, 70000.0],
        }
    )
    imputer = RandomForestImputerModel()
    imputed_df = imputer.fit_transform(df)

    assert imputed_df["income"].isna().sum() == 0
    assert imputed_df.loc[1, "income"] == 60000.0  # Median fallback


def test_rf_engine_execution(nonlinear_numeric_df: pd.DataFrame) -> None:
    """Verify BaselineImputationEngine.impute_rf_dataset execution and audit output."""
    engine = BaselineImputationEngine()
    result = engine.impute_rf_dataset(
        nonlinear_numeric_df,
        experiment_id="rf_engine_test",
        n_estimators=20,
        random_seed=42,
    )

    assert result.experiment_id == "rf_engine_test"
    assert result.total_records == 25
    assert result.total_cells_imputed > 0
    assert result.imputed_dataset["feat_x"].isna().sum() == 0
    assert result.imputed_dataset["feat_y"].isna().sum() == 0

    json_meta = result.to_json()
    assert "rf_engine_test" in json_meta
    assert "n_estimators" in json_meta
