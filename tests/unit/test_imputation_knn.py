"""Unit tests for KNNImputerModel and KNNImputationConfig."""

import numpy as np
import pandas as pd
import pytest

from missing_data_platform.exceptions import ConfigurationError, DataQualityError, ImputationError
from missing_data_platform.imputation.knn import (
    KNNImputationConfig,
    KNNImputerModel,
    KNNWeighting,
    ScalingStrategy,
)


@pytest.fixture
def knn_synthetic_df() -> pd.DataFrame:
    """Fixture providing numeric features for KNN similarity estimation."""
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4", "C5"],
            "age": [20.0, 21.0, 22.0, 50.0, 52.0],
            "income": [30000.0, 31000.0, np.nan, 90000.0, 95000.0],  # C3 is closest to C1, C2
            "total_spend": [100.0, 110.0, 105.0, 500.0, 550.0],
            "purchase_next_month": [0, 0, 1, 1, 1],
        }
    )


def test_knn_config_validation() -> None:
    """Verify KNNImputationConfig validation rules."""
    config = KNNImputationConfig(n_neighbors=3, weights=KNNWeighting.DISTANCE)
    assert config.n_neighbors == 3
    assert config.weights == KNNWeighting.DISTANCE

    # k < 1 must raise ConfigurationError
    with pytest.raises(ConfigurationError):
        KNNImputationConfig(n_neighbors=0)

    # Targeting protected feature must raise ConfigurationError
    with pytest.raises(ConfigurationError):
        KNNImputationConfig(target_features=["income", "customer_id"])


def test_knn_imputer_standard_scaling(knn_synthetic_df: pd.DataFrame) -> None:
    """Verify KNN imputation with standard scaling produces expected neighbor estimate."""
    config = KNNImputationConfig(
        n_neighbors=2,
        scaling_strategy=ScalingStrategy.STANDARD,
    )
    imputer = KNNImputerModel(config=config)
    imputed_df, metrics = imputer.fit(knn_synthetic_df).transform_with_metrics(knn_synthetic_df)

    # C3 (age=22, spend=105) is closest to C1 (30000) and C2 (31000) -> expected income approx 30500
    imputed_val = imputed_df.loc[2, "income"]
    assert 30000.0 <= imputed_val <= 31500.0
    assert imputed_df["income"].isna().sum() == 0

    # Protected columns untouched
    assert imputed_df["customer_id"].tolist() == ["C1", "C2", "C3", "C4", "C5"]
    assert imputed_df["purchase_next_month"].tolist() == [0, 0, 1, 1, 1]


def test_knn_imputer_minmax_scaling(knn_synthetic_df: pd.DataFrame) -> None:
    """Verify KNN imputation with MinMaxScaler."""
    config = KNNImputationConfig(
        n_neighbors=2,
        scaling_strategy=ScalingStrategy.MINMAX,
    )
    imputer = KNNImputerModel(config=config)
    imputed_df = imputer.fit_transform(knn_synthetic_df)
    assert imputed_df["income"].isna().sum() == 0


def test_knn_imputer_no_scaling(knn_synthetic_df: pd.DataFrame) -> None:
    """Verify KNN imputation with raw unscaled features."""
    config = KNNImputationConfig(
        n_neighbors=2,
        scaling_strategy=ScalingStrategy.NONE,
    )
    imputer = KNNImputerModel(config=config)
    imputed_df = imputer.fit_transform(knn_synthetic_df)
    assert imputed_df["income"].isna().sum() == 0


def test_knn_imputer_zero_distance_handling() -> None:
    """Verify safe zero-distance handling when identical points exist."""
    df = pd.DataFrame(
        {
            "age": [25.0, 25.0, 25.0],
            "income": [50000.0, 50000.0, np.nan],
        }
    )
    config = KNNImputationConfig(n_neighbors=2)
    imputer = KNNImputerModel(config=config)
    imputed_df = imputer.fit_transform(df)
    assert imputed_df.loc[2, "income"] == 50000.0


def test_knn_imputer_unfitted_transform_raises_error() -> None:
    """Verify calling transform on unfitted KNNImputerModel raises ImputationError."""
    imputer = KNNImputerModel()
    with pytest.raises(ImputationError):
        imputer.transform(pd.DataFrame({"income": [10.0, np.nan]}))


def test_knn_imputer_empty_df_raises_error() -> None:
    """Verify fitting on empty DataFrame raises DataQualityError."""
    imputer = KNNImputerModel()
    with pytest.raises(DataQualityError):
        imputer.fit(pd.DataFrame())
