"""Unit tests for IterativeImputerModel and IterativeImputationConfig."""

import numpy as np
import pandas as pd
import pytest

from missing_data_platform.exceptions import ConfigurationError, DataQualityError, ImputationError
from missing_data_platform.imputation.iterative import (
    ImputationOrder,
    InitialStrategy,
    IterativeImputationConfig,
    IterativeImputerModel,
)


@pytest.fixture
def correlated_numeric_df() -> pd.DataFrame:
    """Fixture with strongly correlated numeric features (y = 2*x + noise)."""
    x = [float(i) for i in range(1, 21)]
    y = [float(2 * xi + 0.1 * (xi % 3)) for xi in x]
    # Insert missing values
    y[5] = np.nan
    y[12] = np.nan
    x[8] = np.nan

    return pd.DataFrame(
        {
            "customer_id": [f"C{i:02d}" for i in range(20)],
            "feat_x": x,
            "feat_y": y,
            "feat_z": [float(xi * 0.5) for xi in range(20)],
            "purchase_next_month": [i % 2 for i in range(20)],
        }
    )


def test_iterative_config_validation() -> None:
    """Verify IterativeImputationConfig validation invariants."""
    config = IterativeImputationConfig(
        max_iter=15,
        tol=1e-4,
        initial_strategy=InitialStrategy.MEDIAN,
        imputation_order=ImputationOrder.DESCENDING,
        random_seed=42,
    )
    assert config.max_iter == 15
    assert config.tol == 1e-4
    assert config.initial_strategy == InitialStrategy.MEDIAN
    assert config.imputation_order == ImputationOrder.DESCENDING

    with pytest.raises(ConfigurationError):
        IterativeImputationConfig(max_iter=0)

    with pytest.raises(ConfigurationError):
        IterativeImputationConfig(tol=-0.5)

    with pytest.raises(ConfigurationError):
        IterativeImputationConfig(target_features=["feat_x", "purchase_next_month"])


def test_iterative_imputer_multivariate_imputation(correlated_numeric_df: pd.DataFrame) -> None:
    """Verify multivariate iterative chained equations reconstruct correlated values."""
    config = IterativeImputationConfig(
        max_iter=10,
        random_seed=42,
        target_features=["feat_x", "feat_y", "feat_z"],
    )
    imputer = IterativeImputerModel(config=config)
    imputed_df, metrics = imputer.fit(correlated_numeric_df).transform_with_metrics(
        correlated_numeric_df
    )

    # Missing values in x and y should be filled
    assert imputed_df["feat_x"].isna().sum() == 0
    assert imputed_df["feat_y"].isna().sum() == 0

    # The reconstructed y[5] (where x=6) should be approx 12 (since y = 2*x)
    imputed_y5 = imputed_df.loc[5, "feat_y"]
    assert 11.0 <= imputed_y5 <= 13.0

    # Protected columns untouched
    assert imputed_df["customer_id"].tolist() == correlated_numeric_df["customer_id"].tolist()
    assert (
        imputed_df["purchase_next_month"].tolist()
        == correlated_numeric_df["purchase_next_month"].tolist()
    )


def test_iterative_imputer_unfitted_transform_raises_error() -> None:
    """Verify calling transform on unfitted IterativeImputerModel raises ImputationError."""
    imputer = IterativeImputerModel()
    with pytest.raises(ImputationError):
        imputer.transform(pd.DataFrame({"feat_x": [1.0, np.nan]}))


def test_iterative_imputer_empty_df_raises_error() -> None:
    """Verify fitting on empty DataFrame raises DataQualityError."""
    imputer = IterativeImputerModel()
    with pytest.raises(DataQualityError):
        imputer.fit(pd.DataFrame())
