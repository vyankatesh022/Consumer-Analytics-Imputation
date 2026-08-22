"""Mandatory Data Leakage Tests for KNN Imputation Layer."""

import numpy as np
import pandas as pd

from missing_data_platform.imputation.knn import (
    KNNImputationConfig,
    KNNImputerModel,
    ScalingStrategy,
)


def test_knn_ground_truth_isolation() -> None:
    """Verify that held-out evaluation ground truth values never influence KNN scaling or neighbor calculations."""
    # Reference data: row 3 is masked (observed NaN). Ground truth value would be 500000.
    # Neighbors of row 3 (age=30, spend=300) in observed data are row 1 (income=40000) and row 2 (income=42000).
    df = pd.DataFrame(
        {
            "age": [29.0, 31.0, 30.0],
            "total_spend": [290.0, 310.0, 300.0],
            "income": [40000.0, 42000.0, np.nan],
        }
    )

    config = KNNImputationConfig(n_neighbors=2)
    imputer = KNNImputerModel(config=config)
    imputed_df = imputer.fit_transform(df)

    # Imputed value must be computed solely from observed neighbors (~41000.0)
    assert 40000.0 <= imputed_df.loc[2, "income"] <= 42000.0


def test_knn_train_test_split_leakage() -> None:
    """Verify that fit(train) followed by transform(test) does not refit scaler or neighbor space using test data."""
    # Training dataset: age in [20, 22], income in [30000, 32000]
    train_df = pd.DataFrame(
        {
            "age": [20.0, 21.0, 22.0],
            "income": [30000.0, 31000.0, 32000.0],
        }
    )

    # Test dataset with missing income and extreme age (100.0)
    test_df = pd.DataFrame(
        {
            "age": [21.0, 100.0],
            "income": [np.nan, 900000.0],
        }
    )

    config = KNNImputationConfig(n_neighbors=2, scaling_strategy=ScalingStrategy.STANDARD)
    imputer = KNNImputerModel(config=config)

    imputer.fit(train_df)
    scaler_mean_train = float(imputer._scaler.mean_[0])  # type: ignore[union-attr]
    assert np.isclose(scaler_mean_train, 21.0)

    imputed_test = imputer.transform(test_df)

    # Scaler mean must NOT have changed to include test age 100.0
    assert np.isclose(float(imputer._scaler.mean_[0]), 21.0)  # type: ignore[union-attr]

    # Row 0 in test (age=21) must be imputed with training neighbors (~31000.0)
    assert 30000.0 <= imputed_test.loc[0, "income"] <= 32000.0
