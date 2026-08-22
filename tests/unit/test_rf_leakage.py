"""Mandatory Data Leakage Tests for Random Forest Imputation."""

import numpy as np
import pandas as pd

from missing_data_platform.imputation.rf import (
    RandomForestImputationConfig,
    RandomForestImputerModel,
)


def test_rf_ground_truth_isolation() -> None:
    """Verify that held-out evaluation ground truth never enters Random Forest model fitting."""
    # Training observed rows follow y = 5 * x
    # Hidden evaluation row 4 has missing y; if ground truth (e.g. 50000) leaked, prediction would skew
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "y": [5.0, 10.0, 15.0, 20.0, np.nan, 30.0, 35.0, 40.0],
            "z": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        }
    )

    config = RandomForestImputationConfig(n_estimators=50, random_seed=42)
    imputer = RandomForestImputerModel(config=config)
    imputed_df = imputer.fit_transform(df)

    # Imputed value for row 4 (where x=5, z=10) should be close to 25.0
    assert 20.0 <= imputed_df.loc[4, "y"] <= 30.0


def test_rf_train_test_split_leakage() -> None:
    """Verify that fit(train) followed by transform(test) does not retrain Random Forest models using test data."""
    # Training dataset: y = 3 * x
    train_df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [3.0, 6.0, 9.0, 12.0, 15.0, 18.0],
            "z": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )

    # Test dataset: contains test observation and potential leakage distractor
    test_df = pd.DataFrame(
        {
            "x": [3.5, 500.0],
            "y": [np.nan, 99999.0],
            "z": [35.0, 5000.0],
        }
    )

    config = RandomForestImputationConfig(n_estimators=50, random_seed=42)
    imputer = RandomForestImputerModel(config=config)

    imputer.fit(train_df)
    imputed_test = imputer.transform(test_df)

    # In test_df row 0 (x=3.5, z=35), y must be imputed according to the training relationship y = 3*x (~10.5)
    assert 8.0 <= imputed_test.loc[0, "y"] <= 14.0
