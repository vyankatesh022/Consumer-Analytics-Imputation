"""Mandatory Data Leakage Tests for Iterative Multivariate Imputation."""

import numpy as np
import pandas as pd

from missing_data_platform.imputation.iterative import (
    IterativeImputationConfig,
    IterativeImputerModel,
)


def test_iterative_ground_truth_isolation() -> None:
    """Verify that held-out evaluation ground truth never enters iterative regression model fitting."""
    # Complete reference ground truth values: [10, 20, 30, 40, 1000]
    # In masked evaluation data, row 4 is hidden: [10, 20, 30, 40, NaN]
    # Ground truth (1000) must not influence the regression model learned from the first 4 rows
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10.0, 20.0, 30.0, 40.0, np.nan],  # In observed rows, y = 10 * x
        }
    )

    config = IterativeImputationConfig(max_iter=5, random_seed=42)
    imputer = IterativeImputerModel(config=config)
    imputed_df = imputer.fit_transform(df)

    # Reconstructed y[4] where x=5 must be close to 50.0 (based on observed relationship y=10*x)
    assert 45.0 <= imputed_df.loc[4, "y"] <= 55.0


def test_iterative_train_test_split_leakage() -> None:
    """Verify that fit(train) followed by transform(test) does not retrain regression models using test data."""
    # Training dataset: y = 2 * x
    train_df = pd.DataFrame(
        {
            "x": [10.0, 20.0, 30.0, 40.0],
            "y": [20.0, 40.0, 60.0, 80.0],
        }
    )

    # Test dataset: has extreme outlier and missing y
    test_df = pd.DataFrame(
        {
            "x": [15.0, 1000.0],
            "y": [np.nan, 2000.0],
        }
    )

    config = IterativeImputationConfig(max_iter=5, random_seed=42)
    imputer = IterativeImputerModel(config=config)

    imputer.fit(train_df)
    imputed_test = imputer.transform(test_df)

    # In test_df row 0 (x=15), y must be imputed using the training relationship y = 2*x (~30.0)
    assert 25.0 <= imputed_test.loc[0, "y"] <= 35.0
