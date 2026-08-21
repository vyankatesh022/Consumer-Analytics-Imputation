"""Mandatory Data Leakage Tests for Imputation Layer.

1. Ground-Truth Data Leakage Test: Verifies that hidden ground-truth evaluation values
   do not influence baseline parameter fitting.
2. Train/Test Data Leakage Test: Verifies that fitting on training data and transforming
   test data does not recompute statistics from test distribution.
"""

import numpy as np
import pandas as pd

from missing_data_platform.imputation.baseline import BaselineImputer
from missing_data_platform.imputation.config import (
    BaselineImputationConfig,
    BaselineStrategy,
)
from missing_data_platform.imputation.engine import BaselineImputationEngine


def test_ground_truth_isolation_leakage() -> None:
    """Verify that artificially masked evaluation ground-truth values do not leak into parameter estimation."""
    # Complete reference ground truth values: [10, 20, 30, 40, 1000] (Mean = 220)
    # Artificially masked dataset has row 4 (value 1000) hidden as NaN: [10, 20, 30, 40, NaN] (Observed Mean = 25)
    masked_df = pd.DataFrame(
        {
            "income": [10.0, 20.0, 30.0, 40.0, np.nan],
        }
    )

    engine = BaselineImputationEngine()
    result = engine.impute_dataset(
        masked_df,
        numeric_strategy=BaselineStrategy.MEAN,
    )

    # Imputed statistic must be 25.0 (from observed values only), NOT 220.0 (leaked ground truth)
    assert result.imputation_parameters["income"] == 25.0
    assert result.imputed_dataset.loc[4, "income"] == 25.0


def test_train_test_split_leakage() -> None:
    """Verify that fit(train) followed by transform(test) applies training parameters without leaking test stats."""
    # Training dataset: income has Mean = 100
    train_df = pd.DataFrame(
        {
            "income": [80.0, 100.0, 120.0],
        }
    )

    # Test dataset: income has Mean = 500 (with missing value in row 1)
    test_df = pd.DataFrame(
        {
            "income": [400.0, np.nan, 600.0],
        }
    )

    config = BaselineImputationConfig(numeric_strategy=BaselineStrategy.MEAN)
    imputer = BaselineImputer(config=config)

    # Fit on train, transform test
    imputer.fit(train_df)
    assert imputer.imputation_parameters["income"] == 100.0

    imputed_test = imputer.transform(test_df)

    # The missing value in test_df must be imputed with training mean (100.0), NOT test mean (500.0)
    assert imputed_test.loc[1, "income"] == 100.0
