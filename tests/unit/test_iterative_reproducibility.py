"""Mandatory reproducibility tests for Iterative Multivariate Imputation."""

import numpy as np
import pandas as pd

from missing_data_platform.imputation.iterative import (
    IterativeImputationConfig,
    IterativeImputerModel,
)


def test_iterative_imputation_exact_reproducibility() -> None:
    """Verify that two executions with identical random_seed produce bit-for-bit identical imputations."""
    data = {
        "customer_id": [f"C{i:02d}" for i in range(30)],
        "a": [float(i) if i % 4 != 0 else np.nan for i in range(30)],
        "b": [float(3 * i + 2) if i % 5 != 0 else np.nan for i in range(30)],
        "c": [float(i * i * 0.1) if i % 6 != 0 else np.nan for i in range(30)],
    }
    df = pd.DataFrame(data)

    config_a = IterativeImputationConfig(max_iter=10, random_seed=777)
    config_b = IterativeImputationConfig(max_iter=10, random_seed=777)

    imputer_a = IterativeImputerModel(config=config_a)
    imputer_b = IterativeImputerModel(config=config_b)

    result_a = imputer_a.fit_transform(df)
    result_b = imputer_b.fit_transform(df)

    pd.testing.assert_frame_equal(result_a, result_b)
