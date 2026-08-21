"""Unit tests for BaselineImputationEngine."""

import numpy as np
import pandas as pd

from missing_data_platform.imputation.config import BaselineStrategy
from missing_data_platform.imputation.engine import BaselineImputationEngine


def test_baseline_imputation_engine_execution() -> None:
    """Verify BaselineImputationEngine completes execution and generates ImputationResult."""
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "age": [25.0, np.nan, 35.0],
            "gender": ["Male", np.nan, "Female"],
            "purchase_next_month": [1, 0, 1],
        }
    )

    engine = BaselineImputationEngine()
    result = engine.impute_dataset(
        df,
        experiment_id="engine_test_exp",
        numeric_strategy=BaselineStrategy.MEAN,
        categorical_strategy=BaselineStrategy.MODE,
    )

    assert result.experiment_id == "engine_test_exp"
    assert result.total_records == 3
    assert result.total_cells_imputed == 2
    assert result.imputed_dataset["age"].isna().sum() == 0
    assert result.imputed_dataset["gender"].isna().sum() == 0

    # JSON export
    json_str = result.to_json()
    assert "engine_test_exp" in json_str
    assert "total_cells_imputed" in json_str


def test_baseline_engine_preserves_input_immutability() -> None:
    """Verify that source DataFrame is not mutated during imputation."""
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "age": [30.0, np.nan],
            "purchase_next_month": [1, 0],
        }
    )
    original_copy = df.copy(deep=True)

    engine = BaselineImputationEngine()
    _ = engine.impute_dataset(df)

    pd.testing.assert_frame_equal(df, original_copy)
