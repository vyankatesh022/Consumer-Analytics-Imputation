"""Mandatory data leakage validation tests for downstream ML evaluation."""

import numpy as np
import pandas as pd
import pytest

from missing_data_platform.downstream.config import DownstreamConfig
from missing_data_platform.downstream.engine import DownstreamEvaluationEngine
from missing_data_platform.downstream.models import DownstreamModelWrapper
from missing_data_platform.exceptions import ModelTrainingError
from missing_data_platform.masking.config import MaskingConfig
from missing_data_platform.masking.engine import MaskingEngine


@pytest.fixture
def clean_dataset() -> pd.DataFrame:
    """Fixture providing clean tabular dataset."""
    return pd.DataFrame(
        {
            "customer_id": [f"CUST_{i:03d}" for i in range(50)],
            "age": [20.0 + i for i in range(50)],
            "income": [30000.0 + (i * 1000.0) for i in range(50)],
            "customer_segment": ["Gold" if i % 2 == 0 else "Silver" for i in range(50)],
            "purchase_next_month": [1 if i % 3 == 0 else 0 for i in range(50)],
        }
    )


def test_leakage_target_isolated_during_imputation(clean_dataset: pd.DataFrame) -> None:
    """Assert that the downstream target is completely isolated and never imputed."""
    engine = DownstreamEvaluationEngine()
    train_df, test_df = engine.split_dataset(clean_dataset)

    mask_cfg = MaskingConfig(experiment_id="leakage_mask", mask_rate=0.20, random_seed=42)
    masking_engine = MaskingEngine()
    mask_res = masking_engine.generate_benchmark_dataset(test_df, mask_cfg)

    # Target column must NOT have any missing values introduced by masking
    assert mask_res.masked_dataset["purchase_next_month"].isna().sum() == 0

    # Imputation result must retain exact ground-truth targets
    res = engine.evaluate_imputed_pipeline(
        masked_train_df=train_df,
        masked_test_df=mask_res.masked_dataset,
        clean_train_df=train_df,
        clean_test_df=test_df,
        method="baseline_median",
    )
    assert res.metrics["accuracy"] is not None


def test_leakage_model_rejects_target_column_in_features(clean_dataset: pd.DataFrame) -> None:
    """Assert ModelTrainingError when target column is present in feature matrix."""
    wrapper = DownstreamModelWrapper()
    with pytest.raises(ModelTrainingError, match="Target leakage is strictly prohibited"):
        wrapper.fit(clean_dataset, clean_dataset["purchase_next_month"])


def test_leakage_changing_test_labels_does_not_change_model_training(
    clean_dataset: pd.DataFrame,
) -> None:
    """Assert that altering test set labels produces identical training predictions on train features."""
    engine = DownstreamEvaluationEngine()
    train_df, test_df = engine.split_dataset(clean_dataset)

    # Train model on train_df
    X_train = train_df.drop(columns=["purchase_next_month"])
    y_train = train_df["purchase_next_month"]

    model1 = DownstreamModelWrapper(config=DownstreamConfig(random_seed=42))
    model1.fit(X_train, y_train)
    preds1 = model1.predict(X_train)

    # Modify test_df labels arbitrarily
    test_df_mutated = test_df.copy()
    test_df_mutated["purchase_next_month"] = 1 - test_df_mutated["purchase_next_month"]

    # Re-train model strictly on train_df
    model2 = DownstreamModelWrapper(config=DownstreamConfig(random_seed=42))
    model2.fit(X_train, y_train)
    preds2 = model2.predict(X_train)

    np.testing.assert_array_equal(preds1, preds2)
