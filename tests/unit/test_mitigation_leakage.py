"""Mandatory Data Leakage Tests for Bias Mitigation."""

import pandas as pd

from missing_data_platform.masking.config import MaskingConfig, MaskingStrategy
from missing_data_platform.masking.engine import MaskingEngine
from missing_data_platform.mitigation.config import (
    MitigationConfig,
    MitigationStrategy,
)
from missing_data_platform.mitigation.engine import FairnessMitigationEngine


def test_mitigation_training_isolated_from_ground_truth() -> None:
    """Verify that altering evaluation ground truth values does not change mitigated imputer predictions."""
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:02d}" for i in range(30)],
            "customer_segment": ["Gold"] * 20 + ["Bronze"] * 10,
            "x": [float(i) for i in range(30)],
            "y": [float(2 * i) for i in range(30)],
            "purchase_next_month": [0] * 30,
        }
    )

    mask_config = MaskingConfig(
        experiment_id="mit_leak_test",
        mask_rate=0.20,
        random_seed=42,
        strategy=MaskingStrategy.UNIFORM_RANDOM,
        target_features=["y"],
    )

    mask_res = MaskingEngine().generate_benchmark_dataset(df, mask_config)

    config = MitigationConfig(
        enabled=True,
        strategy=MitigationStrategy.SAMPLE_WEIGHTING,
        group_column="customer_segment",
    )
    engine = FairnessMitigationEngine(config=config)

    # Impute 1
    imputed_1 = engine.impute_with_mitigation(mask_res.masked_dataset, method="random_forest")

    # Tamper with ground truth store
    tampered_store = mask_res.ground_truth_store
    tampered_store.original_values["y"] = tampered_store.original_values["y"] + 500_000.0

    # Impute 2 (re-imputing with same masked data and config)
    imputed_2 = engine.impute_with_mitigation(mask_res.masked_dataset, method="random_forest")

    pd.testing.assert_frame_equal(imputed_1, imputed_2)
