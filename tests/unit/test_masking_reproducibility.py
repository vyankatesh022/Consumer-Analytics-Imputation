"""Mandatory reproducibility tests for artificial missingness experiments.

Verifies that two independent executions with identical input, configuration,
and random seed produce bit-for-bit identical masks, ground-truth stores, and datasets.
"""

import pandas as pd

from missing_data_platform.masking.config import MaskingConfig, MaskingStrategy
from missing_data_platform.masking.engine import MaskingEngine


def test_masking_experiment_exact_reproducibility() -> None:
    """Verify bit-for-bit deterministic reproducibility across repeated runs with identical seed."""
    data = {
        "customer_id": [f"C{i:03d}" for i in range(100)],
        "age": [float(20 + i) for i in range(100)],
        "income": [float(50000 + i * 1000) for i in range(100)],
        "total_spend": [float(200 + i * 50) for i in range(100)],
        "purchase_next_month": [i % 2 for i in range(100)],
    }
    df = pd.DataFrame(data)

    config_a = MaskingConfig(
        experiment_id="exp_repro_A",
        mask_rate=0.25,
        random_seed=999,
        strategy=MaskingStrategy.UNIFORM_RANDOM,
        target_features=["age", "income", "total_spend"],
    )

    config_b = MaskingConfig(
        experiment_id="exp_repro_B",
        mask_rate=0.25,
        random_seed=999,  # Identical seed
        strategy=MaskingStrategy.UNIFORM_RANDOM,
        target_features=["age", "income", "total_spend"],
    )

    engine = MaskingEngine()
    result_a = engine.generate_benchmark_dataset(df, config_a)
    result_b = engine.generate_benchmark_dataset(df, config_b)

    # Assert exact equality of boolean mask matrices
    pd.testing.assert_frame_equal(result_a.ground_truth_mask, result_b.ground_truth_mask)

    # Assert exact equality of masked benchmark datasets
    pd.testing.assert_frame_equal(result_a.masked_dataset, result_b.masked_dataset)

    # Assert exact equality of ground-truth extracted values
    for col in ["age", "income", "total_spend"]:
        pd.testing.assert_series_equal(
            result_a.ground_truth_store.get_ground_truth(col),
            result_b.ground_truth_store.get_ground_truth(col),
        )
