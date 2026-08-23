"""Mandatory Data Leakage Tests for the Evaluation Framework."""

import pandas as pd

from missing_data_platform.evaluation.engine import ImputationEvaluator
from missing_data_platform.imputation.engine import BaselineImputationEngine
from missing_data_platform.masking.config import MaskingConfig, MaskingStrategy
from missing_data_platform.masking.engine import MaskingEngine


def test_ground_truth_alteration_does_not_affect_imputer_predictions() -> None:
    """Verify that altering evaluation ground truth values cannot influence imputer predictions."""
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:02d}" for i in range(20)],
            "x": [float(i) for i in range(20)],
            "y": [float(2 * i) for i in range(20)],
            "purchase_next_month": [i % 2 for i in range(20)],
        }
    )

    mask_config = MaskingConfig(
        experiment_id="leakage_test",
        mask_rate=0.2,
        random_seed=42,
        strategy=MaskingStrategy.UNIFORM_RANDOM,
        target_features=["y"],
    )

    masking_engine = MaskingEngine()
    mask_res = masking_engine.generate_benchmark_dataset(df, mask_config)

    # Impute dataset with masked values
    imp_engine = BaselineImputationEngine()
    imp_res_1 = imp_engine.impute_rf_dataset(
        mask_res.masked_dataset,
        experiment_id="rf_leak_1",
        random_seed=42,
    )

    # Now simulate altered ground-truth evaluation values
    tampered_gt_store = mask_res.ground_truth_store
    tampered_gt_store.original_values["y"] = tampered_gt_store.original_values["y"] + 1_000_000.0

    # Imputer receives the same masked input dataset and must produce identical predictions
    imp_res_2 = imp_engine.impute_rf_dataset(
        mask_res.masked_dataset,
        experiment_id="rf_leak_2",
        random_seed=42,
    )

    pd.testing.assert_frame_equal(imp_res_1.imputed_dataset, imp_res_2.imputed_dataset)

    # Evaluator reflects the tampered ground truth (larger MAE) without altering imputer output
    evaluator = ImputationEvaluator()
    eval_res = evaluator.evaluate_method(
        imp_res_2.imputed_dataset,
        tampered_gt_store,
        method_name="rf",
    )
    assert eval_res.weighted_mae is not None and eval_res.weighted_mae > 900_000.0
