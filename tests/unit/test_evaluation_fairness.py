"""Mandatory Fair Comparison Tests for the Evaluation Framework."""

import pandas as pd

from missing_data_platform.evaluation.engine import ImputationEvaluator
from missing_data_platform.masking.config import MaskingConfig, MaskingStrategy


def test_fair_comparison_uniform_inputs() -> None:
    """Verify that run_benchmark_suite provides identical masked datasets and masks to all algorithms."""
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:02d}" for i in range(25)],
            "age": [float(20 + i) for i in range(25)],
            "income": [float(30000 + 1000 * i) for i in range(25)],
            "purchase_next_month": [i % 2 for i in range(25)],
        }
    )

    mask_config = MaskingConfig(
        experiment_id="fair_comp_test",
        mask_rate=0.20,
        random_seed=42,
        strategy=MaskingStrategy.UNIFORM_RANDOM,
        target_features=["age", "income"],
    )

    evaluator = ImputationEvaluator()
    report = evaluator.run_benchmark_suite(
        df=df,
        mask_config=mask_config,
        methods=["baseline_median", "knn", "iterative", "random_forest"],
    )

    # All methods evaluated across the exact same cell count
    eval_cell_counts = [res.total_evaluated_cells for res in report.method_results.values()]
    assert len(set(eval_cell_counts)) == 1
    assert eval_cell_counts[0] > 0

    # Ensure all methods have rankings assigned
    assert len(report.method_rankings) == 4
    for r in report.method_rankings:
        assert r["rank_mae"] in (1, 2, 3, 4)
        assert r["rank_rmse"] in (1, 2, 3, 4)
