"""Mandatory Reproducibility Tests for the Evaluation Framework."""

import pandas as pd

from missing_data_platform.evaluation.engine import ImputationEvaluator
from missing_data_platform.masking.config import MaskingConfig, MaskingStrategy


def test_benchmark_suite_exact_reproducibility() -> None:
    """Verify that running benchmark suite twice with identical seeds produces identical rankings and metrics."""
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:02d}" for i in range(25)],
            "age": [float(20 + i) for i in range(25)],
            "income": [float(30000 + 1000 * i) for i in range(25)],
            "purchase_next_month": [i % 2 for i in range(25)],
        }
    )

    mask_config = MaskingConfig(
        experiment_id="repro_eval",
        mask_rate=0.20,
        random_seed=777,
        strategy=MaskingStrategy.UNIFORM_RANDOM,
        target_features=["age", "income"],
    )

    evaluator = ImputationEvaluator()
    report_1 = evaluator.run_benchmark_suite(
        df=df,
        mask_config=mask_config,
        methods=["baseline_median", "knn", "random_forest"],
    )
    report_2 = evaluator.run_benchmark_suite(
        df=df,
        mask_config=mask_config,
        methods=["baseline_median", "knn", "random_forest"],
    )

    assert report_1.method_rankings == report_2.method_rankings
    pd.testing.assert_frame_equal(
        report_1.to_summary_dataframe(),
        report_2.to_summary_dataframe(),
    )


def test_repeated_benchmark_reproducibility() -> None:
    """Verify repeated multi-seed benchmark produces identical aggregated stats."""
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:02d}" for i in range(20)],
            "feat_a": [float(i * 2) for i in range(20)],
            "feat_b": [float(i * 3) for i in range(20)],
            "purchase_next_month": [0] * 20,
        }
    )

    mask_config = MaskingConfig(
        experiment_id="rep_repro",
        mask_rate=0.15,
        random_seed=123,
        strategy=MaskingStrategy.UNIFORM_RANDOM,
        target_features=["feat_a", "feat_b"],
    )

    evaluator = ImputationEvaluator()
    rep_1 = evaluator.run_repeated_benchmark(
        df=df,
        base_mask_config=mask_config,
        seeds=[10, 20],
        methods=["baseline_median", "knn"],
    )
    rep_2 = evaluator.run_repeated_benchmark(
        df=df,
        base_mask_config=mask_config,
        seeds=[10, 20],
        methods=["baseline_median", "knn"],
    )

    assert rep_1.method_stats == rep_2.method_stats
