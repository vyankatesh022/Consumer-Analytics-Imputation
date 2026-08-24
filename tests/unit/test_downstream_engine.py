"""Unit tests for Downstream ML Evaluation Engine."""

import pandas as pd
import pytest

from missing_data_platform.downstream.config import (
    DownstreamBenchmarkConfig,
    DownstreamConfig,
    DownstreamModelType,
)
from missing_data_platform.downstream.engine import DownstreamEvaluationEngine
from missing_data_platform.masking.config import MaskingConfig


@pytest.fixture
def synthetic_complete_df() -> pd.DataFrame:
    """Generate a clean synthetic consumer dataset for testing."""
    records = []
    for i in range(120):
        records.append(
            {
                "customer_id": f"CUST_{i:04d}",
                "age": 20.0 + (i % 50),
                "gender": "Female" if i % 2 == 0 else "Male",
                "income": 30000.0 + (i * 800.0),
                "education": "Bachelor" if i % 3 == 0 else ("Master" if i % 3 == 1 else "PhD"),
                "occupation": "Engineer" if i % 2 == 0 else "Manager",
                "city": "Seattle" if i % 4 == 0 else "Austin",
                "region": "West" if i % 2 == 0 else "South",
                "purchase_frequency": 1.0 + (i % 10),
                "average_purchase_value": 50.0 + (i * 2.0),
                "total_spend": 200.0 + (i * 20.0),
                "discount_usage": 0.1 * (i % 5),
                "website_visits": 5.0 + (i % 15),
                "campaign_exposure": 1.0 + (i % 4),
                "product_category": "Electronics" if i % 2 == 0 else "Apparel",
                "customer_segment": "Gold"
                if i % 4 == 0
                else ("Silver" if i % 4 == 1 else ("Bronze" if i % 4 == 2 else "Platinum")),
                "purchase_next_month": 1 if (i % 3 == 0 or i % 5 == 0) else 0,
            }
        )
    return pd.DataFrame(records)


def test_downstream_engine_split(synthetic_complete_df: pd.DataFrame) -> None:
    """Assert deterministic train/test partition preserving schemas."""
    engine = DownstreamEvaluationEngine()
    train_df, test_df = engine.split_dataset(synthetic_complete_df)

    assert len(train_df) + len(test_df) == len(synthetic_complete_df)
    assert len(test_df) == int(len(synthetic_complete_df) * 0.20)
    assert "purchase_next_month" in train_df.columns
    assert "purchase_next_month" in test_df.columns


def test_downstream_complete_baseline(synthetic_complete_df: pd.DataFrame) -> None:
    """Assert complete-data baseline produces reference metrics."""
    engine = DownstreamEvaluationEngine()
    train_df, test_df = engine.split_dataset(synthetic_complete_df)

    res = engine.evaluate_complete_baseline(train_df, test_df)

    assert res.imputation_method == "complete_reference"
    assert res.metrics["f1"] is not None
    assert res.metrics["accuracy"] is not None
    assert res.recovery == 100.0
    assert len(res.group_metrics) > 0


def test_downstream_benchmark_suite(synthetic_complete_df: pd.DataFrame) -> None:
    """Assert full benchmark comparison runs complete, imputed, and mitigated pipelines."""
    config = DownstreamConfig(
        model_type=DownstreamModelType.RANDOM_FOREST,
        primary_metric="f1",
        random_seed=42,
    )
    bench_config = DownstreamBenchmarkConfig(
        experiment_id="test_bench_exp",
        methods=["baseline_median", "knn", "random_forest"],
        include_mitigation=True,
        downstream_config=config,
    )
    mask_config = MaskingConfig(
        experiment_id="test_mask_exp",
        mask_rate=0.15,
        random_seed=42,
    )

    engine = DownstreamEvaluationEngine(config=config)
    report = engine.run_benchmark_suite(
        synthetic_complete_df,
        mask_config=mask_config,
        benchmark_config=bench_config,
    )

    assert report.complete_baseline is not None
    assert "baseline_median" in report.method_results
    assert "knn" in report.method_results
    assert "random_forest" in report.method_results
    assert "fairness_weighted_rf" in report.mitigated_results

    df_summary = report.to_comparison_dataframe()
    assert len(df_summary) == 5  # complete + 3 methods + 1 mitigated
    assert "method" in df_summary.columns
    assert "primary_metric" in df_summary.columns
    assert "recovery" in df_summary.columns


def test_downstream_missingness_rate_curve(synthetic_complete_df: pd.DataFrame) -> None:
    """Assert missingness rate sweep generates degradation curves."""
    bench_config = DownstreamBenchmarkConfig(
        experiment_id="test_curve_exp",
        methods=["baseline_median", "knn"],
        missingness_rates=[0.10, 0.30],
        include_mitigation=False,
    )
    engine = DownstreamEvaluationEngine()
    curve_report = engine.run_missingness_rate_curve(
        synthetic_complete_df,
        benchmark_config=bench_config,
    )

    assert len(curve_report.curve_points) == 4  # 2 rates * 2 methods
    df_curve = curve_report.to_dataframe()
    assert set(df_curve["missingness_rate"].unique()) == {0.10, 0.30}


def test_downstream_repeated_benchmark(synthetic_complete_df: pd.DataFrame) -> None:
    """Assert repeated seed benchmark computes mean, std, and confidence intervals."""
    bench_config = DownstreamBenchmarkConfig(
        experiment_id="test_rep_exp",
        methods=["baseline_median", "knn"],
        repeated_seeds=[42, 123],
        include_mitigation=False,
    )
    engine = DownstreamEvaluationEngine()
    rep_report = engine.run_repeated_benchmark(
        synthetic_complete_df,
        seeds=[42, 123],
        benchmark_config=bench_config,
    )

    assert rep_report.total_repetitions == 2
    assert "baseline_median" in rep_report.method_stats
    assert "mean" in rep_report.method_stats["baseline_median"]
    assert "std" in rep_report.method_stats["baseline_median"]
    assert "ci_95_low" in rep_report.method_stats["baseline_median"]
    assert "ci_95_high" in rep_report.method_stats["baseline_median"]
