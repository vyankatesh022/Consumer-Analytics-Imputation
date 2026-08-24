"""Unit tests verifying reproducibility of downstream ML evaluation."""

import pandas as pd

from missing_data_platform.downstream.config import (
    DownstreamBenchmarkConfig,
    DownstreamConfig,
    DownstreamModelType,
)
from missing_data_platform.downstream.engine import DownstreamEvaluationEngine
from missing_data_platform.masking.config import MaskingConfig


def test_downstream_evaluation_exact_reproducibility() -> None:
    """Assert that running the downstream evaluation pipeline with identical seeds produces bit-exact metrics."""
    records = []
    for i in range(80):
        records.append(
            {
                "customer_id": f"CUST_{i:04d}",
                "age": 25.0 + (i % 40),
                "gender": "Female" if i % 2 == 0 else "Male",
                "income": 40000.0 + (i * 500.0),
                "education": "Bachelor" if i % 2 == 0 else "Master",
                "occupation": "Engineer",
                "city": "Seattle",
                "region": "West",
                "purchase_frequency": 2.0 + (i % 5),
                "average_purchase_value": 75.0 + (i * 1.5),
                "total_spend": 300.0 + (i * 15.0),
                "discount_usage": 0.1,
                "website_visits": 10.0,
                "campaign_exposure": 2.0,
                "product_category": "Electronics",
                "customer_segment": "Gold" if i % 2 == 0 else "Silver",
                "purchase_next_month": 1 if i % 2 == 0 else 0,
            }
        )
    df = pd.DataFrame(records)

    config = DownstreamConfig(
        model_type=DownstreamModelType.RANDOM_FOREST,
        primary_metric="f1",
        random_seed=42,
    )
    mask_config = MaskingConfig(
        experiment_id="repro_mask",
        mask_rate=0.20,
        random_seed=42,
    )
    bench_config = DownstreamBenchmarkConfig(
        experiment_id="repro_bench",
        methods=["baseline_median", "knn"],
        include_mitigation=False,
        downstream_config=config,
    )

    engine1 = DownstreamEvaluationEngine(config=config)
    report1 = engine1.run_benchmark_suite(
        df, mask_config=mask_config, benchmark_config=bench_config
    )

    engine2 = DownstreamEvaluationEngine(config=config)
    report2 = engine2.run_benchmark_suite(
        df, mask_config=mask_config, benchmark_config=bench_config
    )

    assert report1.complete_baseline.metrics == report2.complete_baseline.metrics
    assert report1.method_results["knn"].metrics == report2.method_results["knn"].metrics
    assert (
        report1.method_results["baseline_median"].metrics
        == report2.method_results["baseline_median"].metrics
    )
