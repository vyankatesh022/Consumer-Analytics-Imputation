"""Security and data privacy tests for downstream evaluation reports and logs."""

import json

import pandas as pd

from missing_data_platform.downstream.config import (
    DownstreamBenchmarkConfig,
    DownstreamConfig,
)
from missing_data_platform.downstream.engine import DownstreamEvaluationEngine
from missing_data_platform.masking.config import MaskingConfig


def test_downstream_report_contains_no_raw_records() -> None:
    """Assert that serialized downstream benchmark reports contain zero individual records, raw PII, or raw targets."""
    records = []
    for i in range(60):
        records.append(
            {
                "customer_id": f"SENSITIVE_ID_{i:04d}",
                "age": 25.0 + (i % 30),
                "gender": "Female" if i % 2 == 0 else "Male",
                "income": 50000.0 + (i * 1000.0),
                "education": "Master",
                "occupation": "Engineer",
                "city": "SecretCity",
                "region": "ConfidentialRegion",
                "purchase_frequency": 3.0,
                "average_purchase_value": 80.0,
                "total_spend": 500.0,
                "discount_usage": 0.1,
                "website_visits": 10.0,
                "campaign_exposure": 2.0,
                "product_category": "Electronics",
                "customer_segment": "PrivateSegment" if i % 2 == 0 else "ConfidentialSegment",
                "purchase_next_month": 1 if i % 2 == 0 else 0,
            }
        )
    df = pd.DataFrame(records)

    config = DownstreamConfig(random_seed=42)
    engine = DownstreamEvaluationEngine(config=config)
    bench_cfg = DownstreamBenchmarkConfig(
        experiment_id="privacy_test_exp",
        methods=["baseline_median", "knn"],
        include_mitigation=False,
    )
    mask_cfg = MaskingConfig(
        experiment_id="privacy_mask_exp",
        mask_rate=0.20,
        random_seed=42,
    )

    report = engine.run_benchmark_suite(df, mask_config=mask_cfg, benchmark_config=bench_cfg)
    report_json = report.to_json()
    report_dict = json.loads(report_json)

    # Asserts
    assert "SENSITIVE_ID_0001" not in report_json
    assert "SecretCity" not in report_json
    assert "ConfidentialRegion" not in report_json

    # Asserts that metrics are aggregated numbers, not arrays of predictions
    for _method, res in report_dict["method_results"].items():
        assert isinstance(res["metrics"], dict)
        assert isinstance(res["metrics"]["f1"], (float, int, type(None)))
        assert "individual_predictions" not in res
        assert "raw_labels" not in res
