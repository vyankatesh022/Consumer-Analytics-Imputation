"""Unit tests verifying end-to-end pipeline reproducibility."""

from pathlib import Path

import pandas as pd

from missing_data_platform.orchestration.config import (
    ExecutionConfig,
    ExperimentPipelineConfig,
)
from missing_data_platform.orchestration.orchestrator import PipelineOrchestrator
from missing_data_platform.orchestration.stages import StageStatus


def test_pipeline_exact_reproducibility(tmp_path: Path) -> None:
    """Assert running pipeline with identical dataset, config, and seed produces identical fingerprints and outcomes."""
    records = []
    for i in range(50):
        records.append(
            {
                "customer_id": f"CUST_{i:04d}",
                "age": 25.0 + (i % 30),
                "gender": "Female" if i % 2 == 0 else "Male",
                "income": 40000.0 + (i * 500.0),
                "education": "Bachelor",
                "occupation": "Engineer",
                "city": "Seattle",
                "region": "West",
                "purchase_frequency": 2.0,
                "average_purchase_value": 70.0,
                "total_spend": 300.0,
                "discount_usage": 0.1,
                "website_visits": 8.0,
                "campaign_exposure": 2.0,
                "product_category": "Electronics",
                "customer_segment": "Gold" if i % 2 == 0 else "Silver",
                "purchase_next_month": 1 if i % 2 == 0 else 0,
            }
        )
    df = pd.DataFrame(records)

    config1 = ExperimentPipelineConfig(
        experiment_id="repro_run_1",
        random_seed=42,
        imputation_methods=["baseline_median", "knn"],
        execution=ExecutionConfig(output_dir=str(tmp_path / "artifacts1")),
    )

    config2 = ExperimentPipelineConfig(
        experiment_id="repro_run_2",
        random_seed=42,
        imputation_methods=["baseline_median", "knn"],
        execution=ExecutionConfig(output_dir=str(tmp_path / "artifacts2")),
    )

    orchestrator = PipelineOrchestrator()
    manifest1 = orchestrator.execute_pipeline(df, config1)
    manifest2 = orchestrator.execute_pipeline(df, config2)

    assert manifest1.final_status == StageStatus.COMPLETED
    assert manifest2.final_status == StageStatus.COMPLETED
    assert manifest1.dataset_fingerprint == manifest2.dataset_fingerprint
    assert manifest1.stage_statuses == manifest2.stage_statuses
