"""Unit tests verifying failure handling, partial failures, and resource protection."""

from pathlib import Path

import pandas as pd

from missing_data_platform.orchestration.config import (
    ExecutionConfig,
    ExperimentPipelineConfig,
    ResourceLimitsConfig,
)
from missing_data_platform.orchestration.orchestrator import PipelineOrchestrator
from missing_data_platform.orchestration.stages import PipelineStage, StageStatus


def test_pipeline_resource_limit_failure(tmp_path: Path) -> None:
    """Assert pipeline halts in ENVIRONMENT_VALIDATION when dataset exceeds quota."""
    df = pd.DataFrame(
        {
            "customer_id": [f"CUST_{i}" for i in range(20)],
            "age": [30.0] * 20,
            "purchase_next_month": [1] * 20,
        }
    )

    config = ExperimentPipelineConfig(
        experiment_id="quota_fail_exp",
        imputation_methods=["baseline_median"],
        resource_limits=ResourceLimitsConfig(max_records=10),
        execution=ExecutionConfig(output_dir=str(tmp_path / "artifacts")),
    )

    orchestrator = PipelineOrchestrator()
    manifest = orchestrator.execute_pipeline(df, config)

    assert manifest.final_status == StageStatus.FAILED
    assert (
        manifest.stage_statuses[PipelineStage.ENVIRONMENT_VALIDATION.value]
        == StageStatus.FAILED.value
    )


def test_pipeline_partial_imputation_failure(tmp_path: Path) -> None:
    """Assert partial failure tracking when one algorithm fails while others succeed."""
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

    config = ExperimentPipelineConfig(
        experiment_id="partial_fail_exp",
        # Include an invalid/failing algorithm name along with valid ones
        imputation_methods=["baseline_median", "invalid_algorithm_name"],
        execution=ExecutionConfig(
            allow_partial_imputation_failure=True,
            output_dir=str(tmp_path / "artifacts"),
        ),
    )

    orchestrator = PipelineOrchestrator()
    manifest = orchestrator.execute_pipeline(df, config)

    assert manifest.partial_failure is True
    assert manifest.method_statuses["invalid_algorithm_name"]["status"] == StageStatus.FAILED.value
    assert manifest.method_statuses["baseline_median"]["status"] == StageStatus.COMPLETED.value
