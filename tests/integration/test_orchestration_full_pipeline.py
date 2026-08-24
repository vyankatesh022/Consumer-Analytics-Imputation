"""Integration test executing the complete 11-stage hardened pipeline."""

from pathlib import Path

import pandas as pd

from missing_data_platform.orchestration.config import (
    ExecutionConfig,
    ExperimentPipelineConfig,
)
from missing_data_platform.orchestration.orchestrator import PipelineOrchestrator
from missing_data_platform.orchestration.stages import PipelineStage, StageStatus


def test_orchestration_full_11_stage_pipeline(tmp_path: Path) -> None:
    """Execute complete hardened 11-stage experiment pipeline and verify manifest."""
    records = []
    for i in range(120):
        records.append(
            {
                "customer_id": f"CUST_{i:04d}",
                "age": 20.0 + (i % 45),
                "gender": "Female" if i % 2 == 0 else "Male",
                "income": 35000.0 + (i * 600.0),
                "education": "Bachelor" if i % 3 == 0 else "Master",
                "occupation": "Engineer",
                "city": "Seattle" if i % 2 == 0 else "Austin",
                "region": "West" if i % 2 == 0 else "South",
                "purchase_frequency": 2.0 + (i % 6),
                "average_purchase_value": 60.0 + i,
                "total_spend": 300.0 + (i * 10),
                "discount_usage": 0.15,
                "website_visits": 8.0 + (i % 10),
                "campaign_exposure": 2.0,
                "product_category": "Electronics",
                "customer_segment": "Gold" if i % 2 == 0 else "Silver",
                "purchase_next_month": 1 if i % 3 == 0 else 0,
            }
        )
    df = pd.DataFrame(records)

    art_dir = tmp_path / "artifacts"
    chk_dir = tmp_path / "checkpoints"

    config = ExperimentPipelineConfig(
        experiment_id="full_e2e_hardened_pipeline",
        dataset_version="1.0.0",
        random_seed=42,
        imputation_methods=["baseline_median", "knn", "random_forest"],
        execution=ExecutionConfig(
            enable_checkpointing=True,
            checkpoint_dir=str(chk_dir),
            output_dir=str(art_dir),
        ),
    )
    config.mitigation.enabled = True

    orchestrator = PipelineOrchestrator()
    manifest = orchestrator.execute_pipeline(df, config, run_id="e2e_run_01")

    # Assertions
    assert manifest.final_status == StageStatus.COMPLETED
    assert manifest.partial_failure is False
    assert len(manifest.artifact_references) >= 4

    # Assert all 11 stages completed
    for stage in PipelineStage:
        assert stage.value in manifest.stage_statuses
        assert manifest.stage_statuses[stage.value] == StageStatus.COMPLETED.value

    # Assert manifest file persisted to disk
    manifest_file = art_dir / "full_e2e_hardened_pipeline" / "e2e_run_01" / "manifest.json"
    assert manifest_file.exists()
