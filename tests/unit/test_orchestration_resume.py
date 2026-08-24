"""Unit tests verifying pipeline resumption and stage skip behavior."""

from pathlib import Path

import pandas as pd

from missing_data_platform.orchestration.checkpoints import CheckpointManager
from missing_data_platform.orchestration.config import (
    ExecutionConfig,
    ExperimentPipelineConfig,
)
from missing_data_platform.orchestration.orchestrator import PipelineOrchestrator
from missing_data_platform.orchestration.stages import PipelineStage, StageStatus


def test_pipeline_safe_resumption(tmp_path: Path) -> None:
    """Assert pipeline skips previously verified stages when resuming from checkpoint."""
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

    chk_dir = tmp_path / "checkpoints"
    out_dir = tmp_path / "outputs"

    chk_manager = CheckpointManager(checkpoint_dir=chk_dir)
    orchestrator = PipelineOrchestrator(checkpoint_manager=chk_manager)

    config = ExperimentPipelineConfig(
        experiment_id="resume_exp",
        imputation_methods=["baseline_median"],
        execution=ExecutionConfig(
            enable_checkpointing=True,
            checkpoint_dir=str(chk_dir),
            output_dir=str(out_dir),
            resume_from_checkpoint=False,
        ),
    )

    # 1. Run fresh execution to create valid checkpoints
    manifest_initial = orchestrator.execute_pipeline(df, config, run_id="run_fresh")
    assert manifest_initial.final_status == StageStatus.COMPLETED
    assert (
        manifest_initial.stage_statuses[PipelineStage.MASKING.value] == StageStatus.COMPLETED.value
    )

    # 2. Run with resume_from_checkpoint = True for the same run
    config_resume = ExperimentPipelineConfig(
        experiment_id="resume_exp",
        imputation_methods=["baseline_median"],
        execution=ExecutionConfig(
            enable_checkpointing=True,
            checkpoint_dir=str(chk_dir),
            output_dir=str(out_dir),
            resume_from_checkpoint=True,
        ),
    )

    manifest_resumed = orchestrator.execute_pipeline(df, config_resume, run_id="run_fresh")
    assert manifest_resumed.final_status == StageStatus.COMPLETED
    # MASKING should be skipped because verified checkpoint existed
    assert manifest_resumed.stage_statuses[PipelineStage.MASKING.value] == StageStatus.SKIPPED.value
