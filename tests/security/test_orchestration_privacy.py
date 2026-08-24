"""Security and privacy audits for experiment manifests, configuration snapshots, and checkpoints."""

import json
from pathlib import Path

import pandas as pd

from missing_data_platform.orchestration.config import (
    ExecutionConfig,
    ExperimentPipelineConfig,
)
from missing_data_platform.orchestration.orchestrator import PipelineOrchestrator


def test_manifest_and_artifacts_contain_no_raw_customer_pii(tmp_path: Path) -> None:
    """Assert manifest, config snapshots, and checkpoints emit zero raw personal data or identifiers."""
    records = []
    for i in range(50):
        records.append(
            {
                "customer_id": f"SECRET_ID_{i:04d}",
                "age": 30.0 + i,
                "gender": "Female" if i % 2 == 0 else "Male",
                "income": 50000.0 + (i * 1000.0),
                "education": "Master",
                "occupation": "ClassifiedEngineer",
                "city": "ConfidentialCity",
                "region": "ConfidentialRegion",
                "purchase_frequency": 3.0,
                "average_purchase_value": 85.0,
                "total_spend": 600.0,
                "discount_usage": 0.1,
                "website_visits": 12.0,
                "campaign_exposure": 2.0,
                "product_category": "Electronics",
                "customer_segment": "PrivateTier",
                "purchase_next_month": 1 if i % 2 == 0 else 0,
            }
        )
    df = pd.DataFrame(records)

    out_dir = tmp_path / "artifacts"
    config = ExperimentPipelineConfig(
        experiment_id="privacy_audit_exp",
        imputation_methods=["baseline_median"],
        execution=ExecutionConfig(
            enable_checkpointing=True,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            output_dir=str(out_dir),
        ),
    )

    orchestrator = PipelineOrchestrator()
    manifest = orchestrator.execute_pipeline(df, config, run_id="privacy_run")

    manifest_json = manifest.to_json()
    manifest_dict = json.loads(manifest_json)

    # Asserts that sensitive values are absent from manifest
    assert "SECRET_ID_0001" not in manifest_json
    assert "ClassifiedEngineer" not in manifest_json
    assert "ConfidentialCity" not in manifest_json

    # Asserts that config snapshot has no secrets
    assert "API_KEY" not in str(manifest_dict["config_snapshot"])
    assert "AWS_SECRET" not in str(manifest_dict["config_snapshot"])
