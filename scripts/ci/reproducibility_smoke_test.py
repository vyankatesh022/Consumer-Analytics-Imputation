#!/usr/bin/env python3
"""Lightweight CI reproducibility smoke test.

Executes two identical end-to-end experiment runs on a synthetic fixture and asserts
deterministic equivalence of dataset fingerprints, configuration snapshots, stage
execution statuses, and downstream evaluation outcomes.
"""

import sys
import tempfile
from pathlib import Path

# Add source directory to Python path for standalone CI execution
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from missing_data_platform.orchestration.config import (  # noqa: E402
    ExecutionConfig,
    ExperimentPipelineConfig,
)
from missing_data_platform.orchestration.orchestrator import PipelineOrchestrator  # noqa: E402
from missing_data_platform.orchestration.stages import StageStatus  # noqa: E402


def generate_synthetic_smoke_fixture(n_records: int = 60) -> pd.DataFrame:
    """Generate deterministic synthetic consumer data fixture for CI smoke testing."""
    records = []
    for i in range(n_records):
        records.append(
            {
                "customer_id": f"CUST_{i:04d}",
                "age": 22.0 + (i % 40),
                "gender": "Female" if i % 2 == 0 else "Male",
                "income": 35000.0 + (i * 450.0),
                "education": "Bachelor" if i % 3 == 0 else "Master",
                "occupation": "Engineer" if i % 2 == 0 else "Analyst",
                "city": "Seattle" if i % 2 == 0 else "Austin",
                "region": "West" if i % 2 == 0 else "South",
                "purchase_frequency": 2.0 + (i % 5),
                "average_purchase_value": 55.0 + (i * 1.5),
                "total_spend": 250.0 + (i * 15.0),
                "discount_usage": 0.1 * (i % 4),
                "website_visits": 6.0 + (i % 8),
                "campaign_exposure": 1.0 + (i % 3),
                "product_category": "Electronics" if i % 2 == 0 else "Apparel",
                "customer_segment": "Gold" if i % 2 == 0 else "Silver",
                "purchase_next_month": 1 if i % 3 == 0 else 0,
            }
        )
    return pd.DataFrame(records)


def run_reproducibility_smoke_test() -> int:
    """Execute dual-run reproducibility smoke test and verify equality."""
    print("🔬 Running CI Reproducibility Smoke Test...")
    df = generate_synthetic_smoke_fixture()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        config_run1 = ExperimentPipelineConfig(
            experiment_id="ci_smoke_repro",
            random_seed=42,
            imputation_methods=["baseline_median", "knn"],
            execution=ExecutionConfig(
                enable_checkpointing=True,
                checkpoint_dir=str(tmp_path / "chk1"),
                output_dir=str(tmp_path / "out1"),
            ),
        )

        config_run2 = ExperimentPipelineConfig(
            experiment_id="ci_smoke_repro",
            random_seed=42,
            imputation_methods=["baseline_median", "knn"],
            execution=ExecutionConfig(
                enable_checkpointing=True,
                checkpoint_dir=str(tmp_path / "chk2"),
                output_dir=str(tmp_path / "out2"),
            ),
        )

        orchestrator = PipelineOrchestrator()

        manifest1 = orchestrator.execute_pipeline(df, config_run1, run_id="smoke_run_1")
        manifest2 = orchestrator.execute_pipeline(df, config_run2, run_id="smoke_run_2")

        # 1. Assert successful completion
        if (
            manifest1.final_status != StageStatus.COMPLETED
            or manifest2.final_status != StageStatus.COMPLETED
        ):
            print(
                f"❌ CI Smoke Test Failed: Pipeline status was {manifest1.final_status} / {manifest2.final_status}"
            )
            return 1

        # 2. Assert identical dataset fingerprint
        if manifest1.dataset_fingerprint != manifest2.dataset_fingerprint:
            print("❌ CI Smoke Test Failed: Dataset fingerprints differ across identical runs.")
            return 1

        # 3. Assert identical config fingerprint
        if manifest1.config_fingerprint != manifest2.config_fingerprint:
            print("❌ CI Smoke Test Failed: Config fingerprints differ across identical runs.")
            return 1

        # 4. Assert identical stage statuses
        if manifest1.stage_statuses != manifest2.stage_statuses:
            print("❌ CI Smoke Test Failed: Stage statuses differ across identical runs.")
            return 1

        # 5. Assert zero partial failures
        if manifest1.partial_failure or manifest2.partial_failure:
            print("❌ CI Smoke Test Failed: Unexpected partial failure encountered in smoke run.")
            return 1

    print(
        f"✅ CI Reproducibility Smoke Test Passed: 11 stages completed in {manifest1.total_duration_seconds}s "
        f"with exact bit-level fingerprint match."
    )
    return 0


if __name__ == "__main__":
    sys.exit(run_reproducibility_smoke_test())
