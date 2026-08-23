"""End-to-End Integration tests connecting Ingestion -> Quality -> Missingness -> Masking -> Imputers -> Evaluation -> Bias Analysis -> Mitigation."""

from pathlib import Path

from missing_data_platform.ingestion.engine import IngestionEngine
from missing_data_platform.masking.config import MaskingConfig, MaskingStrategy
from missing_data_platform.missingness.engine import MissingnessAnalysisEngine
from missing_data_platform.mitigation.config import (
    MitigationConfig,
    MitigationDecision,
    MitigationStrategy,
)
from missing_data_platform.mitigation.engine import FairnessMitigationEngine
from missing_data_platform.quality.engine import DataQualityEngine


def test_full_pipeline_to_mitigation_decision(sample_csv_path: Path) -> None:
    """Verify end-to-end flow from raw CSV ingestion through to bias mitigation decision."""
    # 1. Ingestion
    ingestion = IngestionEngine()
    ingest_result = ingestion.ingest_file(sample_csv_path, dataset_id="mit_pipeline_ingest")
    assert ingest_result.is_clean is True

    # 2. Quality Audit
    quality = DataQualityEngine()
    q_report = quality.audit_dataset(ingest_result.valid_data, dataset_id="mit_pipeline_quality")
    assert q_report.failed_checks == 0

    # 3. Missingness Diagnostics
    missingness = MissingnessAnalysisEngine()
    m_report = missingness.analyze(ingest_result.valid_data, dataset_id="mit_pipeline_missingness")
    assert m_report.total_records == 10

    # 4. Masking & Mitigation Pipeline
    mask_config = MaskingConfig(
        experiment_id="mit_pipeline_bench",
        mask_rate=0.20,
        random_seed=42,
        strategy=MaskingStrategy.UNIFORM_RANDOM,
        target_features=["average_purchase_value", "total_spend"],
    )

    mit_config = MitigationConfig(
        enabled=True,
        strategy=MitigationStrategy.SAMPLE_WEIGHTING,
        group_column="customer_segment",
        max_allowed_accuracy_degradation=0.50,
        target_disparity_reduction=0.05,
    )
    mit_engine = FairnessMitigationEngine(config=mit_config)

    mit_result = mit_engine.mitigate_and_evaluate(
        df=ingest_result.valid_data,
        mask_config=mask_config,
        method="random_forest",
    )

    assert mit_result.experiment_id == "mit_pipeline_bench"
    assert mit_result.decision in (
        MitigationDecision.ACCEPTED,
        MitigationDecision.REQUIRES_REVIEW,
        MitigationDecision.REJECTED,
    )
    assert len(mit_result.decision_reason) > 0
    assert len(mit_result.group_results_before) > 0
    assert len(mit_result.group_results_after) > 0
