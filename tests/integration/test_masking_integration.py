"""Integration tests connecting Ingestion -> Quality -> Missingness -> Masking Benchmark Dataset."""

from pathlib import Path

import pandas as pd

from missing_data_platform.ingestion.engine import IngestionEngine
from missing_data_platform.masking.config import MaskingConfig, MaskingStrategy
from missing_data_platform.masking.engine import MaskingEngine
from missing_data_platform.missingness.engine import MissingnessAnalysisEngine
from missing_data_platform.quality.engine import DataQualityEngine


def test_full_pipeline_to_benchmark_dataset(sample_csv_path: Path) -> None:
    """Verify entire pipeline flow leading to an artificial missingness benchmark dataset."""
    # 1. Ingestion
    ingestion = IngestionEngine()
    ingest_result = ingestion.ingest_file(sample_csv_path, dataset_id="pipeline_ingest")
    assert ingest_result.is_clean is True

    # 2. Quality Audit
    quality = DataQualityEngine()
    q_report = quality.audit_dataset(ingest_result.valid_data, dataset_id="pipeline_quality")
    assert q_report.failed_checks == 0

    # 3. Missingness Analysis
    missingness = MissingnessAnalysisEngine()
    m_report = missingness.analyze(ingest_result.valid_data, dataset_id="pipeline_missingness")
    assert m_report.total_records == 10

    # 4. Artificial Missingness Masking for Benchmarking
    masking = MaskingEngine()
    mask_config = MaskingConfig(
        experiment_id="exp_pipeline_benchmark",
        mask_rate=0.20,
        random_seed=42,
        strategy=MaskingStrategy.UNIFORM_RANDOM,
        target_features=["average_purchase_value", "total_spend"],
    )
    bench_result = masking.generate_benchmark_dataset(ingest_result.valid_data, mask_config)

    assert bench_result.total_records == 10
    assert bench_result.total_artificially_masked_cells > 0
    assert (
        bench_result.ground_truth_store.total_masked_cells
        == bench_result.total_artificially_masked_cells
    )

    # Ensure natural missingness in 'age' and 'income' is still present
    assert pd.isna(bench_result.masked_dataset.loc[1, "age"])
    assert pd.isna(bench_result.masked_dataset.loc[2, "income"])

    # Ensure protected columns are unmasked
    assert not bench_result.masked_dataset["customer_id"].isna().any()
    assert not bench_result.masked_dataset["purchase_next_month"].isna().any()
