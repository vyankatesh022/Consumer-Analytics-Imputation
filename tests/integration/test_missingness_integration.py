"""Integration tests connecting Ingestion -> Data Quality -> Missingness Analysis."""

from pathlib import Path

import pandas as pd

from missing_data_platform.ingestion.engine import IngestionEngine
from missing_data_platform.missingness.engine import MissingnessAnalysisEngine
from missing_data_platform.quality.engine import DataQualityEngine


def test_end_to_end_missingness_analysis_flow(sample_csv_path: Path) -> None:
    """Verify data flow from raw CSV ingestion to data quality validation and missingness diagnostics."""
    # 1. Ingestion
    ingestion_engine = IngestionEngine()
    ingest_result = ingestion_engine.ingest_file(sample_csv_path, dataset_id="e2e_ingest")
    assert ingest_result.is_clean is True

    # 2. Data Quality Audit
    quality_engine = DataQualityEngine()
    quality_report = quality_engine.audit_dataset(
        ingest_result.valid_data, dataset_id="e2e_quality"
    )
    assert quality_report.failed_checks == 0

    # 3. Missingness Mechanism Diagnostics
    missingness_engine = MissingnessAnalysisEngine()
    missingness_report = missingness_engine.analyze(
        ingest_result.valid_data, dataset_id="e2e_missingness"
    )

    assert missingness_report.total_records == 10
    assert missingness_report.features_with_missingness_count > 0
    assert missingness_report.mcar_diagnostics is not None
    assert missingness_report.mar_diagnostics is not None

    # Assert that missingness was NOT imputed at any stage
    assert pd.isna(ingest_result.valid_data.loc[1, "age"])
    assert pd.isna(ingest_result.valid_data.loc[2, "income"])
