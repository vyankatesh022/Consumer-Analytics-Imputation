"""Integration tests connecting Ingestion Layer directly with Data Quality Layer."""

from pathlib import Path

import pandas as pd

from missing_data_platform.ingestion.engine import IngestionEngine
from missing_data_platform.quality.engine import DataQualityEngine
from missing_data_platform.quality.rules import QualityStatus


def test_ingestion_to_quality_audit_pipeline(sample_csv_path: Path) -> None:
    """Verify seamless handoff from IngestionEngine to DataQualityEngine."""
    ingestion_engine = IngestionEngine()
    ingest_result = ingestion_engine.ingest_file(sample_csv_path, dataset_id="pipeline_dataset")

    assert ingest_result.is_clean is True

    # Pass valid data to Data Quality Layer
    quality_engine = DataQualityEngine()
    report = quality_engine.audit_dataset(
        ingest_result.valid_data, dataset_id="pipeline_quality_report"
    )

    assert report.dataset_id == "pipeline_quality_report"
    assert report.total_records == 10
    assert report.total_columns == 17
    assert report.overall_status in (QualityStatus.PASS, QualityStatus.WARN)
    assert report.failed_checks == 0

    # Ensure missingness is faithfully reported but NOT imputed
    assert report.missingness_summary["age"].missing_count == 1
    assert report.missingness_summary["income"].missing_count == 1
    assert pd.isna(ingest_result.valid_data.loc[1, "age"])
    assert pd.isna(ingest_result.valid_data.loc[2, "income"])
