"""Integration tests for end-to-end data ingestion pipeline and Parquet generation."""

import tempfile
from pathlib import Path

import pandas as pd

from missing_data_platform.ingestion.engine import IngestionEngine


def test_full_file_ingestion_pipeline(sample_csv_path: Path) -> None:
    """Verify complete ingestion from raw CSV fixture to validated DataFrame and Parquet."""
    engine = IngestionEngine()

    result = engine.ingest_file(sample_csv_path, dataset_id="integration_test_dataset")

    # Verify results
    assert result.is_clean is True
    assert len(result.valid_data) == 10
    assert len(result.quarantined_data) == 0
    assert result.lineage.total_records == 10
    assert result.lineage.valid_records == 10
    assert result.lineage.source_hash_sha256 is not None

    # Assert missingness was strictly preserved
    assert pd.isna(result.valid_data.loc[1, "age"])
    assert pd.isna(result.valid_data.loc[2, "income"])

    # Test Parquet Export
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "bronze_consumer.parquet"
        result.save_bronze_parquet(parquet_path)

        assert parquet_path.exists()
        reloaded_df = pd.read_parquet(parquet_path)
        assert len(reloaded_df) == 10
        assert pd.isna(reloaded_df.loc[1, "age"])
        assert pd.isna(reloaded_df.loc[2, "income"])


def test_in_memory_ingestion_with_quarantine() -> None:
    """Verify ingestion pipeline handling and isolating malformed records."""
    raw_csv = (
        "customer_id,age,gender,income,education,occupation,city,region,purchase_frequency,"
        "average_purchase_value,total_spend,discount_usage,website_visits,campaign_exposure,"
        "product_category,customer_segment,purchase_next_month\n"
        "C01,25,F,50000,B,Eng,LA,West,2.0,50,100,0.1,5,1,Elec,Gold,1\n"
        ",30,M,60000,M,Doc,NY,East,1.0,40,40,0.2,4,2,App,Silver,0\n"  # Missing ID
        "C03,invalid_age,F,70000,B,Eng,CHI,Midwest,3.0,60,180,0.3,6,3,Home,Bronze,1\n"  # Bad numeric
    )
    engine = IngestionEngine()
    result = engine.ingest_content(raw_csv, dataset_id="test_quarantine_run")

    assert result.is_clean is False
    assert len(result.valid_data) == 1
    assert len(result.quarantined_data) == 2
    assert result.lineage.total_records == 3
    assert result.lineage.valid_records == 1
    assert result.lineage.quarantined_records == 2

    # Verify saving valid and quarantine parquet
    with tempfile.TemporaryDirectory() as tmpdir:
        valid_pq = Path(tmpdir) / "valid.parquet"
        quarantine_pq = Path(tmpdir) / "quarantine.parquet"
        result.save_bronze_parquet(valid_pq, quarantine_pq)

        assert valid_pq.exists()
        assert quarantine_pq.exists()
        q_df = pd.read_parquet(quarantine_pq)
        assert len(q_df) == 2
        assert "_quarantine_reason" in q_df.columns
