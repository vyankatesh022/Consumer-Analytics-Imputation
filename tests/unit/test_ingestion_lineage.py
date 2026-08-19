"""Unit tests for IngestionLineage metadata tracking."""

import tempfile
from pathlib import Path

from missing_data_platform.ingestion.lineage import IngestionLineage


def test_lineage_creation_and_serialization() -> None:
    """Verify that IngestionLineage records and serializes operational metadata."""
    lineage = IngestionLineage(
        dataset_id="ds_test_001",
        source_identifier="test_consumer.csv",
        schema_version="1.0.0",
        total_records=100,
        valid_records=95,
        quarantined_records=5,
        missingness_distribution={"age": 10, "income": 15},
        execution_time_ms=12.45,
        source_hash_sha256="fake_sha256_hash",
    )

    data = lineage.to_dict()
    assert data["dataset_id"] == "ds_test_001"
    assert data["total_records"] == 100
    assert data["valid_records"] == 95
    assert data["quarantined_records"] == 5
    assert data["missingness_distribution"]["age"] == 10

    json_str = lineage.to_json()
    assert "ds_test_001" in json_str
    assert "fake_sha256_hash" in json_str


def test_lineage_file_sha256_computation() -> None:
    """Verify computation of file cryptographic SHA-256 digests."""
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write("deterministic_test_content_123")
        tmp_path = Path(tmp.name)

    try:
        sha256 = IngestionLineage.compute_file_sha256(tmp_path)
        assert len(sha256) == 64
        # Deterministic check: sha256 of same content matches
        sha256_repeat = IngestionLineage.compute_file_sha256(tmp_path)
        assert sha256 == sha256_repeat
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
