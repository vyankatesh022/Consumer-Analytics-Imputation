"""Ingestion lineage and metadata tracking.

Captures provenance, dataset hashing, timestamps, schema version,
record counts, and missingness metrics for every ingestion execution.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class IngestionLineage:
    """Provenance and execution metadata for a data ingestion run."""

    dataset_id: str
    source_identifier: str
    schema_version: str
    total_records: int
    valid_records: int
    quarantined_records: int
    missingness_distribution: dict[str, int]
    execution_time_ms: float
    source_hash_sha256: str | None = None
    ingested_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @classmethod
    def compute_file_sha256(cls, file_path: Path | str) -> str:
        """Compute cryptographic SHA-256 digest of input file for lineage audit."""
        path = Path(file_path)
        if not path.exists():
            return "file-not-found"

        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert lineage metadata to dictionary format."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Export lineage metadata as serialized JSON."""
        return json.dumps(self.to_dict(), indent=indent)
