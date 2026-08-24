"""Comprehensive audit manifest for tracking experiment lineage, stages, and artifacts."""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from missing_data_platform.orchestration.stages import StageStatus


class ArtifactCategory(StrEnum):
    """Classification of generated platform artifacts."""

    CONFIG = "config"
    CHECKPOINT = "checkpoint"
    EVALUATION = "evaluation"
    BIAS = "bias"
    DOWNSTREAM = "downstream"
    MANIFEST = "manifest"
    LOG = "log"


@dataclass
class ArtifactReference:
    """Cryptographically indexed reference to a generated experiment artifact."""

    name: str
    category: ArtifactCategory
    path: str
    sha256_hash: str
    size_bytes: int
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class ExperimentManifest:
    """Immutable audit manifest documenting complete experiment provenance, lineage, and results."""

    experiment_id: str
    run_id: str
    dataset_version: str
    dataset_fingerprint: str
    config_fingerprint: str
    code_version: str
    clean_worktree: bool
    final_status: StageStatus
    stage_statuses: dict[str, str] = field(default_factory=dict)
    stage_durations: dict[str, float] = field(default_factory=dict)
    method_statuses: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifact_references: list[ArtifactReference] = field(default_factory=list)
    seeds: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    partial_failure: bool = False
    start_time_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    end_time_utc: str | None = None
    total_duration_seconds: float = 0.0
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to structured dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "dataset_version": self.dataset_version,
            "dataset_fingerprint": self.dataset_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "code_version": self.code_version,
            "clean_worktree": self.clean_worktree,
            "final_status": self.final_status.value,
            "stage_statuses": self.stage_statuses,
            "stage_durations": self.stage_durations,
            "method_statuses": self.method_statuses,
            "artifact_references": [asdict(a) for a in self.artifact_references],
            "seeds": self.seeds,
            "warnings": self.warnings,
            "partial_failure": self.partial_failure,
            "start_time_utc": self.start_time_utc,
            "end_time_utc": self.end_time_utc,
            "total_duration_seconds": self.total_duration_seconds,
            "config_snapshot": self.config_snapshot,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize manifest to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, target_path: Path | str) -> Path:
        """Persist manifest to specified file path."""
        p = Path(target_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p
