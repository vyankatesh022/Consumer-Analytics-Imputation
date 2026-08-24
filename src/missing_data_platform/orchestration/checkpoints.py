"""Cryptographic checkpoint management, integrity verification, and safe resumption."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from missing_data_platform.__version__ import __version__ as PLATFORM_VERSION
from missing_data_platform.exceptions import StorageError
from missing_data_platform.logging import get_logger
from missing_data_platform.orchestration.stages import PipelineStage

logger = get_logger("orchestration.checkpoints")


@dataclass
class CheckpointMetadata:
    """Cryptographic provenance and validation headers for a saved stage checkpoint."""

    experiment_id: str
    run_id: str
    stage: str
    dataset_fingerprint: str
    config_fingerprint: str
    schema_version: str
    project_version: str
    payload_hash: str
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class StageCheckpoint:
    """Self-contained, verifiable stage checkpoint holding metadata and intermediate payload."""

    metadata: CheckpointMetadata
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint as structured dictionary."""
        return {
            "metadata": asdict(self.metadata),
            "payload": self.payload,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize checkpoint to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class CheckpointManager:
    """Manages disk serialization, cryptographic integrity checking, and safe resumption of stage checkpoints."""

    def __init__(self, checkpoint_dir: Path | str = "./artifacts/checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(
        self, stage: PipelineStage, experiment_id: str, run_id: str | None = None
    ) -> Path:
        """Derive deterministic filename for a stage checkpoint."""
        prefix = f"{experiment_id}_{run_id}" if run_id else experiment_id
        return self.checkpoint_dir / f"{prefix}_{stage.value}.chk.json"

    def save_checkpoint(
        self,
        stage: PipelineStage,
        experiment_id: str,
        run_id: str,
        dataset_fingerprint: str,
        config_fingerprint: str,
        payload: dict[str, Any],
        schema_version: str = "1.0.0",
    ) -> Path:
        """Persist a verifiable stage checkpoint to disk.

        Raises:
            StorageError: If disk I/O fails.
        """
        payload_str = json.dumps(payload, sort_keys=True, default=str)
        payload_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

        metadata = CheckpointMetadata(
            experiment_id=experiment_id,
            run_id=run_id,
            stage=stage.value,
            dataset_fingerprint=dataset_fingerprint,
            config_fingerprint=config_fingerprint,
            schema_version=schema_version,
            project_version=PLATFORM_VERSION,
            payload_hash=payload_hash,
        )

        checkpoint = StageCheckpoint(metadata=metadata, payload=payload)
        chk_path = self._get_checkpoint_path(stage, experiment_id, run_id)

        try:
            chk_path.write_text(checkpoint.to_json(), encoding="utf-8")
            logger.info(
                "Checkpoint saved successfully",
                stage=stage.value,
                experiment_id=experiment_id,
                run_id=run_id,
                path=str(chk_path),
            )
        except Exception as e:
            raise StorageError(
                f"Failed to write stage checkpoint '{stage.value}' to disk: {e}",
                context={"path": str(chk_path)},
            ) from e

        return chk_path

    def load_checkpoint(
        self,
        stage: PipelineStage,
        experiment_id: str,
        run_id: str | None = None,
    ) -> StageCheckpoint | None:
        """Load stage checkpoint from disk if present."""
        chk_path = self._get_checkpoint_path(stage, experiment_id, run_id)
        if not chk_path.exists():
            return None

        try:
            data = json.loads(chk_path.read_text(encoding="utf-8"))
            meta_dict = data.get("metadata", {})
            metadata = CheckpointMetadata(**meta_dict)
            payload = data.get("payload", {})
            return StageCheckpoint(metadata=metadata, payload=payload)
        except Exception as e:
            logger.warning(
                "Corrupted checkpoint detected and rejected",
                stage=stage.value,
                path=str(chk_path),
                error=str(e),
            )
            return None

    def verify_integrity(
        self,
        checkpoint: StageCheckpoint,
        expected_dataset_fp: str,
        expected_config_fp: str,
    ) -> tuple[bool, str]:
        """Verify cryptographic checksums, dataset fingerprint, and configuration compatibility.

        Returns:
            Tuple of (is_valid, reason_message).
        """
        # 1. Verify payload checksum against hash header
        payload_str = json.dumps(checkpoint.payload, sort_keys=True, default=str)
        actual_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

        if actual_hash != checkpoint.metadata.payload_hash:
            return False, "Checkpoint payload hash mismatch (data corruption detected)."

        # 2. Verify dataset fingerprint match
        if checkpoint.metadata.dataset_fingerprint != expected_dataset_fp:
            return False, (
                f"Dataset fingerprint mismatch: Checkpoint had '{checkpoint.metadata.dataset_fingerprint}', "
                f"current dataset has '{expected_dataset_fp}'."
            )

        # 3. Verify configuration fingerprint match
        if checkpoint.metadata.config_fingerprint != expected_config_fp:
            return False, (
                f"Configuration fingerprint mismatch: Checkpoint had '{checkpoint.metadata.config_fingerprint}', "
                f"current configuration has '{expected_config_fp}'."
            )

        return True, "Checkpoint integrity and lineage verified."
