"""Unit tests for checkpoint persistence, cryptographic verification, and corruption detection."""

from pathlib import Path

from missing_data_platform.orchestration.checkpoints import CheckpointManager
from missing_data_platform.orchestration.stages import PipelineStage


def test_checkpoint_save_load_verify(tmp_path: Path) -> None:
    """Assert successful checkpoint save, reload, and cryptographic verification."""
    manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")

    payload = {"status": "ok", "imputed_count": 42}
    saved_path = manager.save_checkpoint(
        stage=PipelineStage.MASKING,
        experiment_id="exp_01",
        run_id="run_01",
        dataset_fingerprint="ds_fp_12345",
        config_fingerprint="cfg_fp_67890",
        payload=payload,
    )

    assert saved_path.exists()

    loaded = manager.load_checkpoint(
        stage=PipelineStage.MASKING,
        experiment_id="exp_01",
        run_id="run_01",
    )
    assert loaded is not None
    assert loaded.metadata.stage == PipelineStage.MASKING.value
    assert loaded.payload["imputed_count"] == 42

    valid, reason = manager.verify_integrity(
        loaded,
        expected_dataset_fp="ds_fp_12345",
        expected_config_fp="cfg_fp_67890",
    )
    assert valid is True
    assert "verified" in reason


def test_checkpoint_corruption_detection(tmp_path: Path) -> None:
    """Assert corruption is detected when checkpoint payload is modified on disk."""
    manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")

    saved_path = manager.save_checkpoint(
        stage=PipelineStage.IMPUTATION,
        experiment_id="exp_corrupt",
        run_id="run_01",
        dataset_fingerprint="ds_fp_1",
        config_fingerprint="cfg_fp_1",
        payload={"result": "valid_result"},
    )

    # Tamper with file content directly
    content = saved_path.read_text(encoding="utf-8")
    tampered_content = content.replace("valid_result", "tampered_result")
    saved_path.write_text(tampered_content, encoding="utf-8")

    loaded = manager.load_checkpoint(
        stage=PipelineStage.IMPUTATION,
        experiment_id="exp_corrupt",
        run_id="run_01",
    )
    assert loaded is not None

    valid, reason = manager.verify_integrity(
        loaded,
        expected_dataset_fp="ds_fp_1",
        expected_config_fp="cfg_fp_1",
    )
    assert valid is False
    assert "data corruption detected" in reason


def test_checkpoint_lineage_mismatch_rejection(tmp_path: Path) -> None:
    """Assert checkpoint verification fails when dataset or configuration fingerprint differs."""
    manager = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")

    manager.save_checkpoint(
        stage=PipelineStage.MASKING,
        experiment_id="exp_mismatch",
        run_id="run_01",
        dataset_fingerprint="ds_fp_original",
        config_fingerprint="cfg_fp_original",
        payload={"masked": True},
    )

    loaded = manager.load_checkpoint(
        stage=PipelineStage.MASKING,
        experiment_id="exp_mismatch",
        run_id="run_01",
    )
    assert loaded is not None

    # Mismatched dataset
    valid_ds, reason_ds = manager.verify_integrity(
        loaded,
        expected_dataset_fp="ds_fp_different",
        expected_config_fp="cfg_fp_original",
    )
    assert valid_ds is False
    assert "Dataset fingerprint mismatch" in reason_ds

    # Mismatched config
    valid_cfg, reason_cfg = manager.verify_integrity(
        loaded,
        expected_dataset_fp="ds_fp_original",
        expected_config_fp="cfg_fp_different",
    )
    assert valid_cfg is False
    assert "Configuration fingerprint mismatch" in reason_cfg
