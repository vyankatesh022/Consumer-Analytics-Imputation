"""Unit tests for orchestration configuration, quotas, and sanitized snapshots."""

import pytest

from missing_data_platform.exceptions import ConfigurationError
from missing_data_platform.orchestration.config import (
    ExecutionConfig,
    ExperimentPipelineConfig,
    ResourceLimitsConfig,
)


def test_orchestration_config_defaults() -> None:
    """Assert valid default parameters for ExperimentPipelineConfig."""
    cfg = ExperimentPipelineConfig(experiment_id="test_exp")
    assert cfg.dataset_version == "1.0.0"
    assert cfg.random_seed == 42
    assert len(cfg.imputation_methods) >= 4
    assert cfg.resource_limits.max_records == 1_000_000
    assert cfg.execution.enable_checkpointing is True


def test_orchestration_config_validation_errors() -> None:
    """Assert ConfigurationError on invalid parameters or quota violations."""
    with pytest.raises(ConfigurationError, match="experiment_id cannot be empty"):
        ExperimentPipelineConfig(experiment_id="")

    with pytest.raises(ConfigurationError, match="imputation_methods list cannot be empty"):
        ExperimentPipelineConfig(experiment_id="test_exp", imputation_methods=[])

    with pytest.raises(ConfigurationError, match="exceeding max_methods quota"):
        ExperimentPipelineConfig(
            experiment_id="test_exp",
            imputation_methods=["m1", "m2", "m3"],
            resource_limits=ResourceLimitsConfig(max_methods=2),
        )


def test_resource_limits_validation() -> None:
    """Assert boundary validation on ResourceLimitsConfig."""
    with pytest.raises(ConfigurationError, match="max_records must be >= 1"):
        ResourceLimitsConfig(max_records=0)

    with pytest.raises(ConfigurationError, match="max_retries must be >= 0"):
        ResourceLimitsConfig(max_retries=-1)


def test_execution_config_validation() -> None:
    """Assert validation on ExecutionConfig."""
    with pytest.raises(ConfigurationError, match="checkpoint_dir cannot be empty"):
        ExecutionConfig(checkpoint_dir="")


def test_config_sanitized_snapshot_and_fingerprint() -> None:
    """Assert sanitized snapshot export and deterministic config fingerprinting."""
    cfg = ExperimentPipelineConfig(experiment_id="snap_exp", random_seed=123)
    snap = cfg.to_sanitized_dict()

    assert snap["experiment_id"] == "snap_exp"
    assert snap["random_seed"] == 123
    assert "masking" in snap
    assert "resource_limits" in snap

    fp1 = cfg.get_config_fingerprint()
    fp2 = cfg.get_config_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256 hex length
