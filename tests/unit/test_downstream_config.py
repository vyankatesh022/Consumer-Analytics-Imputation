"""Unit tests for Downstream ML evaluation configuration models."""

import pytest

from missing_data_platform.downstream.config import (
    DownstreamBenchmarkConfig,
    DownstreamConfig,
    DownstreamModelType,
    DownstreamTaskType,
)
from missing_data_platform.exceptions import ConfigurationError


def test_downstream_config_defaults() -> None:
    """Assert valid default parameters for DownstreamConfig."""
    cfg = DownstreamConfig()
    assert cfg.model_type == DownstreamModelType.RANDOM_FOREST
    assert cfg.task_type == DownstreamTaskType.CLASSIFICATION
    assert cfg.primary_metric == "f1"
    assert cfg.test_size == 0.20
    assert cfg.random_seed == 42
    assert "customer_id" in cfg.protected_features
    assert "purchase_next_month" in cfg.protected_features


def test_downstream_config_invalid_test_size() -> None:
    """Assert ConfigurationError when test_size is outside (0, 1)."""
    with pytest.raises(ConfigurationError, match="Invalid test_size"):
        DownstreamConfig(test_size=0.0)

    with pytest.raises(ConfigurationError, match="Invalid test_size"):
        DownstreamConfig(test_size=1.5)


def test_downstream_config_invalid_group_size() -> None:
    """Assert ConfigurationError when minimum_group_size < 1."""
    with pytest.raises(ConfigurationError, match="Invalid minimum_group_size"):
        DownstreamConfig(minimum_group_size=0)


def test_downstream_config_invalid_primary_metric() -> None:
    """Assert ConfigurationError when primary_metric is incompatible with task type."""
    with pytest.raises(ConfigurationError, match="Invalid classification primary_metric"):
        DownstreamConfig(
            task_type=DownstreamTaskType.CLASSIFICATION,
            primary_metric="rmse",
        )

    with pytest.raises(ConfigurationError, match="Invalid regression primary_metric"):
        DownstreamConfig(
            task_type=DownstreamTaskType.REGRESSION,
            primary_metric="f1",
        )


def test_downstream_benchmark_config_validation() -> None:
    """Assert validation on benchmark sweep settings."""
    cfg = DownstreamBenchmarkConfig()
    assert len(cfg.methods) >= 4
    assert len(cfg.missingness_rates) >= 3
    assert len(cfg.repeated_seeds) >= 3

    with pytest.raises(ConfigurationError, match="methods list cannot be empty"):
        DownstreamBenchmarkConfig(methods=[])

    with pytest.raises(ConfigurationError, match="Invalid missingness_rate"):
        DownstreamBenchmarkConfig(missingness_rates=[0.2, 1.5])

    with pytest.raises(ConfigurationError, match="repeated_seeds cannot be empty"):
        DownstreamBenchmarkConfig(repeated_seeds=[])
