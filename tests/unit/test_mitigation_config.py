"""Unit tests for MitigationConfig, MitigationStrategy, and MitigationDecision."""

import pytest

from missing_data_platform.exceptions import ConfigurationError
from missing_data_platform.mitigation.config import (
    MitigationConfig,
    MitigationDecision,
    MitigationStrategy,
)


def test_valid_mitigation_config() -> None:
    """Verify creation of valid MitigationConfig with custom parameters."""
    config = MitigationConfig(
        enabled=True,
        strategy=MitigationStrategy.SAMPLE_WEIGHTING,
        group_column="region",
        minimum_group_size=8,
        max_sample_weight=4.0,
        max_allowed_accuracy_degradation=0.20,
        target_disparity_reduction=0.15,
        random_seed=123,
    )
    assert config.enabled is True
    assert config.strategy == MitigationStrategy.SAMPLE_WEIGHTING
    assert config.group_column == "region"
    assert config.minimum_group_size == 8
    assert config.max_sample_weight == 4.0
    assert config.max_allowed_accuracy_degradation == 0.20
    assert config.target_disparity_reduction == 0.15
    assert config.random_seed == 123


def test_mitigation_config_invariants() -> None:
    """Verify invalid parameters raise ConfigurationError."""
    with pytest.raises(ConfigurationError):
        MitigationConfig(group_column="")

    with pytest.raises(ConfigurationError):
        MitigationConfig(minimum_group_size=0)

    with pytest.raises(ConfigurationError):
        MitigationConfig(max_sample_weight=0.5)

    with pytest.raises(ConfigurationError):
        MitigationConfig(max_allowed_accuracy_degradation=-0.1)

    with pytest.raises(ConfigurationError):
        MitigationConfig(target_disparity_reduction=1.5)


def test_mitigation_enums() -> None:
    """Verify mitigation decision and strategy enum values."""
    assert MitigationStrategy.SAMPLE_WEIGHTING == "sample_weighting"
    assert MitigationStrategy.GROUP_SPECIFIC == "group_specific"
    assert MitigationDecision.ACCEPTED == "ACCEPTED"
    assert MitigationDecision.REJECTED == "REJECTED"
    assert MitigationDecision.REQUIRES_REVIEW == "REQUIRES_REVIEW"
