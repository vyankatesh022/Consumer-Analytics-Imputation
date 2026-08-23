"""Unit tests for GroupDefinitionConfig and MissingGroupPolicy."""

import pytest

from missing_data_platform.bias.config import (
    GroupDefinitionConfig,
    MissingGroupPolicy,
)
from missing_data_platform.exceptions import ConfigurationError


def test_valid_group_definition_config() -> None:
    """Verify valid group configuration creation."""
    config = GroupDefinitionConfig(
        group_column="region",
        minimum_group_size=10,
        missing_group_policy=MissingGroupPolicy.EXCLUDE,
        metrics=["mae", "rmse"],
    )
    assert config.group_column == "region"
    assert config.minimum_group_size == 10
    assert config.missing_group_policy == MissingGroupPolicy.EXCLUDE
    assert config.metrics == ["mae", "rmse"]


def test_invalid_group_definition_invariants() -> None:
    """Verify invalid group configuration raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        GroupDefinitionConfig(group_column="")

    with pytest.raises(ConfigurationError):
        GroupDefinitionConfig(minimum_group_size=0)

    with pytest.raises(ConfigurationError):
        GroupDefinitionConfig(metrics=[])
