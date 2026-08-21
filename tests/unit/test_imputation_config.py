"""Unit tests for BaselineImputationConfig and BaselineStrategy."""

import pytest

from missing_data_platform.exceptions import ConfigurationError
from missing_data_platform.imputation.config import (
    BaselineImputationConfig,
    BaselineStrategy,
)


def test_baseline_strategy_enum() -> None:
    """Verify BaselineStrategy enumeration values."""
    assert BaselineStrategy.MEAN == "mean"
    assert BaselineStrategy.MEDIAN == "median"
    assert BaselineStrategy.MODE == "mode"
    assert BaselineStrategy.CONSTANT == "constant"


def test_valid_baseline_imputation_config() -> None:
    """Verify default and customized valid BaselineImputationConfig."""
    config = BaselineImputationConfig(
        numeric_strategy=BaselineStrategy.MEAN,
        categorical_strategy=BaselineStrategy.MODE,
    )
    assert config.numeric_strategy == BaselineStrategy.MEAN
    assert config.categorical_strategy == BaselineStrategy.MODE
    assert "customer_id" in config.protected_features
    assert "purchase_next_month" in config.protected_features


def test_targeting_protected_column_raises_error() -> None:
    """Verify that targeting customer_id or purchase_next_month raises ConfigurationError."""
    with pytest.raises(ConfigurationError) as exc_info:
        BaselineImputationConfig(
            target_features=["income", "customer_id"],
        )
    assert "Protected features" in str(exc_info.value)


def test_invalid_numeric_strategy_raises_error() -> None:
    """Verify that unsupported numeric_strategy raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        BaselineImputationConfig(
            numeric_strategy=BaselineStrategy.MODE
        )  # Mode is not allowed for numeric


def test_invalid_categorical_strategy_raises_error() -> None:
    """Verify that unsupported categorical_strategy raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        BaselineImputationConfig(
            categorical_strategy=BaselineStrategy.MEAN
        )  # Mean is not allowed for categorical
