"""Unit tests for MaskingConfig and MaskingStrategy."""

import pytest

from missing_data_platform.exceptions import ConfigurationError
from missing_data_platform.masking.config import MaskingConfig, MaskingStrategy


def test_valid_masking_config() -> None:
    """Verify valid masking configuration creation."""
    config = MaskingConfig(
        experiment_id="exp_test_01",
        mask_rate=0.20,
        random_seed=123,
        strategy=MaskingStrategy.UNIFORM_RANDOM,
    )
    assert config.experiment_id == "exp_test_01"
    assert config.mask_rate == 0.20
    assert config.random_seed == 123
    assert config.strategy == MaskingStrategy.UNIFORM_RANDOM


def test_invalid_mask_rate_raises_error() -> None:
    """Verify that mask_rate outside [0.0, 1.0] raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        MaskingConfig(experiment_id="exp_bad_rate", mask_rate=1.5)

    with pytest.raises(ConfigurationError):
        MaskingConfig(experiment_id="exp_neg_rate", mask_rate=-0.1)


def test_empty_experiment_id_raises_error() -> None:
    """Verify that empty experiment ID raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        MaskingConfig(experiment_id="   ")


def test_targeting_protected_columns_raises_error() -> None:
    """Verify that targeting protected columns (customer_id or target) raises ConfigurationError."""
    with pytest.raises(ConfigurationError) as exc_info:
        MaskingConfig(
            experiment_id="exp_target_protected",
            target_features=["income", "purchase_next_month"],
        )
    assert "Protected features" in str(exc_info.value)


def test_mar_strategy_without_covariate_raises_error() -> None:
    """Verify that MAR covariate strategy without specifying covariate raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        MaskingConfig(
            experiment_id="exp_mar_missing_cov",
            strategy=MaskingStrategy.MAR_COVARIATE,
            conditioning_covariate=None,
        )
