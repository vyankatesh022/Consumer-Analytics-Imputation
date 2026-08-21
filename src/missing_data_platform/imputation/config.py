"""Baseline imputation strategy configuration and parameters.

Defines supported baseline strategies (mean, median, mode, constant) and guards
protecting identifiers and downstream targets from accidental imputation.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from missing_data_platform.exceptions import ConfigurationError


class BaselineStrategy(StrEnum):
    """Supported baseline imputation statistical strategies."""

    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"


@dataclass
class BaselineImputationConfig:
    """Master configuration for baseline statistical imputation."""

    numeric_strategy: BaselineStrategy = BaselineStrategy.MEDIAN
    categorical_strategy: BaselineStrategy = BaselineStrategy.MODE
    target_features: list[str] | None = None
    protected_features: list[str] = field(
        default_factory=lambda: ["customer_id", "purchase_next_month"]
    )
    constant_fill_value: Any = "Missing"

    def __post_init__(self) -> None:
        """Validate baseline configuration invariants."""
        if self.numeric_strategy not in (
            BaselineStrategy.MEAN,
            BaselineStrategy.MEDIAN,
            BaselineStrategy.CONSTANT,
        ):
            raise ConfigurationError(
                f"Invalid numeric_strategy: {self.numeric_strategy}. Must be mean, median, or constant."
            )

        if self.categorical_strategy not in (
            BaselineStrategy.MODE,
            BaselineStrategy.CONSTANT,
        ):
            raise ConfigurationError(
                f"Invalid categorical_strategy: {self.categorical_strategy}. Must be mode or constant."
            )

        if self.target_features is not None:
            conflicts = [col for col in self.target_features if col in self.protected_features]
            if conflicts:
                raise ConfigurationError(
                    f"Protected features cannot be targeted for baseline imputation: {conflicts}",
                    context={"conflicts": conflicts},
                )
