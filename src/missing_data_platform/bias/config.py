"""Configuration and policy schemas for group bias and representation analysis."""

from dataclasses import dataclass, field
from enum import StrEnum

from missing_data_platform.exceptions import ConfigurationError


class MissingGroupPolicy(StrEnum):
    """Handling strategy for rows where the group attribute itself is missing."""

    UNKNOWN = "unknown"  # Label as 'Unknown' and include as an explicit group
    EXCLUDE = "exclude"  # Exclude rows with missing group labels from group-level analysis


@dataclass
class GroupDefinitionConfig:
    """Configuration parameters for population group fairness and representation analysis."""

    group_column: str = "customer_segment"
    minimum_group_size: int = 5
    missing_group_policy: MissingGroupPolicy = MissingGroupPolicy.UNKNOWN
    metrics: list[str] = field(default_factory=lambda: ["mae", "rmse"])
    target_features: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate group analysis configuration invariants."""
        if not self.group_column or not self.group_column.strip():
            raise ConfigurationError("group_column cannot be empty.")

        if self.minimum_group_size < 1:
            raise ConfigurationError(
                f"Invalid minimum_group_size: {self.minimum_group_size}. Must be >= 1.",
                context={"minimum_group_size": self.minimum_group_size},
            )

        if not self.metrics:
            raise ConfigurationError("Must specify at least one metric for bias analysis.")
