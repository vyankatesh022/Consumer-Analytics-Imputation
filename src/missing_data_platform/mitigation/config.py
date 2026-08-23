"""Configuration schemas and mitigation strategy definitions."""

from dataclasses import dataclass, field
from enum import StrEnum

from missing_data_platform.exceptions import ConfigurationError


class MitigationStrategy(StrEnum):
    """Supported fairness-aware mitigation strategies for imputation."""

    SAMPLE_WEIGHTING = "sample_weighting"  # Balanced inverse-frequency sample weighting
    GROUP_SPECIFIC = "group_specific"  # Group-specific sub-models for distinct cohorts
    GROUP_CONDITIONED = "group_conditioned"  # Conditioning imputation on group indicators


class MitigationDecision(StrEnum):
    """Outcome classification for evaluated mitigation interventions."""

    ACCEPTED = "ACCEPTED"  # Fairness improved within accuracy and coverage constraints
    REJECTED = "REJECTED"  # Violation of accuracy constraint or disparity increased
    REQUIRES_REVIEW = "REQUIRES_REVIEW"  # Tradeoff requires domain expert / human review


@dataclass
class MitigationConfig:
    """Configuration parameters for imputation bias mitigation."""

    enabled: bool = False
    strategy: MitigationStrategy = MitigationStrategy.SAMPLE_WEIGHTING
    group_column: str = "customer_segment"
    minimum_group_size: int = 5
    max_sample_weight: float = 5.0
    max_allowed_accuracy_degradation: float = 0.15  # Max 15% increase in MAE/RMSE
    target_disparity_reduction: float = 0.10  # Min 10% reduction in max disparity
    random_seed: int = 42
    target_features: list[str] | None = None
    protected_features: list[str] = field(
        default_factory=lambda: ["customer_id", "purchase_next_month"]
    )

    def __post_init__(self) -> None:
        """Validate mitigation configuration invariants."""
        if not self.group_column or not self.group_column.strip():
            raise ConfigurationError("group_column cannot be empty.")

        if self.minimum_group_size < 1:
            raise ConfigurationError(
                f"Invalid minimum_group_size: {self.minimum_group_size}. Must be >= 1.",
                context={"minimum_group_size": self.minimum_group_size},
            )

        if self.max_sample_weight < 1.0:
            raise ConfigurationError(
                f"Invalid max_sample_weight: {self.max_sample_weight}. Must be >= 1.0.",
                context={"max_sample_weight": self.max_sample_weight},
            )

        if self.max_allowed_accuracy_degradation < 0.0:
            raise ConfigurationError(
                f"Invalid max_allowed_accuracy_degradation: {self.max_allowed_accuracy_degradation}. Must be >= 0.0.",
                context={"max_allowed_accuracy_degradation": self.max_allowed_accuracy_degradation},
            )

        if self.target_disparity_reduction < 0.0 or self.target_disparity_reduction > 1.0:
            raise ConfigurationError(
                f"Invalid target_disparity_reduction: {self.target_disparity_reduction}. Must be between 0.0 and 1.0.",
                context={"target_disparity_reduction": self.target_disparity_reduction},
            )
