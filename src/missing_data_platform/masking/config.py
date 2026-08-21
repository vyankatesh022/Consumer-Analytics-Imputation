"""Configuration and strategy enumerations for artificial missingness simulation.

Defines parameters, random seeds, masking rates, and protected column guards
for benchmarking imputation algorithms against known ground truth.
"""

from dataclasses import dataclass, field
from enum import StrEnum

from missing_data_platform.exceptions import ConfigurationError


class MaskingStrategy(StrEnum):
    """Supported artificial missingness generation strategies."""

    UNIFORM_RANDOM = "uniform_random"  # MCAR-like random dropout on eligible cells
    MAR_COVARIATE = "mar_covariate"  # Missingness conditioned on observed auxiliary feature
    GROUP_STRATIFIED = "group_stratified"  # Equal missingness proportion across demographic cohorts


@dataclass
class MaskingConfig:
    """Master configuration for an artificial missingness masking experiment."""

    experiment_id: str
    mask_rate: float = 0.15
    random_seed: int = 42
    strategy: MaskingStrategy = MaskingStrategy.UNIFORM_RANDOM
    target_features: list[str] | None = None
    protected_features: list[str] = field(
        default_factory=lambda: ["customer_id", "purchase_next_month"]
    )
    conditioning_covariate: str | None = None
    dataset_version: str = "1.0.0"

    def __post_init__(self) -> None:
        """Validate experiment configuration invariants."""
        if not (0.0 <= self.mask_rate <= 1.0):
            raise ConfigurationError(
                f"Invalid mask_rate: {self.mask_rate}. Must be between 0.0 and 1.0.",
                context={"mask_rate": self.mask_rate},
            )

        if not self.experiment_id or not self.experiment_id.strip():
            raise ConfigurationError("experiment_id cannot be empty")

        if self.target_features is not None:
            # Enforce that protected features (target, ID) are never in target_features
            conflicts = [col for col in self.target_features if col in self.protected_features]
            if conflicts:
                raise ConfigurationError(
                    f"Protected features cannot be targeted for artificial masking: {conflicts}",
                    context={"conflicts": conflicts},
                )

        if self.strategy == MaskingStrategy.MAR_COVARIATE and not self.conditioning_covariate:
            raise ConfigurationError(
                "strategy 'mar_covariate' requires specifying 'conditioning_covariate'."
            )
