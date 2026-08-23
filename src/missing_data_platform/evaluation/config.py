"""Evaluation configuration and parameter definitions.

Defines supported evaluation metrics, aggregation schemes, and method definitions.
"""

from dataclasses import dataclass, field
from enum import StrEnum

from missing_data_platform.exceptions import ConfigurationError


class AggregationStrategy(StrEnum):
    """Supported aggregation schemes across evaluated features."""

    FEATURE_WEIGHTED = "feature_weighted"  # Weighted by number of evaluated cells per feature
    MACRO_AVERAGE = "macro_average"  # Simple unweighted mean across features
    CELL_WEIGHTED = "cell_weighted"  # Global metric across all flattened evaluated cells


@dataclass
class EvaluationConfig:
    """Configuration parameters for imputation evaluation."""

    metrics: list[str] = field(default_factory=lambda: ["mae", "rmse"])
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FEATURE_WEIGHTED
    supported_methods: list[str] = field(
        default_factory=lambda: [
            "baseline_median",
            "baseline_mean",
            "knn",
            "iterative",
            "random_forest",
        ]
    )

    def __post_init__(self) -> None:
        """Validate evaluation configuration invariants."""
        valid_metrics = {"mae", "rmse", "nrmse", "accuracy"}
        for m in self.metrics:
            if m.lower() not in valid_metrics:
                raise ConfigurationError(
                    f"Unsupported metric: {m}. Valid metrics are {valid_metrics}.",
                    context={"invalid_metric": m},
                )

        if not self.metrics:
            raise ConfigurationError("EvaluationConfig must specify at least one metric.")
