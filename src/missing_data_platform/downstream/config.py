"""Configuration schemas for downstream ML model impact evaluation.

Defines supported downstream model architectures, predictive tasks, hyperparameter presets,
leakage protection rules, and benchmark experiment configurations.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from missing_data_platform.exceptions import ConfigurationError


class DownstreamTaskType(StrEnum):
    """Supported predictive task types for downstream evaluation."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class DownstreamModelType(StrEnum):
    """Supported downstream estimator families."""

    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    GRADIENT_BOOSTING = "gradient_boosting"
    RIDGE = "ridge"


@dataclass
class DownstreamConfig:
    """Master configuration for downstream machine learning evaluation."""

    model_type: DownstreamModelType = DownstreamModelType.RANDOM_FOREST
    task_type: DownstreamTaskType = DownstreamTaskType.CLASSIFICATION
    primary_metric: str = "f1"
    test_size: float = 0.20
    random_seed: int = 42
    group_column: str = "customer_segment"
    minimum_group_size: int = 5
    model_params: dict[str, Any] = field(default_factory=dict)
    protected_features: list[str] = field(
        default_factory=lambda: ["customer_id", "purchase_next_month"]
    )
    dataset_version: str = "1.0.0"

    def __post_init__(self) -> None:
        """Validate downstream configuration constraints."""
        if not (0.0 < self.test_size < 1.0):
            raise ConfigurationError(
                f"Invalid test_size: {self.test_size}. Must be between 0.0 and 1.0.",
                context={"test_size": self.test_size},
            )

        if self.minimum_group_size < 1:
            raise ConfigurationError(
                f"Invalid minimum_group_size: {self.minimum_group_size}. Must be >= 1.",
                context={"minimum_group_size": self.minimum_group_size},
            )

        if not self.primary_metric or not self.primary_metric.strip():
            raise ConfigurationError("primary_metric cannot be empty.")

        # Default valid metrics check
        valid_clf_metrics = {"f1", "accuracy", "precision", "recall", "roc_auc", "pr_auc"}
        valid_reg_metrics = {"rmse", "mae", "r2", "nrmse"}

        if self.task_type == DownstreamTaskType.CLASSIFICATION:
            if self.primary_metric not in valid_clf_metrics:
                raise ConfigurationError(
                    f"Invalid classification primary_metric: '{self.primary_metric}'. "
                    f"Must be one of {sorted(valid_clf_metrics)}.",
                    context={"primary_metric": self.primary_metric},
                )
        elif (
            self.task_type == DownstreamTaskType.REGRESSION
            and self.primary_metric not in valid_reg_metrics
        ):
            raise ConfigurationError(
                f"Invalid regression primary_metric: '{self.primary_metric}'. "
                f"Must be one of {sorted(valid_reg_metrics)}.",
                context={"primary_metric": self.primary_metric},
            )


@dataclass
class DownstreamBenchmarkConfig:
    """Configuration for sweeping imputation methods, missingness rates, and random seeds."""

    experiment_id: str = "downstream_benchmark_exp"
    methods: list[str] = field(
        default_factory=lambda: [
            "baseline_median",
            "baseline_mean",
            "knn",
            "iterative",
            "random_forest",
        ]
    )
    missingness_rates: list[float] = field(default_factory=lambda: [0.10, 0.20, 0.30, 0.40, 0.50])
    repeated_seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    downstream_config: DownstreamConfig = field(default_factory=DownstreamConfig)
    include_mitigation: bool = True

    def __post_init__(self) -> None:
        """Validate benchmark sweep parameters."""
        if not self.experiment_id or not self.experiment_id.strip():
            raise ConfigurationError("experiment_id cannot be empty.")

        if not self.methods:
            raise ConfigurationError("methods list cannot be empty.")

        for rate in self.missingness_rates:
            if not (0.0 <= rate <= 1.0):
                raise ConfigurationError(
                    f"Invalid missingness_rate: {rate}. Must be between 0.0 and 1.0.",
                    context={"rate": rate},
                )

        if not self.repeated_seeds:
            raise ConfigurationError("repeated_seeds cannot be empty.")
