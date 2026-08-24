"""Reporting dataclasses and audit schemas for downstream ML impact validation."""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd


@dataclass
class GroupDownstreamMetric:
    """Downstream ML performance metrics for a specific customer or demographic cohort."""

    group_value: str
    sample_count: int
    is_small_group: bool
    metrics: dict[str, float | None]


@dataclass
class DownstreamEvaluationResult:
    """Comprehensive evaluation record for a single downstream experiment run."""

    experiment_id: str
    dataset_version: str
    missingness_rate: float
    mask_seed: int
    imputation_method: str
    mitigation_enabled: bool
    downstream_model: str
    primary_metric: str
    metrics: dict[str, float | None]
    group_metrics: list[GroupDownstreamMetric]
    group_disparities: dict[str, Any]
    performance_delta: dict[str, float | None]
    recovery: float | None
    imputation_mae: float | None = None
    imputation_rmse: float | None = None
    runtime_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    reproducibility_metadata: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_metadata_dict(self) -> dict[str, Any]:
        """Export sanitized metadata dictionary without customer records."""
        return {
            "experiment_id": self.experiment_id,
            "dataset_version": self.dataset_version,
            "missingness_rate": self.missingness_rate,
            "mask_seed": self.mask_seed,
            "imputation_method": self.imputation_method,
            "mitigation_enabled": self.mitigation_enabled,
            "downstream_model": self.downstream_model,
            "primary_metric": self.primary_metric,
            "metrics": self.metrics,
            "group_metrics": [asdict(g) for g in self.group_metrics],
            "group_disparities": self.group_disparities,
            "performance_delta": self.performance_delta,
            "recovery": self.recovery,
            "imputation_mae": self.imputation_mae,
            "imputation_rmse": self.imputation_rmse,
            "runtime_seconds": self.runtime_seconds,
            "warnings": self.warnings,
            "reproducibility_metadata": self.reproducibility_metadata,
            "created_at_utc": self.created_at_utc,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize evaluation result to JSON string."""
        return json.dumps(self.to_metadata_dict(), indent=indent)


@dataclass
class DownstreamBenchmarkReport:
    """Master benchmark report comparing complete baseline, candidate imputers, and mitigations."""

    experiment_id: str
    dataset_version: str
    downstream_model: str
    primary_metric: str
    complete_baseline: DownstreamEvaluationResult
    method_results: dict[str, DownstreamEvaluationResult]
    mitigated_results: dict[str, DownstreamEvaluationResult] = field(default_factory=dict)
    comparison_table: list[dict[str, Any]] = field(default_factory=list)
    imputation_vs_downstream_summary: dict[str, Any] = field(default_factory=dict)
    group_disparity_summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_comparison_dataframe(self) -> pd.DataFrame:
        """Convert comparative results table into a pandas DataFrame."""
        return pd.DataFrame(self.comparison_table)

    def to_metadata_dict(self) -> dict[str, Any]:
        """Export sanitized benchmark summary dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "dataset_version": self.dataset_version,
            "downstream_model": self.downstream_model,
            "primary_metric": self.primary_metric,
            "complete_baseline": self.complete_baseline.to_metadata_dict(),
            "method_results": {m: res.to_metadata_dict() for m, res in self.method_results.items()},
            "mitigated_results": {
                m: res.to_metadata_dict() for m, res in self.mitigated_results.items()
            },
            "comparison_table": self.comparison_table,
            "imputation_vs_downstream_summary": self.imputation_vs_downstream_summary,
            "group_disparity_summary": self.group_disparity_summary,
            "warnings": self.warnings,
            "created_at_utc": self.created_at_utc,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize benchmark report to JSON string."""
        return json.dumps(self.to_metadata_dict(), indent=indent)


@dataclass
class RepeatedDownstreamReport:
    """Summary of repeated downstream runs across random seeds to measure performance stability."""

    experiment_id: str
    repeated_seeds: list[int]
    total_repetitions: int
    primary_metric: str
    method_stats: dict[str, dict[str, float]]
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_metadata_dict(self) -> dict[str, Any]:
        """Export sanitized repeated experiment metadata."""
        return {
            "experiment_id": self.experiment_id,
            "repeated_seeds": self.repeated_seeds,
            "total_repetitions": self.total_repetitions,
            "primary_metric": self.primary_metric,
            "method_stats": self.method_stats,
            "created_at_utc": self.created_at_utc,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize repeated experiment report to JSON string."""
        return json.dumps(self.to_metadata_dict(), indent=indent)


@dataclass
class MissingnessCurveReport:
    """Performance degradation trajectory across increasing missingness rates."""

    experiment_id: str
    primary_metric: str
    missingness_rates: list[float]
    curve_points: list[dict[str, Any]]
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dataframe(self) -> pd.DataFrame:
        """Convert curve data into a pandas DataFrame."""
        return pd.DataFrame(self.curve_points)

    def to_metadata_dict(self) -> dict[str, Any]:
        """Export sanitized curve metadata."""
        return {
            "experiment_id": self.experiment_id,
            "primary_metric": self.primary_metric,
            "missingness_rates": self.missingness_rates,
            "curve_points": self.curve_points,
            "created_at_utc": self.created_at_utc,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize curve report to JSON string."""
        return json.dumps(self.to_metadata_dict(), indent=indent)
