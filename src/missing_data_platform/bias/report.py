"""Reporting containers and summary structures for bias and representation analysis."""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd


@dataclass
class GroupRepresentation:
    """Demographic or segment group representation statistics in the dataset."""

    group_value: str
    population_count: int
    population_percentage: float
    eligible_evaluation_cells: int
    evaluation_percentage: float
    is_small_group: bool


@dataclass
class GroupMissingness:
    """Missingness rates per group and target feature."""

    group_value: str
    feature_name: str
    missing_count: int
    observed_count: int
    missing_rate: float
    is_small_group: bool


@dataclass
class GroupImputationPerformance:
    """Imputation algorithm performance metrics for an individual group and feature."""

    group_value: str
    method: str
    feature_name: str
    metric_name: str
    metric_value: float | None
    sample_count: int
    valid_prediction_count: int
    missing_prediction_count: int
    is_suppressed: bool
    warning: str | None = None


@dataclass
class DisparityResult:
    """Pairwise disparity comparison between two groups for a given method, feature, and metric."""

    comparison_name: str
    method: str
    feature_name: str
    metric_name: str
    group_a: str
    group_b: str
    value_group_a: float | None
    value_group_b: float | None
    absolute_disparity: float | None
    relative_disparity: float | None
    sample_count_a: int
    sample_count_b: int


@dataclass
class BiasAnalysisResult:
    """Comprehensive container for group representation, missingness disparities, and performance."""

    experiment_id: str
    dataset_version: str
    grouping_column: str
    minimum_group_size: int
    representation_results: list[GroupRepresentation]
    missingness_results: list[GroupMissingness]
    performance_results: list[GroupImputationPerformance]
    disparity_results: list[DisparityResult]
    best_method_per_group: dict[str, str]
    global_best_method: str | None
    warnings: list[str]
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_representation_dataframe(self) -> pd.DataFrame:
        """Convert representation statistics to DataFrame."""
        return pd.DataFrame([asdict(r) for r in self.representation_results])

    def to_disparity_dataframe(self) -> pd.DataFrame:
        """Convert disparity results to DataFrame."""
        return pd.DataFrame([asdict(d) for d in self.disparity_results])

    def to_metadata_dict(self) -> dict[str, Any]:
        """Export sanitized metadata summary without raw customer values."""
        return {
            "experiment_id": self.experiment_id,
            "dataset_version": self.dataset_version,
            "grouping_column": self.grouping_column,
            "minimum_group_size": self.minimum_group_size,
            "representation_results": [asdict(r) for r in self.representation_results],
            "missingness_results": [asdict(m) for m in self.missingness_results],
            "performance_results": [asdict(p) for p in self.performance_results],
            "disparity_results": [asdict(d) for d in self.disparity_results],
            "best_method_per_group": self.best_method_per_group,
            "global_best_method": self.global_best_method,
            "warnings": self.warnings,
            "created_at_utc": self.created_at_utc,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize bias analysis summary to JSON string."""
        return json.dumps(self.to_metadata_dict(), indent=indent)
