"""Evaluation result containers, comparison reports, and metric reporting schemas."""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd


@dataclass
class FeatureEvaluationResult:
    """Evaluation metrics for a single feature under a specific imputation method."""

    feature_name: str
    feature_type: str
    method: str
    evaluated_count: int
    missing_prediction_count: int
    mae: float | None
    rmse: float | None
    nrmse: float | None = None
    accuracy: float | None = None
    additional_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class MethodEvaluationResult:
    """Aggregate evaluation outcomes for a specific imputation algorithm."""

    method: str
    experiment_id: str
    total_evaluated_cells: int
    missing_prediction_count: int
    macro_mae: float | None
    macro_rmse: float | None
    weighted_mae: float | None
    weighted_rmse: float | None
    feature_results: list[FeatureEvaluationResult]
    rank_mae: int | None = None
    rank_rmse: int | None = None


@dataclass
class BenchmarkComparisonReport:
    """Comprehensive benchmark comparison report across multiple imputation algorithms."""

    experiment_id: str
    dataset_version: str
    mask_strategy: str
    mask_rate: float
    mask_seed: int
    method_results: dict[str, MethodEvaluationResult]
    method_rankings: list[dict[str, Any]]
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_summary_dataframe(self) -> pd.DataFrame:
        """Convert benchmark method results into a tabular DataFrame for display."""
        rows = []
        for r in self.method_rankings:
            rows.append(
                {
                    "Rank (MAE)": r.get("rank_mae"),
                    "Rank (RMSE)": r.get("rank_rmse"),
                    "Method": r.get("method"),
                    "Evaluated Cells": r.get("total_evaluated_cells"),
                    "Missing Predictions": r.get("missing_prediction_count"),
                    "Weighted MAE": r.get("weighted_mae"),
                    "Weighted RMSE": r.get("weighted_rmse"),
                    "Macro MAE": r.get("macro_mae"),
                    "Macro RMSE": r.get("macro_rmse"),
                }
            )
        return pd.DataFrame(rows)

    def to_metadata_dict(self) -> dict[str, Any]:
        """Export sanitized benchmark metadata dictionary without raw data arrays."""
        return {
            "experiment_id": self.experiment_id,
            "dataset_version": self.dataset_version,
            "mask_strategy": self.mask_strategy,
            "mask_rate": self.mask_rate,
            "mask_seed": self.mask_seed,
            "method_rankings": self.method_rankings,
            "method_results": {
                m: {
                    "method": res.method,
                    "total_evaluated_cells": res.total_evaluated_cells,
                    "missing_prediction_count": res.missing_prediction_count,
                    "macro_mae": res.macro_mae,
                    "macro_rmse": res.macro_rmse,
                    "weighted_mae": res.weighted_mae,
                    "weighted_rmse": res.weighted_rmse,
                    "rank_mae": res.rank_mae,
                    "rank_rmse": res.rank_rmse,
                    "feature_results": [asdict(fr) for fr in res.feature_results],
                }
                for m, res in self.method_results.items()
            },
            "created_at_utc": self.created_at_utc,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize benchmark report to JSON string."""
        return json.dumps(self.to_metadata_dict(), indent=indent)


@dataclass
class RepeatedExperimentReport:
    """Statistical summary across repeated masking runs for imputation stability assessment."""

    experiment_id: str
    repeated_seeds: list[int]
    total_repetitions: int
    method_stats: dict[str, dict[str, float]]  # method -> {mean_mae, std_mae, mean_rmse, std_rmse}
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_metadata_dict(self) -> dict[str, Any]:
        """Export sanitized repeated experiment metadata."""
        return {
            "experiment_id": self.experiment_id,
            "repeated_seeds": self.repeated_seeds,
            "total_repetitions": self.total_repetitions,
            "method_stats": self.method_stats,
            "created_at_utc": self.created_at_utc,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize repeated experiment report to JSON string."""
        return json.dumps(self.to_metadata_dict(), indent=indent)
