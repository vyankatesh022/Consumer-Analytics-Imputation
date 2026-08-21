"""Experiment results and validation metrics for artificial missingness benchmarking."""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from missing_data_platform.masking.config import MaskingStrategy
from missing_data_platform.masking.ground_truth import GroundTruthStore


@dataclass
class FeatureMaskingSummary:
    """Summary of artificial masking outcomes for an individual feature."""

    feature_name: str
    total_records: int
    natural_missing_count: int
    eligible_observed_count: int
    requested_mask_rate: float
    artificially_masked_count: int
    actual_mask_rate: float
    total_missing_after_masking: int


@dataclass
class MaskingExperimentResult:
    """Complete container for artificial missingness benchmark artifacts and metadata."""

    experiment_id: str
    dataset_version: str
    strategy: MaskingStrategy
    random_seed: int
    requested_mask_rate: float
    total_records: int
    total_artificially_masked_cells: int
    overall_actual_mask_rate: float
    feature_summaries: list[FeatureMaskingSummary]
    masked_dataset: pd.DataFrame
    ground_truth_mask: pd.DataFrame
    ground_truth_store: GroundTruthStore
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_metadata_dict(self) -> dict[str, Any]:
        """Export sanitized experiment metadata without including full DataFrames."""
        return {
            "experiment_id": self.experiment_id,
            "dataset_version": self.dataset_version,
            "strategy": self.strategy.value,
            "random_seed": self.random_seed,
            "requested_mask_rate": self.requested_mask_rate,
            "total_records": self.total_records,
            "total_artificially_masked_cells": self.total_artificially_masked_cells,
            "overall_actual_mask_rate": self.overall_actual_mask_rate,
            "feature_summaries": [asdict(f) for f in self.feature_summaries],
            "created_at_utc": self.created_at_utc,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize metadata summary to JSON."""
        return json.dumps(self.to_metadata_dict(), indent=indent)
