"""Imputation tracking metrics and result container schemas."""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from missing_data_platform.imputation.config import BaselineStrategy


@dataclass
class FeatureImputationMetric:
    """Tracking statistics for imputation on a single feature."""

    feature_name: str
    strategy_applied: BaselineStrategy
    statistic_value: Any
    missing_before: int
    imputed_count: int
    missing_after: int


@dataclass
class ImputationResult:
    """Comprehensive container for imputed dataset and execution audit metadata."""

    imputed_dataset: pd.DataFrame
    experiment_id: str
    numeric_strategy: BaselineStrategy
    categorical_strategy: BaselineStrategy
    total_records: int
    total_cells_imputed: int
    feature_metrics: list[FeatureImputationMetric]
    imputation_parameters: dict[str, Any]
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_metadata_dict(self) -> dict[str, Any]:
        """Export execution metadata without leaking dataset records."""
        return {
            "experiment_id": self.experiment_id,
            "numeric_strategy": self.numeric_strategy.value,
            "categorical_strategy": self.categorical_strategy.value,
            "total_records": self.total_records,
            "total_cells_imputed": self.total_cells_imputed,
            "imputation_parameters": self.imputation_parameters,
            "feature_metrics": [asdict(m) for m in self.feature_metrics],
            "created_at_utc": self.created_at_utc,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize metadata summary to JSON string."""
        return json.dumps(self.to_metadata_dict(), indent=indent)
