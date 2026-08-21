"""Ground Truth storage and mask management.

Securely encapsulates original observed values for artificially hidden cells and maintains
the boolean mask matrix separating natural missingness from artificial missingness.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class GroundTruthStore:
    """Encapsulates ground truth values and mask indicators for an experiment."""

    experiment_id: str
    mask_matrix: pd.DataFrame  # Boolean DataFrame: True if cell was artificially masked
    original_values: dict[str, pd.Series] = field(
        default_factory=dict
    )  # column -> true series of masked cells

    @property
    def total_masked_cells(self) -> int:
        """Total number of cells artificially masked across all features."""
        return int(self.mask_matrix.sum().sum())

    @property
    def feature_masked_counts(self) -> dict[str, int]:
        """Number of artificially masked cells per feature."""
        return {col: int(self.mask_matrix[col].sum()) for col in self.mask_matrix.columns}

    def get_ground_truth(self, column_name: str) -> pd.Series:
        """Retrieve original observed values for artificially masked cells in a column."""
        if column_name not in self.original_values:
            return pd.Series(dtype=object)
        return self.original_values[column_name]

    def is_artificially_masked(self, column_name: str, index: Any) -> bool:
        """Check if a specific cell was artificially masked by the experiment."""
        if column_name not in self.mask_matrix.columns:
            return False
        return bool(self.mask_matrix.loc[index, column_name])

    def to_metadata_dict(self) -> dict[str, Any]:
        """Export sanitized metadata summary without leaking sensitive cell values."""
        return {
            "experiment_id": self.experiment_id,
            "total_masked_cells": self.total_masked_cells,
            "features_masked": self.feature_masked_counts,
        }
