"""Bias Mitigation and Fairness-Aware Imputation package."""

from missing_data_platform.mitigation.config import (
    MitigationConfig,
    MitigationDecision,
    MitigationStrategy,
)
from missing_data_platform.mitigation.engine import (
    FairnessMitigationEngine,
    WeightedRandomForestImputer,
)
from missing_data_platform.mitigation.report import MitigationResult
from missing_data_platform.mitigation.weighting import calculate_group_sample_weights

__all__ = [
    "MitigationConfig",
    "MitigationStrategy",
    "MitigationDecision",
    "FairnessMitigationEngine",
    "WeightedRandomForestImputer",
    "MitigationResult",
    "calculate_group_sample_weights",
]
