"""Downstream ML Model Impact and End-to-End Validation Package."""

from missing_data_platform.downstream.config import (
    DownstreamBenchmarkConfig,
    DownstreamConfig,
    DownstreamModelType,
    DownstreamTaskType,
)
from missing_data_platform.downstream.engine import DownstreamEvaluationEngine
from missing_data_platform.downstream.metrics import (
    calculate_classification_metrics,
    calculate_group_disparity,
    calculate_group_downstream_metrics,
    calculate_imputation_downstream_correlation,
    calculate_performance_delta,
    calculate_performance_recovery,
    calculate_regression_metrics,
)
from missing_data_platform.downstream.models import DownstreamModelWrapper
from missing_data_platform.downstream.report import (
    DownstreamBenchmarkReport,
    DownstreamEvaluationResult,
    GroupDownstreamMetric,
    MissingnessCurveReport,
    RepeatedDownstreamReport,
)

__all__ = [
    "DownstreamTaskType",
    "DownstreamModelType",
    "DownstreamConfig",
    "DownstreamBenchmarkConfig",
    "DownstreamModelWrapper",
    "DownstreamEvaluationEngine",
    "GroupDownstreamMetric",
    "DownstreamEvaluationResult",
    "DownstreamBenchmarkReport",
    "RepeatedDownstreamReport",
    "MissingnessCurveReport",
    "calculate_classification_metrics",
    "calculate_regression_metrics",
    "calculate_performance_delta",
    "calculate_performance_recovery",
    "calculate_group_downstream_metrics",
    "calculate_group_disparity",
    "calculate_imputation_downstream_correlation",
]
