"""AI/ML-Based Missing Data Imputation & Bias Reduction Platform.

A production data and ML framework designed to benchmark imputation strategies,
audit representation stability across customer segments, and optimize downstream
purchase prediction models.
"""

from missing_data_platform.__version__ import __author__, __version__
from missing_data_platform.config import AppEnvironment, LogLevel, Settings, get_settings
from missing_data_platform.exceptions import (
    ConfigurationError,
    DataQualityError,
    EvaluationError,
    ImputationError,
    ModelTrainingError,
    PlatformError,
    StorageError,
)
from missing_data_platform.ingestion import (
    ColumnDefinition,
    CsvParser,
    ExtraColumnsAction,
    IngestionEngine,
    IngestionLineage,
    IngestionResult,
    RawDataContract,
    SchemaValidator,
)
from missing_data_platform.logging import configure_logging, get_logger
from missing_data_platform.quality import (
    CheckDetail,
    DataQualityConfig,
    DataQualityEngine,
    DistributionSummary,
    DuplicateMetric,
    MissingnessMetric,
    QualityReport,
    QualitySeverity,
    QualityStatus,
)

__all__ = [
    "__version__",
    "__author__",
    "Settings",
    "get_settings",
    "AppEnvironment",
    "LogLevel",
    "configure_logging",
    "get_logger",
    "PlatformError",
    "ConfigurationError",
    "StorageError",
    "DataQualityError",
    "ImputationError",
    "EvaluationError",
    "ModelTrainingError",
    "RawDataContract",
    "ColumnDefinition",
    "CsvParser",
    "SchemaValidator",
    "ExtraColumnsAction",
    "IngestionLineage",
    "IngestionEngine",
    "IngestionResult",
    "QualityStatus",
    "QualitySeverity",
    "DataQualityConfig",
    "CheckDetail",
    "MissingnessMetric",
    "DuplicateMetric",
    "DistributionSummary",
    "QualityReport",
    "DataQualityEngine",
]
