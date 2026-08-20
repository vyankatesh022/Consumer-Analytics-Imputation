"""Data Quality Report and metric models.

Provides structured, serializable data structures representing findings across
schema, missingness, duplicates, numerical boundaries, categorical vocabularies,
and statistical distributions without mutating the underlying dataset.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from missing_data_platform.quality.rules import QualitySeverity, QualityStatus


@dataclass
class CheckDetail:
    """Individual data quality check finding."""

    check_name: str
    status: QualityStatus
    severity: QualitySeverity
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class MissingnessMetric:
    """Detailed missingness statistics for a single column."""

    column_name: str
    total_records: int
    missing_count: int
    missing_percentage: float
    is_nullable: bool


@dataclass
class DuplicateMetric:
    """Duplicate record statistics across rows and identifiers."""

    total_records: int
    full_row_duplicates: int
    identifier_duplicates: int
    duplicate_row_percentage: float
    duplicate_id_percentage: float


@dataclass
class DistributionSummary:
    """Descriptive statistics for a numeric or categorical feature."""

    column_name: str
    count: int
    missing_count: int
    distinct_count: int
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    std_value: float | None = None
    median_value: float | None = None
    top_categories: dict[str, int] | None = None


@dataclass
class QualityReport:
    """Comprehensive Data Quality Audit Report."""

    dataset_id: str
    total_records: int
    total_columns: int
    overall_status: QualityStatus
    passed_checks: int
    warning_checks: int
    failed_checks: int
    checks: list[CheckDetail] = field(default_factory=list)
    missingness_summary: dict[str, MissingnessMetric] = field(default_factory=dict)
    duplicate_summary: DuplicateMetric | None = None
    distribution_summaries: dict[str, DistributionSummary] = field(default_factory=dict)
    generated_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def has_failures(self) -> bool:
        """True if any error-level check failed."""
        return self.overall_status == QualityStatus.FAIL

    @property
    def has_warnings(self) -> bool:
        """True if one or more warning-level checks were triggered."""
        return self.warning_checks > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete quality report as structured dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize complete quality report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
