"""Data Quality rules, status enums, and configuration specifications.

Defines status levels (PASS, WARN, FAIL), severity ratings (INFO, WARNING, ERROR),
and configurable validation thresholds for automated data quality assessment.
"""

from dataclasses import dataclass, field
from enum import StrEnum


class QualityStatus(StrEnum):
    """Execution status of a data quality check or report."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class QualitySeverity(StrEnum):
    """Severity classification for quality findings."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class QualityRule:
    """Specification of an individual data quality rule."""

    name: str
    description: str
    severity_on_failure: QualitySeverity = QualitySeverity.ERROR
    threshold: float | None = None


@dataclass
class DataQualityConfig:
    """Master configuration parameters for Data Quality Engine execution."""

    max_missing_percentage_warning: float = 30.0
    max_missing_percentage_error: float = 80.0
    allow_duplicate_records: bool = False
    allow_duplicate_ids: bool = False
    strict_schema_matching: bool = True
    custom_rules: dict[str, QualityRule] = field(default_factory=dict)
