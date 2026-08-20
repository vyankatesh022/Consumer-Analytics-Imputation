"""Data models for Missingness Analysis and Mechanism Diagnostics Reports.

Defines serializable schemas for feature-level, row-level, pattern-level, group-level,
and statistical mechanism diagnostic findings.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class FeatureMissingnessProfile:
    """Missingness summary for an individual column."""

    column_name: str
    total_records: int
    missing_count: int
    observed_count: int
    missing_percentage: float
    rank: int


@dataclass
class RowMissingnessProfile:
    """Row-level missingness distribution across the dataset."""

    total_rows: int
    completely_observed_rows: int
    completely_observed_percentage: float
    rows_with_any_missing: int
    rows_with_any_missing_percentage: float
    max_missing_columns_in_a_row: int
    missing_counts_distribution: dict[int, int]  # number of missing columns -> row count


@dataclass
class MissingnessPattern:
    """Distinct combination of missing columns observed in the dataset."""

    pattern_id: int
    missing_columns: list[str]
    row_count: int
    percentage: float


@dataclass
class GroupMissingnessStat:
    """Missingness rates across customer demographic/behavioral groups."""

    feature_analyzed: str
    grouping_variable: str
    group_value: str
    group_population: int
    missing_count: int
    missing_percentage: float


@dataclass
class GroupDisparitySummary:
    """Disparity analysis across segments for a specific feature."""

    feature_analyzed: str
    grouping_variable: str
    min_missing_group: str
    min_missing_percentage: float
    max_missing_group: str
    max_missing_percentage: float
    disparity_ratio: float  # max_pct / min_pct (or max - min if min=0)
    groups: list[GroupMissingnessStat] = field(default_factory=list)


@dataclass
class StatisticalTestResult:
    """Result of a formal bivariate hypothesis test against a missingness indicator."""

    test_name: str
    target_feature_missing_indicator: str
    conditioned_on_variable: str
    test_statistic: float
    p_value: float
    effect_size: float
    effect_size_metric: str  # e.g., "Cohen's d", "Cramer's V"
    is_statistically_significant: bool  # alpha = 0.05
    interpretation: str


@dataclass
class MCARDiagnosticReport:
    """Summary of tests evaluating consistency with MCAR assumptions."""

    tests_evaluated: list[StatisticalTestResult]
    total_tests_conducted: int
    significant_tests_count: int
    evidence_summary: str
    test_limitations: str


@dataclass
class MARDiagnosticReport:
    """Summary of tests evaluating associations between observed variables and missingness."""

    associations: list[StatisticalTestResult]
    significant_associations_count: int
    evidence_summary: str


@dataclass
class MissingnessAnalysisReport:
    """Comprehensive Missingness Analysis & Mechanism Diagnostics Report."""

    dataset_id: str
    total_records: int
    total_features: int
    features_with_missingness_count: int
    feature_profiles: list[FeatureMissingnessProfile] = field(default_factory=list)
    row_profile: RowMissingnessProfile | None = None
    top_patterns: list[MissingnessPattern] = field(default_factory=list)
    group_disparities: list[GroupDisparitySummary] = field(default_factory=list)
    mcar_diagnostics: MCARDiagnosticReport | None = None
    mar_diagnostics: MARDiagnosticReport | None = None
    mnar_limitation_statement: str = ""
    executive_statistical_interpretation: str = ""
    generated_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to structured dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
