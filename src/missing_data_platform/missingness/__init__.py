"""Missingness Analysis & Mechanism Diagnostics package."""

from missing_data_platform.missingness.diagnostics import (
    generate_mnar_limitation_statement,
    run_mar_association_tests,
    run_mcar_diagnostics,
)
from missing_data_platform.missingness.engine import MissingnessAnalysisEngine
from missing_data_platform.missingness.group_analysis import (
    analyze_feature_missingness_by_group,
    create_age_bands,
)
from missing_data_platform.missingness.indicators import (
    compute_missingness_correlation,
    create_missingness_indicators,
)
from missing_data_platform.missingness.profiler import (
    profile_feature_missingness,
    profile_missingness_patterns,
    profile_row_missingness,
)
from missing_data_platform.missingness.report import (
    FeatureMissingnessProfile,
    GroupDisparitySummary,
    GroupMissingnessStat,
    MARDiagnosticReport,
    MCARDiagnosticReport,
    MissingnessAnalysisReport,
    MissingnessPattern,
    RowMissingnessProfile,
    StatisticalTestResult,
)

__all__ = [
    "create_missingness_indicators",
    "compute_missingness_correlation",
    "profile_feature_missingness",
    "profile_row_missingness",
    "profile_missingness_patterns",
    "create_age_bands",
    "analyze_feature_missingness_by_group",
    "run_mcar_diagnostics",
    "run_mar_association_tests",
    "generate_mnar_limitation_statement",
    "MissingnessAnalysisEngine",
    "FeatureMissingnessProfile",
    "RowMissingnessProfile",
    "MissingnessPattern",
    "GroupMissingnessStat",
    "GroupDisparitySummary",
    "StatisticalTestResult",
    "MCARDiagnosticReport",
    "MARDiagnosticReport",
    "MissingnessAnalysisReport",
]
