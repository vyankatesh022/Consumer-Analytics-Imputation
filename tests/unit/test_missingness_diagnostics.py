"""Unit tests for statistical mechanism diagnostics (MCAR, MAR, MNAR)."""

import numpy as np
import pandas as pd

from missing_data_platform.missingness.diagnostics import (
    compute_cohens_d,
    generate_mnar_limitation_statement,
    run_mar_association_tests,
    run_mcar_diagnostics,
)


def test_compute_cohens_d() -> None:
    """Verify calculation of Cohen's d effect size."""
    g1 = pd.Series([10.0, 12.0, 11.0, 13.0])
    g2 = pd.Series([20.0, 22.0, 21.0, 23.0])
    d = compute_cohens_d(g1, g2)
    assert d < -5.0  # Clear large negative effect difference


def test_run_mcar_diagnostics_detection() -> None:
    """Verify that Welch t-test flags significant covariate mean differences."""
    # Dataset where missingness in income is strongly associated with high age
    df = pd.DataFrame(
        {
            "income": [50000.0, 52000.0, 48000.0, np.nan, np.nan, np.nan],
            "age": [22.0, 24.0, 23.0, 70.0, 72.0, 71.0],  # Age differs drastically between cohorts
        }
    )

    report = run_mcar_diagnostics(
        df=df,
        target_missing_features=["income"],
        auxiliary_continuous_features=["age"],
        alpha=0.05,
    )

    assert report.total_tests_conducted == 1
    assert report.significant_tests_count == 1
    assert report.tests_evaluated[0].is_statistically_significant is True
    assert "inconsistent with pure MCAR" in report.evidence_summary


def test_run_mar_association_tests_detection() -> None:
    """Verify Chi-Square test detecting categorical association with missingness."""
    # Dataset with sufficient sample size where 'East' always has missing income and 'West' never does
    df = pd.DataFrame(
        {
            "income": [50000.0] * 10 + [np.nan] * 10,
            "region": ["West"] * 10 + ["East"] * 10,
        }
    )

    report = run_mar_association_tests(
        df=df,
        target_missing_features=["income"],
        categorical_covariates=["region"],
        alpha=0.05,
    )

    assert len(report.associations) == 1
    assert report.associations[0].is_statistically_significant is True
    assert report.associations[0].effect_size > 0.5  # Strong Cramér's V


def test_generate_mnar_limitation_statement() -> None:
    """Verify formal MNAR methodology caveat generation."""
    statement = generate_mnar_limitation_statement()
    assert "MNAR cannot be confirmed or ruled out" in statement
    assert "observational data" in statement
