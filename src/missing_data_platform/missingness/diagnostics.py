"""Statistical diagnostics and mechanism hypothesis testing for MCAR, MAR, and MNAR.

Implements bivariate mean difference tests (Welch t-tests with Cohen's d) and categorical
contingency tests (Chi-Square independence with Cramer's V) to evaluate evidence
compatible or inconsistent with MCAR and MAR assumptions without making unprovable claims.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats

from missing_data_platform.missingness.report import (
    MARDiagnosticReport,
    MCARDiagnosticReport,
    StatisticalTestResult,
)


def compute_cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """Compute Cohen's d effect size for two independent continuous samples."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    var1, var2 = float(group1.var(ddof=1)), float(group2.var(ddof=1))
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    mean_diff = float(group1.mean() - group2.mean())
    return float(mean_diff / pooled_std)


def compute_cramers_v(contingency_table: pd.DataFrame) -> float:
    """Compute Cramer's V effect size for categorical association from a contingency matrix."""
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    if n == 0:
        return 0.0

    min_dim = min(contingency_table.shape) - 1
    if min_dim <= 0:
        return 0.0

    v = np.sqrt(chi2 / (n * min_dim))
    return float(v)


def run_mcar_diagnostics(
    df: pd.DataFrame,
    target_missing_features: list[str],
    auxiliary_continuous_features: list[str],
    alpha: float = 0.05,
) -> MCARDiagnosticReport:
    """Evaluate whether missingness in target features is independent of observed continuous variables."""
    tests: list[StatisticalTestResult] = []

    for target_col in target_missing_features:
        if target_col not in df.columns:
            continue

        null_mask = df[target_col].isna()
        if null_mask.sum() == 0 or null_mask.sum() == len(df):
            continue  # No variance in missingness

        for aux_col in auxiliary_continuous_features:
            if aux_col not in df.columns or aux_col == target_col:
                continue

            aux_series = pd.to_numeric(df[aux_col], errors="coerce")
            group_obs = aux_series[~null_mask].dropna()
            group_miss = aux_series[null_mask].dropna()

            if len(group_obs) < 3 or len(group_miss) < 3:
                continue

            # Two-sample Welch t-test (does not assume equal variance)
            t_stat, p_val = stats.ttest_ind(group_obs, group_miss, equal_var=False)
            effect_d = compute_cohens_d(group_obs, group_miss)

            p_value_float = float(p_val) if not np.isnan(p_val) else 1.0
            t_stat_float = float(t_stat) if not np.isnan(t_stat) else 0.0
            is_sig = p_value_float < alpha

            if is_sig:
                interp = (
                    f"Statistically significant difference in '{aux_col}' observed between missing vs "
                    f"observed cohorts of '{target_col}' (p={p_value_float:.4e}, d={effect_d:.2f}). "
                    f"Evidence is inconsistent with pure MCAR."
                )
            else:
                interp = (
                    f"No statistically significant difference in '{aux_col}' across '{target_col}' "
                    f"missingness cohorts (p={p_value_float:.4f}). Compatible with MCAR."
                )

            tests.append(
                StatisticalTestResult(
                    test_name="Welch_Two_Sample_T_Test",
                    target_feature_missing_indicator=f"is_missing_{target_col}",
                    conditioned_on_variable=aux_col,
                    test_statistic=round(t_stat_float, 4),
                    p_value=round(p_value_float, 6),
                    effect_size=round(abs(effect_d), 4),
                    effect_size_metric="Cohen's d",
                    is_statistically_significant=is_sig,
                    interpretation=interp,
                )
            )

    sig_count = sum(1 for t in tests if t.is_statistically_significant)
    if sig_count > 0:
        summary = (
            f"{sig_count} of {len(tests)} bivariate tests showed significant covariate mean differences. "
            f"Observed evidence is inconsistent with pure MCAR and suggests MAR or MNAR mechanisms."
        )
    else:
        summary = (
            f"Zero significant covariate differences detected across {len(tests)} tests. "
            f"Data missingness is statistically compatible with MCAR under the observed auxiliary variables."
        )

    limitations = (
        "Note: Bivariate t-tests evaluate only pairwise linear mean differences across observed continuous "
        "covariates. They cannot detect complex non-linear associations or latent MNAR dependencies."
    )

    return MCARDiagnosticReport(
        tests_evaluated=tests,
        total_tests_conducted=len(tests),
        significant_tests_count=sig_count,
        evidence_summary=summary,
        test_limitations=limitations,
    )


def run_mar_association_tests(
    df: pd.DataFrame,
    target_missing_features: list[str],
    categorical_covariates: list[str],
    alpha: float = 0.05,
) -> MARDiagnosticReport:
    """Evaluate associations between missingness indicators and observed categorical attributes."""
    tests: list[StatisticalTestResult] = []

    for target_col in target_missing_features:
        if target_col not in df.columns:
            continue

        null_mask = df[target_col].isna()
        if null_mask.sum() == 0 or null_mask.sum() == len(df):
            continue

        for cat_col in categorical_covariates:
            if cat_col not in df.columns or cat_col == target_col:
                continue

            contingency = pd.crosstab(null_mask, df[cat_col].fillna("Missing"))
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue

            chi2_stat, p_val, _, _ = stats.chi2_contingency(contingency)
            cramers_v = compute_cramers_v(contingency)

            p_value_float = float(p_val) if not np.isnan(p_val) else 1.0
            chi2_stat_float = float(chi2_stat) if not np.isnan(chi2_stat) else 0.0
            is_sig = p_value_float < alpha

            if is_sig:
                interp = (
                    f"Statistically significant dependence between missingness of '{target_col}' and '{cat_col}' "
                    f"(Chi2={chi2_stat_float:.2f}, p={p_value_float:.4e}, Cramer's V={cramers_v:.3f}). "
                    f"Evidence indicates missingness is conditioned on observed attributes (compatible with MAR)."
                )
            else:
                interp = (
                    f"No significant association detected between '{target_col}' missingness and '{cat_col}' "
                    f"(p={p_value_float:.4f})."
                )

            tests.append(
                StatisticalTestResult(
                    test_name="Chi_Square_Independence_Test",
                    target_feature_missing_indicator=f"is_missing_{target_col}",
                    conditioned_on_variable=cat_col,
                    test_statistic=round(chi2_stat_float, 4),
                    p_value=round(p_value_float, 6),
                    effect_size=round(cramers_v, 4),
                    effect_size_metric="Cramer's V",
                    is_statistically_significant=is_sig,
                    interpretation=interp,
                )
            )

    sig_count = sum(1 for t in tests if t.is_statistically_significant)
    if sig_count > 0:
        summary = (
            f"Observed categorical attributes significantly correlate with missingness patterns "
            f"({sig_count} of {len(tests)} associations significant), supporting MAR modeling assumptions."
        )
    else:
        summary = "No strong categorical dependencies observed for missingness indicators."

    return MARDiagnosticReport(
        associations=tests,
        significant_associations_count=sig_count,
        evidence_summary=summary,
    )


def generate_mnar_limitation_statement() -> str:
    """Generate formal statistical disclaimer on the mathematical unprovability of MNAR."""
    return (
        "STATISTICAL METHODOLOGY DISCLAIMER: Missing Not At Random (MNAR) implies that the probability of "
        "a value being missing is directly dependent on the unobserved value itself. By mathematical definition, "
        "MNAR cannot be confirmed or ruled out using purely observational data without external ground-truth "
        "follow-up measurements. Downstream imputation methods (e.g., KNN, MICE, Random Forest) will operate "
        "under conditional MAR assumptions, and sensitivity analyses should be applied to evaluate robustness."
    )
