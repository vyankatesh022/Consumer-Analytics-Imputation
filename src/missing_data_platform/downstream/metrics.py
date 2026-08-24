"""Mathematical and statistical evaluation metrics for downstream ML models.

Computes task-specific classification and regression metrics, performance degradation deltas,
performance recovery percentages, demographic group disparities, and rank correlations.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from missing_data_platform.downstream.config import DownstreamTaskType

HIGHER_IS_BETTER_METRICS = {
    "f1",
    "accuracy",
    "precision",
    "recall",
    "roc_auc",
    "pr_auc",
    "r2",
}

LOWER_IS_BETTER_METRICS = {
    "mae",
    "rmse",
    "nrmse",
    "mse",
}


def calculate_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series | None = None,
) -> dict[str, float | None]:
    """Calculate comprehensive standard classification performance metrics.

    Args:
        y_true: Ground truth binary or multiclass labels.
        y_pred: Predicted class labels.
        y_prob: Optional predicted class probabilities for the positive class.

    Returns:
        Dictionary containing accuracy, precision, recall, f1, roc_auc, and pr_auc.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()

    if len(y_t) == 0:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "pr_auc": None,
        }

    unique_classes = np.unique(y_t)
    is_binary = len(unique_classes) <= 2

    # Calculate discrete metrics
    acc = float(accuracy_score(y_t, y_p))
    prec = float(
        precision_score(
            y_t,
            y_p,
            average="binary" if is_binary else "macro",
            zero_division=0,
        )
    )
    rec = float(
        recall_score(
            y_t,
            y_p,
            average="binary" if is_binary else "macro",
            zero_division=0,
        )
    )
    f1 = float(
        f1_score(
            y_t,
            y_p,
            average="binary" if is_binary else "macro",
            zero_division=0,
        )
    )

    # Probabilistic metrics
    roc_auc: float | None = None
    pr_auc: float | None = None

    if y_prob is not None:
        probs = np.asarray(y_prob).ravel()
        if len(unique_classes) >= 2 and len(probs) == len(y_t):
            try:
                roc_auc = float(roc_auc_score(y_t, probs))
            except Exception:
                roc_auc = None

            if is_binary:
                try:
                    pr_auc = float(average_precision_score(y_t, probs))
                except Exception:
                    pr_auc = None

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
    }


def calculate_regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float | None]:
    """Calculate comprehensive standard regression performance metrics.

    Args:
        y_true: Ground truth continuous target values.
        y_pred: Predicted continuous values.

    Returns:
        Dictionary containing mae, rmse, r2, and nrmse.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()

    if len(y_t) == 0:
        return {
            "mae": None,
            "rmse": None,
            "r2": None,
            "nrmse": None,
        }

    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))

    # R-squared (requires at least 2 distinct observations)
    r2 = float(r2_score(y_t, y_p)) if len(y_t) > 1 and np.var(y_t) > 1e-9 else 0.0

    # NRMSE (normalized by target range)
    t_range = float(np.max(y_t) - np.min(y_t))
    nrmse = float(rmse / t_range) if t_range > 1e-9 else None

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "nrmse": round(nrmse, 4) if nrmse is not None else None,
    }


def calculate_performance_delta(
    complete_metrics: Mapping[str, float | None],
    imputed_metrics: Mapping[str, float | None],
) -> dict[str, float | None]:
    """Calculate absolute performance difference between imputed and complete-data baseline.

    For all metrics:
        delta = imputed_metric - complete_metric

    A negative delta for higher-is-better metrics (e.g. F1) indicates degradation.
    A positive delta for lower-is-better metrics (e.g. RMSE) indicates degradation.
    """
    deltas: dict[str, float | None] = {}
    for metric_name, comp_val in complete_metrics.items():
        imp_val = imputed_metrics.get(metric_name)
        if comp_val is not None and imp_val is not None:
            deltas[metric_name] = round(float(imp_val - comp_val), 4)
        else:
            deltas[metric_name] = None
    return deltas


def calculate_performance_recovery(
    complete_val: float | None,
    baseline_missing_val: float | None,
    imputed_val: float | None,
    metric_name: str = "f1",
) -> float | None:
    r"""Calculate the percentage of complete-data performance recovered by imputation.

    Mathematical Definition:
        For Higher-Is-Better (F1, Accuracy, ROC-AUC, R2):
            $$\text{Recovery} = \frac{P_{\text{imputed}} - P_{\text{baseline\_missing}}}{P_{\text{complete}} - P_{\text{baseline\_missing}}} \times 100\%$$

        For Lower-Is-Better (RMSE, MAE):
            $$\text{Recovery} = \frac{P_{\text{baseline\_missing}} - P_{\text{imputed}}}{P_{\text{baseline\_missing}} - P_{\text{complete}}} \times 100\%$$

    Args:
        complete_val: Performance on idealized complete dataset.
        baseline_missing_val: Performance on unmitigated / naive baseline imputed dataset.
        imputed_val: Performance on candidate imputed dataset.
        metric_name: Identifier of the metric being evaluated.

    Returns:
        Percentage of lost performance recovered (can be negative if worse than baseline).
    """
    if complete_val is None or baseline_missing_val is None or imputed_val is None:
        return None

    is_higher_better = metric_name.lower() in HIGHER_IS_BETTER_METRICS

    if is_higher_better:
        denominator = complete_val - baseline_missing_val
        numerator = imputed_val - baseline_missing_val
    else:
        denominator = baseline_missing_val - complete_val
        numerator = baseline_missing_val - imputed_val

    if abs(denominator) < 1e-9:
        # No gap between complete and baseline
        return 100.0 if abs(imputed_val - complete_val) < 1e-9 else 0.0

    recovery_pct = (numerator / denominator) * 100.0
    return round(float(recovery_pct), 2)


def calculate_group_downstream_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    group_series: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series | None = None,
    task_type: DownstreamTaskType = DownstreamTaskType.CLASSIFICATION,
    minimum_group_size: int = 5,
) -> dict[str, dict[str, Any]]:
    """Compute downstream ML metrics sliced by demographic or customer segment groups.

    Args:
        y_true: Target values on test partition.
        y_pred: Predicted values on test partition.
        group_series: Protected/demographic group labels for test partition.
        y_prob: Optional predicted probabilities.
        task_type: Classification or Regression.
        minimum_group_size: Threshold below which group is marked small/suppressed.

    Returns:
        Mapping from group_value to dictionary with sample_count, is_small_group, and metrics.
    """
    groups = pd.Series(group_series).fillna("Unknown").astype(str).values
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    y_pr = np.asarray(y_prob).ravel() if y_prob is not None else None

    unique_groups = np.unique(groups)
    results: dict[str, dict[str, Any]] = {}

    for g in unique_groups:
        mask = groups == g
        g_count = int(np.sum(mask))
        is_small = g_count < minimum_group_size

        if g_count == 0:
            continue

        g_yt = y_t[mask]
        g_yp = y_p[mask]
        g_ypr = y_pr[mask] if y_pr is not None else None

        if task_type == DownstreamTaskType.CLASSIFICATION:
            m = calculate_classification_metrics(g_yt, g_yp, g_ypr)
        else:
            m = calculate_regression_metrics(g_yt, g_yp)

        results[str(g)] = {
            "sample_count": g_count,
            "is_small_group": is_small,
            "metrics": m,
        }

    return results


def calculate_group_disparity(
    group_metrics: dict[str, dict[str, Any]],
    metric_name: str = "f1",
) -> dict[str, float | None]:
    """Calculate maximum absolute disparity and ratio across demographic groups.

    Args:
        group_metrics: Output from calculate_group_downstream_metrics.
        metric_name: Metric to compute disparity for.

    Returns:
        Dictionary with max_disparity, min_value, max_value, and disparity_ratio.
    """
    valid_vals: list[float] = []
    for _g, data in group_metrics.items():
        v = data["metrics"].get(metric_name)
        if v is not None:
            valid_vals.append(float(v))

    if not valid_vals:
        return {
            "max_disparity": None,
            "min_value": None,
            "max_value": None,
            "disparity_ratio": None,
        }

    min_v = float(min(valid_vals))
    max_v = float(max(valid_vals))
    abs_disp = max_v - min_v
    disp_ratio = (min_v / max_v) if max_v > 1e-9 else (1.0 if min_v == max_v else 0.0)

    return {
        "max_disparity": round(abs_disp, 4),
        "min_value": round(min_v, 4),
        "max_value": round(max_v, 4),
        "disparity_ratio": round(disp_ratio, 4),
    }


def calculate_imputation_downstream_correlation(
    imputation_maes: list[float],
    downstream_f1s: list[float],
) -> dict[str, float | None]:
    """Calculate Spearman rank correlation between imputation error and downstream performance.

    Evaluates whether lower imputation MAE reliably predicts higher downstream F1 score.
    """
    if len(imputation_maes) < 3 or len(downstream_f1s) < 3:
        return {
            "spearman_rho": None,
            "p_value": None,
        }

    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr_res = spearmanr(imputation_maes, downstream_f1s)
            rho = (
                float(corr_res.statistic) if hasattr(corr_res, "statistic") else float(corr_res[0])
            )
            p_val = float(corr_res.pvalue) if hasattr(corr_res, "pvalue") else float(corr_res[1])
        return {
            "spearman_rho": round(rho, 4) if not np.isnan(rho) else None,
            "p_value": round(p_val, 4) if not np.isnan(p_val) else None,
        }
    except Exception:
        return {
            "spearman_rho": None,
            "p_value": None,
        }
