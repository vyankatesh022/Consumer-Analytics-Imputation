"""Statistical evaluation metrics for missing data imputation benchmarking.

Provides numerical (MAE, RMSE, NRMSE) and categorical metrics with zero-division protection,
shape alignment validation, and explicit tracking of missing/invalid predictions.
"""

import numpy as np
import pandas as pd

from missing_data_platform.exceptions import EvaluationError


def validate_and_filter_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Validate alignment and filter valid prediction pairs from missing/invalid ones.

    Args:
        y_true: Ground-truth series or array.
        y_pred: Imputed/predicted series or array.

    Returns:
        tuple[np.ndarray, np.ndarray, int, int]:
            - clean_y_true: numpy array of valid ground truth values.
            - clean_y_pred: numpy array of valid predicted values.
            - missing_pred_count: count of predictions that were NaN/None.
            - invalid_pred_count: count of predictions that were Infinite or invalid.

    Raises:
        EvaluationError: If lengths mismatch or ground-truth is empty.
    """
    arr_true = np.asarray(y_true)
    arr_pred = np.asarray(y_pred)

    if arr_true.shape != arr_pred.shape:
        raise EvaluationError(
            f"Shape mismatch between ground truth {arr_true.shape} and predictions {arr_pred.shape}.",
            context={"true_shape": arr_true.shape, "pred_shape": arr_pred.shape},
        )

    if arr_true.size == 0:
        raise EvaluationError("Cannot compute evaluation metrics on 0 evaluated cells.")

    # Check for NaN / None predictions
    if pd.api.types.is_numeric_dtype(arr_pred):
        is_missing = np.isnan(arr_pred)
        is_inf = np.isinf(arr_pred)
    else:
        is_missing = pd.isna(arr_pred)
        is_inf = np.zeros(arr_pred.shape, dtype=bool)

    missing_count = int(np.sum(is_missing))
    invalid_count = int(np.sum(is_inf))

    valid_mask = ~(is_missing | is_inf)
    clean_true = arr_true[valid_mask]
    clean_pred = arr_pred[valid_mask]

    return clean_true, clean_pred, missing_count, invalid_count


def calculate_mae(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """Calculate Mean Absolute Error (MAE) strictly over valid prediction pairs.

    MAE = mean(|y_true - y_pred|)

    Raises:
        EvaluationError: If no valid pairs exist or input is invalid.
    """
    clean_true, clean_pred, _, _ = validate_and_filter_predictions(y_true, y_pred)
    if clean_true.size == 0:
        raise EvaluationError("No valid numerical prediction pairs available to compute MAE.")

    return float(np.mean(np.abs(clean_true.astype(float) - clean_pred.astype(float))))


def calculate_rmse(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """Calculate Root Mean Squared Error (RMSE) strictly over valid prediction pairs.

    RMSE = sqrt(mean((y_true - y_pred)^2))

    Raises:
        EvaluationError: If no valid pairs exist or input is invalid.
    """
    clean_true, clean_pred, _, _ = validate_and_filter_predictions(y_true, y_pred)
    if clean_true.size == 0:
        raise EvaluationError("No valid numerical prediction pairs available to compute RMSE.")

    diff = clean_true.astype(float) - clean_pred.astype(float)
    return float(np.sqrt(np.mean(diff**2)))


def calculate_nrmse(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """Calculate Normalized Root Mean Squared Error (NRMSE = RMSE / std(y_true)).

    Falls back to (RMSE / range(y_true)) or 0.0 if standard deviation is 0.
    """
    rmse = calculate_rmse(y_true, y_pred)
    clean_true, _, _, _ = validate_and_filter_predictions(y_true, y_pred)

    std_val = float(np.std(clean_true.astype(float)))
    if std_val > 1e-9:
        return float(rmse / std_val)

    val_range = float(np.ptp(clean_true.astype(float)))
    if val_range > 1e-9:
        return float(rmse / val_range)

    return 0.0 if rmse < 1e-9 else float(rmse)


def calculate_accuracy(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    """Calculate exact match accuracy for categorical feature imputation.

    Accuracy = mean(y_true == y_pred)
    """
    clean_true, clean_pred, _, _ = validate_and_filter_predictions(y_true, y_pred)
    if clean_true.size == 0:
        raise EvaluationError(
            "No valid categorical prediction pairs available to compute Accuracy."
        )

    return float(np.mean(clean_true == clean_pred))
