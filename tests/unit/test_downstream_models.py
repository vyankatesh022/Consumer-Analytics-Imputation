"""Unit tests for downstream model wrappers and preprocessing pipelines."""

import numpy as np
import pandas as pd
import pytest

from missing_data_platform.downstream.config import (
    DownstreamConfig,
    DownstreamModelType,
    DownstreamTaskType,
)
from missing_data_platform.downstream.models import DownstreamModelWrapper
from missing_data_platform.exceptions import DataQualityError, ModelTrainingError


def test_downstream_model_wrapper_fit_predict_rf() -> None:
    """Assert RandomForest downstream wrapper trains and predicts accurately."""
    df = pd.DataFrame(
        {
            "age": [25, 30, 45, 50, 22, 60, 35, 40],
            "income": [50000.0, 60000.0, 80000.0, 95000.0, 40000.0, 110000.0, 70000.0, 75000.0],
            "customer_segment": [
                "Bronze",
                "Silver",
                "Gold",
                "Gold",
                "Bronze",
                "Gold",
                "Silver",
                "Silver",
            ],
        }
    )
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])

    config = DownstreamConfig(
        model_type=DownstreamModelType.RANDOM_FOREST,
        task_type=DownstreamTaskType.CLASSIFICATION,
    )
    wrapper = DownstreamModelWrapper(config=config)
    wrapper.fit(df, y)

    preds = wrapper.predict(df)
    probs = wrapper.predict_proba(df)

    assert len(preds) == len(df)
    assert probs is not None
    assert len(probs) == len(df)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_downstream_model_wrapper_logistic_regression() -> None:
    """Assert LogisticRegression downstream wrapper works for classification."""
    df = pd.DataFrame(
        {
            "age": [25, 30, 45, 50, 22, 60, 35, 40],
            "income": [50000.0, 60000.0, 80000.0, 95000.0, 40000.0, 110000.0, 70000.0, 75000.0],
            "gender": ["Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male"],
        }
    )
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])

    config = DownstreamConfig(
        model_type=DownstreamModelType.LOGISTIC_REGRESSION,
        task_type=DownstreamTaskType.CLASSIFICATION,
    )
    wrapper = DownstreamModelWrapper(config=config)
    wrapper.fit(df, y)

    preds = wrapper.predict(df)
    probs = wrapper.predict_proba(df)
    assert len(preds) == len(df)
    assert probs is not None


def test_downstream_model_wrapper_gradient_boosting_and_ridge() -> None:
    """Assert GradientBoosting and Ridge wrappers function correctly."""
    df = pd.DataFrame(
        {
            "age": [25, 30, 45, 50, 22, 60, 35, 40],
            "income": [50000.0, 60000.0, 80000.0, 95000.0, 40000.0, 110000.0, 70000.0, 75000.0],
        }
    )
    y_clf = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_reg = np.array([100.0, 120.0, 200.0, 250.0, 90.0, 300.0, 180.0, 190.0])

    # Gradient Boosting Classifier
    gb_clf = DownstreamModelWrapper(
        config=DownstreamConfig(
            model_type=DownstreamModelType.GRADIENT_BOOSTING,
            task_type=DownstreamTaskType.CLASSIFICATION,
        )
    )
    gb_clf.fit(df, y_clf)
    assert len(gb_clf.predict(df)) == len(df)

    # Ridge Classifier
    ridge_clf = DownstreamModelWrapper(
        config=DownstreamConfig(
            model_type=DownstreamModelType.RIDGE,
            task_type=DownstreamTaskType.CLASSIFICATION,
        )
    )
    ridge_clf.fit(df, y_clf)
    assert len(ridge_clf.predict(df)) == len(df)

    # Random Forest Regressor
    rf_reg = DownstreamModelWrapper(
        config=DownstreamConfig(
            model_type=DownstreamModelType.RANDOM_FOREST,
            task_type=DownstreamTaskType.REGRESSION,
            primary_metric="rmse",
        )
    )
    rf_reg.fit(df, y_reg)
    assert len(rf_reg.predict(df)) == len(df)
    assert rf_reg.predict_proba(df) is None


def test_downstream_model_wrapper_unfitted_predict_error() -> None:
    """Assert ModelTrainingError when predicting with unfitted wrapper."""
    wrapper = DownstreamModelWrapper()
    df = pd.DataFrame({"age": [25, 30]})
    with pytest.raises(ModelTrainingError, match="not fitted"):
        wrapper.predict(df)
    with pytest.raises(ModelTrainingError, match="not fitted"):
        wrapper.predict_proba(df)


def test_downstream_model_wrapper_target_leakage_guard() -> None:
    """Assert ModelTrainingError when target column is accidentally included in features."""
    df = pd.DataFrame(
        {
            "age": [25, 30, 45, 50],
            "purchase_next_month": [0, 0, 1, 1],
        }
    )
    y = np.array([0, 0, 1, 1])

    wrapper = DownstreamModelWrapper()
    with pytest.raises(ModelTrainingError, match="Target leakage is strictly prohibited"):
        wrapper.fit(df, y)


def test_downstream_model_wrapper_empty_dataframe_error() -> None:
    """Assert DataQualityError when feature dataframe is empty."""
    df = pd.DataFrame()
    y = np.array([])

    wrapper = DownstreamModelWrapper()
    with pytest.raises(DataQualityError, match="empty feature DataFrame"):
        wrapper.fit(df, y)
