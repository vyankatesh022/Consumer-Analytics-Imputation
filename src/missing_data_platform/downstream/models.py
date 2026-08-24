"""Downstream machine learning model wrappers and preprocessing pipelines.

Provides leakage-free preprocessing (categorical encoding and numerical scaling)
coupled with standard scikit-learn estimators for classification and regression tasks.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from missing_data_platform.downstream.config import (
    DownstreamConfig,
    DownstreamModelType,
    DownstreamTaskType,
)
from missing_data_platform.exceptions import DataQualityError, ModelTrainingError
from missing_data_platform.ingestion.contract import DataType, RawDataContract
from missing_data_platform.logging import get_logger

logger = get_logger("downstream.models")


class DownstreamModelWrapper:
    """Production wrapper encapsulating feature preprocessing and downstream estimator."""

    def __init__(
        self,
        config: DownstreamConfig | None = None,
        contract: RawDataContract | None = None,
    ) -> None:
        self.config = config or DownstreamConfig()
        self.contract = contract or RawDataContract.default_consumer_contract()
        self.pipeline: Pipeline | None = None
        self.is_fitted: bool = False
        self.feature_names_in_: list[str] = []
        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = []

    def _build_estimator(self) -> Any:
        """Instantiate configured scikit-learn estimator."""
        m_type = self.config.model_type
        t_type = self.config.task_type
        seed = self.config.random_seed
        params = self.config.model_params.copy()

        if t_type == DownstreamTaskType.CLASSIFICATION:
            if m_type == DownstreamModelType.RANDOM_FOREST:
                defaults = {"n_estimators": 100, "max_depth": 10, "random_state": seed}
                defaults.update(params)
                return RandomForestClassifier(**defaults)
            elif m_type == DownstreamModelType.LOGISTIC_REGRESSION:
                defaults = {"max_iter": 1000, "random_state": seed}
                defaults.update(params)
                return LogisticRegression(**defaults)
            elif m_type == DownstreamModelType.GRADIENT_BOOSTING:
                defaults = {"n_estimators": 100, "random_state": seed}
                defaults.update(params)
                return GradientBoostingClassifier(**defaults)
            elif m_type == DownstreamModelType.RIDGE:
                defaults = {"random_state": seed}
                defaults.update(params)
                return RidgeClassifier(**defaults)
            else:
                raise ModelTrainingError(
                    f"Unsupported classification model type: {m_type}",
                    context={"model_type": m_type},
                )
        else:  # Regression
            if m_type == DownstreamModelType.RANDOM_FOREST:
                defaults = {"n_estimators": 100, "max_depth": 10, "random_state": seed}
                defaults.update(params)
                return RandomForestRegressor(**defaults)
            elif m_type == DownstreamModelType.GRADIENT_BOOSTING:
                defaults = {"n_estimators": 100, "random_state": seed}
                defaults.update(params)
                return GradientBoostingRegressor(**defaults)
            elif m_type == DownstreamModelType.RIDGE:
                defaults = {"random_state": seed}
                defaults.update(params)
                from sklearn.linear_model import Ridge

                return Ridge(**defaults)
            else:
                raise ModelTrainingError(
                    f"Unsupported regression model type: {m_type}",
                    context={"model_type": m_type},
                )

    def _inspect_and_split_features(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Identify numeric and categorical features strictly excluding ID and Target columns."""
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []

        for col in sorted(df.columns):
            if col in self.config.protected_features or col in (
                self.contract.id_column,
                self.contract.target_column,
            ):
                continue

            col_defn = self.contract.get_column(col)
            is_num = pd.api.types.is_numeric_dtype(df[col]) or (
                col_defn is not None and col_defn.data_type in (DataType.FLOAT, DataType.INTEGER)
            )

            if is_num:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        return numeric_cols, categorical_cols

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "DownstreamModelWrapper":
        """Fit feature preprocessors and downstream estimator strictly on training partition."""
        if X.empty:
            raise DataQualityError("Cannot fit downstream model on empty feature DataFrame.")

        y_arr = np.asarray(y).ravel()
        if len(y_arr) != len(X):
            raise ModelTrainingError(
                f"Feature row count ({len(X)}) does not match target count ({len(y_arr)})."
            )

        # Enforce target isolation
        if self.contract.target_column in X.columns:
            raise ModelTrainingError(
                f"Target column '{self.contract.target_column}' detected in feature matrix! "
                "Target leakage is strictly prohibited."
            )
        if self.contract.id_column in X.columns:
            X_clean = X.drop(columns=[self.contract.id_column])
        else:
            X_clean = X

        self.feature_names_in_ = list(X_clean.columns)
        self.numeric_features, self.categorical_features = self._inspect_and_split_features(X_clean)

        transformers = []
        if self.numeric_features:
            num_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("num", num_pipe, self.numeric_features))

        if self.categorical_features:
            cat_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )
            transformers.append(("cat", cat_pipe, self.categorical_features))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        estimator = self._build_estimator()

        self.pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("estimator", estimator),
            ]
        )

        try:
            self.pipeline.fit(X_clean, y_arr)
            self.is_fitted = True
            logger.info(
                "Downstream model fitted successfully",
                model_type=self.config.model_type.value,
                task_type=self.config.task_type.value,
                num_features=len(self.numeric_features),
                cat_features=len(self.categorical_features),
                total_samples=len(X_clean),
            )
        except Exception as e:
            raise ModelTrainingError(
                f"Failed to fit downstream model: {e}",
                context={"model_type": self.config.model_type.value},
            ) from e

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for evaluation samples."""
        if not self.is_fitted or self.pipeline is None:
            raise ModelTrainingError("DownstreamModelWrapper is not fitted.")

        X_clean = (
            X.drop(columns=[self.contract.id_column]) if self.contract.id_column in X.columns else X
        )
        if self.contract.target_column in X_clean.columns:
            X_clean = X_clean.drop(columns=[self.contract.target_column])

        return np.asarray(self.pipeline.predict(X_clean))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        """Generate positive class probabilities for classification tasks."""
        if not self.is_fitted or self.pipeline is None:
            raise ModelTrainingError("DownstreamModelWrapper is not fitted.")

        if self.config.task_type != DownstreamTaskType.CLASSIFICATION:
            return None

        estimator = self.pipeline.named_steps["estimator"]
        if not hasattr(estimator, "predict_proba"):
            return None

        X_clean = (
            X.drop(columns=[self.contract.id_column]) if self.contract.id_column in X.columns else X
        )
        if self.contract.target_column in X_clean.columns:
            X_clean = X_clean.drop(columns=[self.contract.target_column])

        try:
            probs = self.pipeline.predict_proba(X_clean)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return np.asarray(probs[:, 1])
            elif probs.ndim == 1:
                return np.asarray(probs)
            return None
        except Exception:
            return None
