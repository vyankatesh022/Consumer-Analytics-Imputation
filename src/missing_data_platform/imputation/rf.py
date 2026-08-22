"""Random Forest Imputation Layer.

Implements multi-feature nonlinear regression-based imputation where each missing variable
is modeled as a function of eligible observed features using Random Forest regressors.
"""

from dataclasses import dataclass, field
from typing import Self

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from missing_data_platform.exceptions import ConfigurationError, DataQualityError, ImputationError
from missing_data_platform.imputation.base import BaseImputer
from missing_data_platform.imputation.config import BaselineStrategy
from missing_data_platform.imputation.report import FeatureImputationMetric
from missing_data_platform.ingestion.contract import DataType, RawDataContract


@dataclass
class RandomForestImputationConfig:
    """Configuration parameters for Random Forest Imputation."""

    n_estimators: int = 100
    max_depth: int | None = 15
    min_samples_leaf: int = 1
    max_features: float | int | str | None = "sqrt"
    random_seed: int = 42
    n_jobs: int = 1
    target_features: list[str] | None = None
    protected_features: list[str] = field(
        default_factory=lambda: ["customer_id", "purchase_next_month"]
    )
    fallback_strategy: BaselineStrategy = BaselineStrategy.MEDIAN

    def __post_init__(self) -> None:
        """Validate Random Forest configuration invariants and resource limits."""
        if self.n_estimators < 1 or self.n_estimators > 500:
            raise ConfigurationError(
                f"Invalid n_estimators: {self.n_estimators}. Must be between 1 and 500.",
                context={"n_estimators": self.n_estimators},
            )

        if self.max_depth is not None and (self.max_depth < 1 or self.max_depth > 50):
            raise ConfigurationError(
                f"Invalid max_depth: {self.max_depth}. Must be between 1 and 50 or None.",
                context={"max_depth": self.max_depth},
            )

        if self.min_samples_leaf < 1:
            raise ConfigurationError(
                f"Invalid min_samples_leaf: {self.min_samples_leaf}. Must be >= 1.",
                context={"min_samples_leaf": self.min_samples_leaf},
            )

        if self.n_jobs < -1 or self.n_jobs == 0:
            raise ConfigurationError(
                f"Invalid n_jobs: {self.n_jobs}. Must be >= 1 or -1.",
                context={"n_jobs": self.n_jobs},
            )

        if self.target_features is not None:
            conflicts = [col for col in self.target_features if col in self.protected_features]
            if conflicts:
                raise ConfigurationError(
                    f"Protected features cannot be targeted for Random Forest imputation: {conflicts}",
                    context={"conflicts": conflicts},
                )


class RandomForestImputerModel(BaseImputer):
    """Production Random Forest Imputer with target-specific regressors and leakage isolation."""

    def __init__(
        self,
        config: RandomForestImputationConfig | None = None,
        contract: RawDataContract | None = None,
    ) -> None:
        super().__init__()
        self.config = config or RandomForestImputationConfig()
        self.contract = contract or RawDataContract.default_consumer_contract()
        self.numeric_features: list[str] = []
        self._models: dict[str, RandomForestRegressor] = {}
        self._fallbacks: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> Self:
        """Fit target-specific Random Forest regressors strictly from training data.

        Raises:
            DataQualityError: If input DataFrame is empty.
            ImputationError: If no numeric features are available or all values are missing.
        """
        if df.empty:
            raise DataQualityError("Cannot fit RandomForestImputerModel on an empty DataFrame.")

        # Determine eligible numeric features for Random Forest modeling
        if self.config.target_features is not None:
            candidate_cols = self.config.target_features
        else:
            candidate_cols = [
                col
                for col in df.columns
                if col not in self.config.protected_features
                and col != self.contract.id_column
                and col != self.contract.target_column
            ]

        self.numeric_features = []
        for col in candidate_cols:
            if col not in df.columns:
                continue
            col_defn = self.contract.get_column(col)
            is_numeric = pd.api.types.is_numeric_dtype(df[col]) or (
                col_defn is not None and col_defn.data_type in (DataType.FLOAT, DataType.INTEGER)
            )
            if is_numeric:
                self.numeric_features.append(col)

        # Ensure deterministic feature ordering
        self.numeric_features = sorted(self.numeric_features)

        if not self.numeric_features:
            raise ImputationError(
                "No valid numeric features found for Random Forest regression modeling."
            )

        numeric_df = df[self.numeric_features].apply(pd.to_numeric, errors="coerce")

        # Compute univariate preliminary fallbacks strictly from training observed values
        self._fallbacks.clear()
        for col in self.numeric_features:
            valid_vals = numeric_df[col].dropna()
            if valid_vals.empty:
                raise ImputationError(
                    f"Feature '{col}' has 0 observed values to fit Random Forest models.",
                    context={"feature": col},
                )
            if self.config.fallback_strategy == BaselineStrategy.MEAN:
                self._fallbacks[col] = float(valid_vals.mean())
            else:
                self._fallbacks[col] = float(valid_vals.median())

        # Fit target-specific Random Forest models
        self._models.clear()
        for target_col in self.numeric_features:
            predictor_cols = [col for col in self.numeric_features if col != target_col]
            if not predictor_cols:
                # If only one numeric feature exists in the dataset, fallback will be used
                continue

            observed_mask = numeric_df[target_col].notna()
            obs_count = int(observed_mask.sum())

            # Validate sufficient training rows
            if obs_count < self.config.min_samples_leaf:
                raise ImputationError(
                    f"Feature '{target_col}' has insufficient observed values ({obs_count}) "
                    f"to satisfy min_samples_leaf={self.config.min_samples_leaf}.",
                    context={"feature": target_col, "observed_count": obs_count},
                )

            # Build predictor matrix for observed rows
            X_train = numeric_df.loc[observed_mask, predictor_cols].copy()
            # Handle missing values in predictors using fitted preliminary training fallbacks
            for p_col in predictor_cols:
                if X_train[p_col].isna().any():
                    X_train[p_col] = X_train[p_col].fillna(self._fallbacks[p_col])

            y_train = numeric_df.loc[observed_mask, target_col]

            rf = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                max_features=self.config.max_features,
                random_state=self.config.random_seed,
                n_jobs=self.config.n_jobs,
            )
            rf.fit(X_train, y_train)
            self._models[target_col] = rf

        self.imputation_parameters = {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "min_samples_leaf": self.config.min_samples_leaf,
            "max_features": self.config.max_features,
            "random_seed": self.config.random_seed,
            "n_jobs": self.config.n_jobs,
            "numeric_features": self.numeric_features,
            "trained_models": list(self._models.keys()),
            "fallbacks": self._fallbacks,
        }
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing numeric values using pre-fitted Random Forest models."""
        imputed_df, _ = self.transform_with_metrics(df)
        return imputed_df

    def transform_with_metrics(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[FeatureImputationMetric]]:
        """Impute missing values and generate before/after feature tracking metrics.

        Raises:
            ImputationError: If imputer is not fitted.
        """
        if not self.is_fitted:
            raise ImputationError(
                "RandomForestImputerModel must be fitted before calling transform."
            )

        imputed_df = df.copy(deep=True)
        metrics: list[FeatureImputationMetric] = []

        numeric_df = imputed_df[self.numeric_features].apply(pd.to_numeric, errors="coerce")

        for target_col in self.numeric_features:
            missing_before = int(numeric_df[target_col].isna().sum())

            if missing_before > 0:
                predictor_cols = [col for col in self.numeric_features if col != target_col]
                missing_mask = numeric_df[target_col].isna()

                if predictor_cols and target_col in self._models:
                    # Prepare predictor matrix for missing rows
                    X_miss = numeric_df.loc[missing_mask, predictor_cols].copy()
                    # Apply fitted fallbacks to any missing predictors
                    for p_col in predictor_cols:
                        if X_miss[p_col].isna().any():
                            X_miss[p_col] = X_miss[p_col].fillna(self._fallbacks[p_col])

                    predictions = self._models[target_col].predict(X_miss)
                    imputed_df.loc[missing_mask, target_col] = predictions
                    numeric_df.loc[missing_mask, target_col] = predictions

                # If any residual NaNs remain, fill with training fallback
                residual_nans = int(imputed_df[target_col].isna().sum())
                if residual_nans > 0:
                    fallback_val = self._fallbacks.get(target_col, 0.0)
                    imputed_df[target_col] = imputed_df[target_col].fillna(fallback_val)
                    numeric_df[target_col] = numeric_df[target_col].fillna(fallback_val)

            missing_after = int(imputed_df[target_col].isna().sum())
            imputed_count = missing_before - missing_after

            metrics.append(
                FeatureImputationMetric(
                    feature_name=target_col,
                    strategy_applied=BaselineStrategy.CONSTANT,  # Custom RF marker
                    statistic_value=f"RandomForest(trees={self.config.n_estimators})",
                    missing_before=missing_before,
                    imputed_count=imputed_count,
                    missing_after=missing_after,
                )
            )

        return imputed_df, metrics
