"""Iterative / Multivariate Imputation Layer (MICE / Chained Equations).

Implements round-robin feature-by-feature predictive modeling where each missing variable
is modeled as a function of all other observed features using Bayesian Ridge regression.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer as SklearnIterativeImputer
from sklearn.linear_model import BayesianRidge

from missing_data_platform.exceptions import ConfigurationError, DataQualityError, ImputationError
from missing_data_platform.imputation.base import BaseImputer
from missing_data_platform.imputation.config import BaselineStrategy
from missing_data_platform.imputation.report import FeatureImputationMetric
from missing_data_platform.ingestion.contract import DataType, RawDataContract


class InitialStrategy(StrEnum):
    """Initial univariate fill strategy prior to iterative round-robin updates."""

    MEAN = "mean"
    MEDIAN = "median"
    MOST_FREQUENT = "most_frequent"


class ImputationOrder(StrEnum):
    """Order in which features are updated during each iteration cycle."""

    ASCENDING = "ascending"  # Features with fewest missing values updated first
    DESCENDING = "descending"  # Features with most missing values updated first
    ROMAN = "roman"  # Left to right
    ARABIC = "arabic"  # Right to left
    RANDOM = "random"  # Random order per iteration


@dataclass
class IterativeImputationConfig:
    """Configuration parameters for Iterative Multivariate Imputation."""

    max_iter: int = 10
    tol: float = 1e-3
    initial_strategy: InitialStrategy = InitialStrategy.MEAN
    imputation_order: ImputationOrder = ImputationOrder.ASCENDING
    random_seed: int = 42
    target_features: list[str] | None = None
    protected_features: list[str] = field(
        default_factory=lambda: ["customer_id", "purchase_next_month"]
    )
    fallback_strategy: BaselineStrategy = BaselineStrategy.MEDIAN

    def __post_init__(self) -> None:
        """Validate iterative configuration invariants."""
        if self.max_iter < 1:
            raise ConfigurationError(
                f"Invalid max_iter: {self.max_iter}. Must be >= 1.",
                context={"max_iter": self.max_iter},
            )

        if self.tol < 0.0:
            raise ConfigurationError(
                f"Invalid tol: {self.tol}. Must be >= 0.0.",
                context={"tol": self.tol},
            )

        if self.target_features is not None:
            conflicts = [col for col in self.target_features if col in self.protected_features]
            if conflicts:
                raise ConfigurationError(
                    f"Protected features cannot be targeted for iterative imputation: {conflicts}",
                    context={"conflicts": conflicts},
                )


class IterativeImputerModel(BaseImputer):
    """Production Iterative Imputer implementing chained predictive regression equations."""

    def __init__(
        self,
        config: IterativeImputationConfig | None = None,
        contract: RawDataContract | None = None,
    ) -> None:
        super().__init__()
        self.config = config or IterativeImputationConfig()
        self.contract = contract or RawDataContract.default_consumer_contract()
        self.numeric_features: list[str] = []
        self._iterative_imputer: SklearnIterativeImputer | None = None
        self._fallbacks: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> Self:
        """Fit multivariate iterative regression models strictly from training data.

        Raises:
            DataQualityError: If input DataFrame is empty.
            ImputationError: If no numeric features are available or all cells are missing.
        """
        if df.empty:
            raise DataQualityError("Cannot fit IterativeImputerModel on an empty DataFrame.")

        # Determine eligible numeric features for multivariate modeling
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

        if not self.numeric_features:
            raise ImputationError(
                "No valid numeric features found for iterative multivariate modeling."
            )

        numeric_df = df[self.numeric_features].apply(pd.to_numeric, errors="coerce")

        # Compute univariate fallbacks strictly from training observed values
        self._fallbacks.clear()
        for col in self.numeric_features:
            valid_vals = numeric_df[col].dropna()
            if valid_vals.empty:
                raise ImputationError(
                    f"Feature '{col}' has 0 observed values to fit iterative regression models.",
                    context={"feature": col},
                )
            if self.config.fallback_strategy == BaselineStrategy.MEAN:
                self._fallbacks[col] = float(valid_vals.mean())
            else:
                self._fallbacks[col] = float(valid_vals.median())

        estimator = BayesianRidge()
        self._iterative_imputer = SklearnIterativeImputer(
            estimator=estimator,
            max_iter=self.config.max_iter,
            tol=self.config.tol,
            initial_strategy=self.config.initial_strategy.value,
            imputation_order=self.config.imputation_order.value,
            random_state=self.config.random_seed,
        )

        self._iterative_imputer.fit(numeric_df)

        actual_iterations = getattr(self._iterative_imputer, "n_iter_", self.config.max_iter)
        converged = actual_iterations < self.config.max_iter

        self.imputation_parameters = {
            "max_iter": self.config.max_iter,
            "actual_iterations": actual_iterations,
            "converged": converged,
            "tol": self.config.tol,
            "initial_strategy": self.config.initial_strategy.value,
            "imputation_order": self.config.imputation_order.value,
            "random_seed": self.config.random_seed,
            "numeric_features": self.numeric_features,
            "fallbacks": self._fallbacks,
        }
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing numeric values using pre-fitted iterative chained models."""
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
        if not self.is_fitted or self._iterative_imputer is None:
            raise ImputationError("IterativeImputerModel must be fitted before calling transform.")

        imputed_df = df.copy(deep=True)
        metrics: list[FeatureImputationMetric] = []

        numeric_df = imputed_df[self.numeric_features].apply(pd.to_numeric, errors="coerce")
        missing_before_counts = {
            col: int(numeric_df[col].isna().sum()) for col in self.numeric_features
        }

        # Apply pre-fitted iterative model
        imputed_numeric_matrix = self._iterative_imputer.transform(numeric_df)

        imputed_numeric_df = pd.DataFrame(
            imputed_numeric_matrix,
            index=df.index,
            columns=self.numeric_features,
        )

        for col in self.numeric_features:
            missing_before = missing_before_counts[col]
            imputed_df[col] = imputed_numeric_df[col]

            # Handle any residual NaNs with fallback
            residual_nans = int(imputed_df[col].isna().sum())
            if residual_nans > 0:
                fallback_val = self._fallbacks.get(col, 0.0)
                imputed_df[col] = imputed_df[col].fillna(fallback_val)

            missing_after = int(imputed_df[col].isna().sum())
            imputed_count = missing_before - missing_after

            metrics.append(
                FeatureImputationMetric(
                    feature_name=col,
                    strategy_applied=BaselineStrategy.CONSTANT,  # MICE iterative marker
                    statistic_value=f"MICE(iter={self.imputation_parameters.get('actual_iterations', self.config.max_iter)})",
                    missing_before=missing_before,
                    imputed_count=imputed_count,
                    missing_after=missing_after,
                )
            )

        return imputed_df, metrics
