"""K-Nearest Neighbors (KNN) Imputation Layer.

Implements multi-feature similarity-based imputation with robust feature scaling,
zero-division protection, leakage-safe fit/transform lifecycle, and target/ID guardrails.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

import pandas as pd
from sklearn.impute import KNNImputer as SklearnKNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from missing_data_platform.exceptions import ConfigurationError, DataQualityError, ImputationError
from missing_data_platform.imputation.base import BaseImputer
from missing_data_platform.imputation.config import BaselineStrategy
from missing_data_platform.imputation.report import FeatureImputationMetric
from missing_data_platform.ingestion.contract import DataType, RawDataContract


class ScalingStrategy(StrEnum):
    """Supported feature scaling strategies prior to distance calculation."""

    STANDARD = "standard"  # Zero mean, unit variance
    MINMAX = "minmax"  # Bounded [0, 1] range
    NONE = "none"  # Raw feature scale


class KNNWeighting(StrEnum):
    """Weighting functions for neighbor contribution."""

    UNIFORM = "uniform"
    DISTANCE = "distance"


@dataclass
class KNNImputationConfig:
    """Configuration parameters for KNN Imputation."""

    n_neighbors: int = 5
    weights: KNNWeighting = KNNWeighting.UNIFORM
    scaling_strategy: ScalingStrategy = ScalingStrategy.STANDARD
    target_features: list[str] | None = None
    protected_features: list[str] = field(
        default_factory=lambda: ["customer_id", "purchase_next_month"]
    )
    fallback_strategy: BaselineStrategy = BaselineStrategy.MEDIAN

    def __post_init__(self) -> None:
        """Validate KNN configuration invariants."""
        if self.n_neighbors < 1:
            raise ConfigurationError(
                f"Invalid n_neighbors: {self.n_neighbors}. Must be >= 1.",
                context={"n_neighbors": self.n_neighbors},
            )

        if self.target_features is not None:
            conflicts = [col for col in self.target_features if col in self.protected_features]
            if conflicts:
                raise ConfigurationError(
                    f"Protected features cannot be targeted for KNN imputation: {conflicts}",
                    context={"conflicts": conflicts},
                )


class KNNImputerModel(BaseImputer):
    """Production KNN Imputer with leakage-safe scaling and neighbor estimation."""

    def __init__(
        self,
        config: KNNImputationConfig | None = None,
        contract: RawDataContract | None = None,
    ) -> None:
        super().__init__()
        self.config = config or KNNImputationConfig()
        self.contract = contract or RawDataContract.default_consumer_contract()
        self.numeric_features: list[str] = []
        self._scaler: StandardScaler | MinMaxScaler | None = None
        self._knn_imputer: SklearnKNNImputer | None = None
        self._fallbacks: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> Self:
        """Fit feature scaler and KNN neighbor representation strictly from training data.

        Raises:
            DataQualityError: If input DataFrame is empty.
            ImputationError: If no numeric features are available or all cells are missing.
        """
        if df.empty:
            raise DataQualityError("Cannot fit KNNImputerModel on an empty DataFrame.")

        # Determine eligible numeric features for distance calculation and imputation
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
                "No valid numeric features found for KNN distance calculation and imputation."
            )

        # Extract numeric submatrix
        numeric_df = df[self.numeric_features].apply(pd.to_numeric, errors="coerce")

        # Compute univariate fallbacks strictly from training observed values
        self._fallbacks.clear()
        for col in self.numeric_features:
            valid_vals = numeric_df[col].dropna()
            if valid_vals.empty:
                raise ImputationError(
                    f"Feature '{col}' has 0 observed values to fit KNN neighbor models.",
                    context={"feature": col},
                )
            if self.config.fallback_strategy == BaselineStrategy.MEAN:
                self._fallbacks[col] = float(valid_vals.mean())
            else:
                self._fallbacks[col] = float(valid_vals.median())

        # Fit Scaler
        if self.config.scaling_strategy == ScalingStrategy.STANDARD:
            self._scaler = StandardScaler()
            scaled_matrix = self._scaler.fit_transform(numeric_df)
        elif self.config.scaling_strategy == ScalingStrategy.MINMAX:
            self._scaler = MinMaxScaler()
            scaled_matrix = self._scaler.fit_transform(numeric_df)
        else:
            self._scaler = None
            scaled_matrix = numeric_df.to_numpy()

        # Fit KNN imputer
        effective_k = min(self.config.n_neighbors, len(df))
        self._knn_imputer = SklearnKNNImputer(
            n_neighbors=effective_k,
            weights=self.config.weights.value,
            metric="nan_euclidean",
        )
        self._knn_imputer.fit(scaled_matrix)

        self.imputation_parameters = {
            "n_neighbors": effective_k,
            "weights": self.config.weights.value,
            "scaling_strategy": self.config.scaling_strategy.value,
            "numeric_features": self.numeric_features,
            "fallbacks": self._fallbacks,
        }
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing numeric values using fitted scaler and KNN imputer."""
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
        if not self.is_fitted or self._knn_imputer is None:
            raise ImputationError("KNNImputerModel must be fitted before calling transform.")

        imputed_df = df.copy(deep=True)
        metrics: list[FeatureImputationMetric] = []

        # Extract numeric target matrix
        numeric_df = imputed_df[self.numeric_features].apply(pd.to_numeric, errors="coerce")

        # Record missingness before imputation
        missing_before_counts = {
            col: int(numeric_df[col].isna().sum()) for col in self.numeric_features
        }

        # Apply pre-fitted scaler (WITHOUT refitting)
        if self._scaler is not None:
            scaled_target = self._scaler.transform(numeric_df)
        else:
            scaled_target = numeric_df.to_numpy()

        # Impute missing values via KNN
        imputed_scaled = self._knn_imputer.transform(scaled_target)

        # Inverse transform back to original scale
        if self._scaler is not None:
            imputed_numeric_matrix = self._scaler.inverse_transform(imputed_scaled)
        else:
            imputed_numeric_matrix = imputed_scaled

        # Assign imputed matrix back to DataFrame
        imputed_numeric_df = pd.DataFrame(
            imputed_numeric_matrix,
            index=df.index,
            columns=self.numeric_features,
        )

        for col in self.numeric_features:
            missing_before = missing_before_counts[col]
            # Replace NaNs with KNN values
            imputed_df[col] = imputed_numeric_df[col]
            # If any residual NaN remains (e.g. observation with all features missing), apply fallback
            residual_nans = int(imputed_df[col].isna().sum())
            if residual_nans > 0:
                fallback_val = self._fallbacks.get(col, 0.0)
                imputed_df[col] = imputed_df[col].fillna(fallback_val)

            missing_after = int(imputed_df[col].isna().sum())
            imputed_count = missing_before - missing_after

            metrics.append(
                FeatureImputationMetric(
                    feature_name=col,
                    strategy_applied=BaselineStrategy.CONSTANT,  # Custom KNN marker
                    statistic_value=f"KNN(k={self.config.n_neighbors})",
                    missing_before=missing_before,
                    imputed_count=imputed_count,
                    missing_after=missing_after,
                )
            )

        return imputed_df, metrics
