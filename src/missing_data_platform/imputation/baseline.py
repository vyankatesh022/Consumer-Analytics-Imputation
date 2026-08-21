"""Baseline statistical imputer implementation (Mean, Median, Mode, Constant)."""

from typing import Self

import pandas as pd

from missing_data_platform.exceptions import DataQualityError, ImputationError
from missing_data_platform.imputation.base import BaseImputer
from missing_data_platform.imputation.config import (
    BaselineImputationConfig,
    BaselineStrategy,
)
from missing_data_platform.imputation.report import FeatureImputationMetric
from missing_data_platform.ingestion.contract import DataType, RawDataContract


class BaselineImputer(BaseImputer):
    """Baseline imputer calculating univariate statistical replacement values."""

    def __init__(
        self,
        config: BaselineImputationConfig | None = None,
        contract: RawDataContract | None = None,
    ) -> None:
        super().__init__()
        self.config = config or BaselineImputationConfig()
        self.contract = contract or RawDataContract.default_consumer_contract()
        self.target_features: list[str] = []
        self.feature_strategies: dict[str, BaselineStrategy] = {}

    def fit(self, df: pd.DataFrame) -> Self:
        """Fit univariate statistical parameters from observed values in training data.

        Raises:
            DataQualityError: If input DataFrame is empty.
            ImputationError: If an eligible column has 0 observed values to compute statistics from.
        """
        if df.empty:
            raise DataQualityError("Cannot fit BaselineImputer on an empty DataFrame.")

        # Determine target features for imputation (strictly excluding protected features)
        if self.config.target_features is not None:
            self.target_features = [
                col
                for col in self.config.target_features
                if col in df.columns and col not in self.config.protected_features
            ]
        else:
            self.target_features = [
                col
                for col in df.columns
                if col not in self.config.protected_features
                and col != self.contract.id_column
                and col != self.contract.target_column
            ]

        self.imputation_parameters.clear()
        self.feature_strategies.clear()

        for col in self.target_features:
            series = df[col]
            observed_series = series.dropna()

            col_defn = self.contract.get_column(col)
            is_numeric = pd.api.types.is_numeric_dtype(series) or (
                col_defn is not None and col_defn.data_type in (DataType.FLOAT, DataType.INTEGER)
            )

            if is_numeric:
                strategy = self.config.numeric_strategy
                self.feature_strategies[col] = strategy

                if strategy == BaselineStrategy.CONSTANT:
                    self.imputation_parameters[col] = self.config.constant_fill_value
                else:
                    if observed_series.empty:
                        raise ImputationError(
                            f"Cannot compute baseline '{strategy.value}' for column '{col}': all values are missing.",
                            context={"feature": col, "strategy": strategy.value},
                        )
                    clean_numeric = pd.to_numeric(observed_series, errors="coerce").dropna()
                    if clean_numeric.empty:
                        raise ImputationError(
                            f"Column '{col}' has no valid numeric values to compute '{strategy.value}'.",
                            context={"feature": col},
                        )

                    if strategy == BaselineStrategy.MEAN:
                        self.imputation_parameters[col] = float(clean_numeric.mean())
                    elif strategy == BaselineStrategy.MEDIAN:
                        self.imputation_parameters[col] = float(clean_numeric.median())
            else:
                # Categorical column
                strategy = self.config.categorical_strategy
                self.feature_strategies[col] = strategy

                if strategy == BaselineStrategy.CONSTANT:
                    self.imputation_parameters[col] = str(self.config.constant_fill_value)
                elif strategy == BaselineStrategy.MODE:
                    if observed_series.empty:
                        raise ImputationError(
                            f"Cannot compute mode for categorical column '{col}': all values are missing.",
                            context={"feature": col},
                        )
                    # Mode with deterministic alphabetical tie-breaking
                    mode_counts = observed_series.astype(str).value_counts()
                    max_freq = mode_counts.max()
                    top_modes = sorted(
                        [cat for cat, freq in mode_counts.items() if freq == max_freq]
                    )
                    self.imputation_parameters[col] = top_modes[0]

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in the DataFrame using pre-fitted parameters.

        Raises:
            ImputationError: If the imputer is not yet fitted.
        """
        imputed_df, _ = self.transform_with_metrics(df)
        return imputed_df

    def transform_with_metrics(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[FeatureImputationMetric]]:
        """Impute missing values and generate detailed before/after feature tracking metrics."""
        if not self.is_fitted:
            raise ImputationError("BaselineImputer must be fitted before calling transform.")

        imputed_df = df.copy(deep=True)
        metrics: list[FeatureImputationMetric] = []

        for col in self.target_features:
            if col not in imputed_df.columns:
                continue

            replacement_val = self.imputation_parameters.get(col)
            if replacement_val is None:
                continue

            missing_before = int(imputed_df[col].isna().sum())
            if missing_before > 0:
                imputed_df[col] = imputed_df[col].fillna(replacement_val)

            missing_after = int(imputed_df[col].isna().sum())
            imputed_count = missing_before - missing_after

            metrics.append(
                FeatureImputationMetric(
                    feature_name=col,
                    strategy_applied=self.feature_strategies[col],
                    statistic_value=replacement_val,
                    missing_before=missing_before,
                    imputed_count=imputed_count,
                    missing_after=missing_after,
                )
            )

        return imputed_df, metrics
