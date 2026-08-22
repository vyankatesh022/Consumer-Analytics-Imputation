"""Baseline Imputation Engine Orchestrator.

Orchestrates baseline statistical imputation runs, prevents data leakage from evaluation
ground-truth sets, and generates comprehensive ImputationResult audit reports.
"""

import pandas as pd

from missing_data_platform.exceptions import DataQualityError
from missing_data_platform.imputation.baseline import BaselineImputer
from missing_data_platform.imputation.config import (
    BaselineImputationConfig,
    BaselineStrategy,
)
from missing_data_platform.imputation.report import ImputationResult
from missing_data_platform.ingestion.contract import RawDataContract
from missing_data_platform.logging import get_logger

logger = get_logger("imputation.engine")


class BaselineImputationEngine:
    """Orchestrator for baseline univariate imputation executions."""

    def __init__(
        self,
        contract: RawDataContract | None = None,
    ) -> None:
        self.contract = contract or RawDataContract.default_consumer_contract()

    def impute_dataset(
        self,
        df: pd.DataFrame,
        experiment_id: str = "baseline_impute_exp",
        numeric_strategy: BaselineStrategy = BaselineStrategy.MEDIAN,
        categorical_strategy: BaselineStrategy = BaselineStrategy.MODE,
        target_features: list[str] | None = None,
        train_df: pd.DataFrame | None = None,
    ) -> ImputationResult:
        """Fit baseline statistics and transform the dataset.

        Args:
            df: Target dataset containing missing values to impute.
            experiment_id: Unique experiment tracking identifier.
            numeric_strategy: Statistical strategy for numeric columns (mean, median, constant).
            categorical_strategy: Statistical strategy for categorical columns (mode, constant).
            target_features: Optional explicit list of features to impute (protected columns excluded).
            train_df: Optional reference/training partition from which to compute statistics.
                     If None, statistics are computed directly from observed values in df.

        Raises:
            DataQualityError: If target DataFrame is empty.
        """
        if df.empty:
            raise DataQualityError(
                "Cannot perform baseline imputation on an empty DataFrame.",
                context={"experiment_id": experiment_id},
            )

        logger.info(
            "Starting baseline imputation",
            experiment_id=experiment_id,
            numeric_strategy=numeric_strategy.value,
            categorical_strategy=categorical_strategy.value,
            total_records=len(df),
        )

        config = BaselineImputationConfig(
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
            target_features=target_features,
            protected_features=[self.contract.id_column, self.contract.target_column],
        )

        imputer = BaselineImputer(config=config, contract=self.contract)

        # Fit strictly on train_df or df observed values (never held-out ground truth)
        fit_source = train_df if train_df is not None else df
        imputer.fit(fit_source)

        imputed_df, metrics = imputer.transform_with_metrics(df)
        total_cells_imputed = sum(m.imputed_count for m in metrics)

        result = ImputationResult(
            imputed_dataset=imputed_df,
            experiment_id=experiment_id,
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
            total_records=len(df),
            total_cells_imputed=total_cells_imputed,
            feature_metrics=metrics,
            imputation_parameters=imputer.imputation_parameters,
        )

        logger.info(
            "Baseline imputation completed",
            experiment_id=experiment_id,
            total_cells_imputed=total_cells_imputed,
            features_imputed_count=len(metrics),
        )

        return result

    def impute_knn_dataset(
        self,
        df: pd.DataFrame,
        experiment_id: str = "knn_impute_exp",
        n_neighbors: int = 5,
        scaling_strategy: str = "standard",
        target_features: list[str] | None = None,
        train_df: pd.DataFrame | None = None,
    ) -> ImputationResult:
        """Fit KNN neighbor representation and impute numeric features.

        Args:
            df: Target dataset containing missing values to impute.
            experiment_id: Unique experiment tracking identifier.
            n_neighbors: Number of nearest neighbors (k >= 1).
            scaling_strategy: 'standard', 'minmax', or 'none'.
            target_features: Optional explicit list of features to impute.
            train_df: Optional reference/training partition from which to compute neighbor space.

        Raises:
            DataQualityError: If target DataFrame is empty.
        """
        if df.empty:
            raise DataQualityError(
                "Cannot perform KNN imputation on an empty DataFrame.",
                context={"experiment_id": experiment_id},
            )

        from missing_data_platform.imputation.knn import (
            KNNImputationConfig,
            KNNImputerModel,
            KNNWeighting,
            ScalingStrategy,
        )

        logger.info(
            "Starting KNN imputation",
            experiment_id=experiment_id,
            n_neighbors=n_neighbors,
            scaling_strategy=scaling_strategy,
            total_records=len(df),
        )

        knn_config = KNNImputationConfig(
            n_neighbors=n_neighbors,
            weights=KNNWeighting.UNIFORM,
            scaling_strategy=ScalingStrategy(scaling_strategy),
            target_features=target_features,
            protected_features=[self.contract.id_column, self.contract.target_column],
        )

        imputer = KNNImputerModel(config=knn_config, contract=self.contract)
        fit_source = train_df if train_df is not None else df
        imputer.fit(fit_source)

        imputed_df, metrics = imputer.transform_with_metrics(df)
        total_cells_imputed = sum(m.imputed_count for m in metrics)

        result = ImputationResult(
            imputed_dataset=imputed_df,
            experiment_id=experiment_id,
            numeric_strategy=BaselineStrategy.CONSTANT,
            categorical_strategy=BaselineStrategy.MODE,
            total_records=len(df),
            total_cells_imputed=total_cells_imputed,
            feature_metrics=metrics,
            imputation_parameters=imputer.imputation_parameters,
        )

        logger.info(
            "KNN imputation completed",
            experiment_id=experiment_id,
            total_cells_imputed=total_cells_imputed,
            features_imputed_count=len(metrics),
        )

        return result
