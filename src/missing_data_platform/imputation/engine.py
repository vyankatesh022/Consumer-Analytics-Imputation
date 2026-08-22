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

    def impute_iterative_dataset(
        self,
        df: pd.DataFrame,
        experiment_id: str = "iterative_impute_exp",
        max_iter: int = 10,
        tol: float = 1e-3,
        random_seed: int = 42,
        target_features: list[str] | None = None,
        train_df: pd.DataFrame | None = None,
    ) -> ImputationResult:
        """Fit iterative multivariate chained models and impute missing numerical features.

        Args:
            df: Target dataset containing missing values to impute.
            experiment_id: Unique experiment tracking identifier.
            max_iter: Maximum number of round-robin imputation cycles (>= 1).
            tol: Tolerance for stopping criterion.
            random_seed: Deterministic random state for Bayesian Ridge regressions.
            target_features: Optional explicit list of features to impute.
            train_df: Optional reference/training partition from which to train chained models.

        Raises:
            DataQualityError: If target DataFrame is empty.
        """
        if df.empty:
            raise DataQualityError(
                "Cannot perform iterative imputation on an empty DataFrame.",
                context={"experiment_id": experiment_id},
            )

        from missing_data_platform.imputation.iterative import (
            ImputationOrder,
            InitialStrategy,
            IterativeImputationConfig,
            IterativeImputerModel,
        )

        logger.info(
            "Starting iterative multivariate imputation",
            experiment_id=experiment_id,
            max_iter=max_iter,
            tol=tol,
            random_seed=random_seed,
            total_records=len(df),
        )

        iter_config = IterativeImputationConfig(
            max_iter=max_iter,
            tol=tol,
            initial_strategy=InitialStrategy.MEAN,
            imputation_order=ImputationOrder.ASCENDING,
            random_seed=random_seed,
            target_features=target_features,
            protected_features=[self.contract.id_column, self.contract.target_column],
        )

        imputer = IterativeImputerModel(config=iter_config, contract=self.contract)
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
            "Iterative imputation completed",
            experiment_id=experiment_id,
            total_cells_imputed=total_cells_imputed,
            features_imputed_count=len(metrics),
            converged=imputer.imputation_parameters.get("converged"),
        )

        return result

    def impute_rf_dataset(
        self,
        df: pd.DataFrame,
        experiment_id: str = "rf_impute_exp",
        n_estimators: int = 100,
        max_depth: int | None = 15,
        min_samples_leaf: int = 1,
        max_features: float | int | str | None = "sqrt",
        random_seed: int = 42,
        n_jobs: int = 1,
        target_features: list[str] | None = None,
        train_df: pd.DataFrame | None = None,
    ) -> ImputationResult:
        """Fit Random Forest regression models and impute missing numerical features.

        Args:
            df: Target dataset containing missing values to impute.
            experiment_id: Unique experiment tracking identifier.
            n_estimators: Number of trees in the forest (1 <= n <= 500).
            max_depth: Maximum depth of the trees (1 <= d <= 50 or None).
            min_samples_leaf: Minimum samples per leaf (>= 1).
            max_features: Number of features to consider when looking for best split.
            random_seed: Deterministic random state for Random Forest models.
            n_jobs: Number of parallel jobs for tree fitting (>= 1 or -1).
            target_features: Optional explicit list of features to impute.
            train_df: Optional reference/training partition from which to train models.

        Raises:
            DataQualityError: If target DataFrame is empty.
        """
        if df.empty:
            raise DataQualityError(
                "Cannot perform Random Forest imputation on an empty DataFrame.",
                context={"experiment_id": experiment_id},
            )

        from missing_data_platform.imputation.rf import (
            RandomForestImputationConfig,
            RandomForestImputerModel,
        )

        logger.info(
            "Starting Random Forest imputation",
            experiment_id=experiment_id,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_seed=random_seed,
            n_jobs=n_jobs,
            total_records=len(df),
        )

        rf_config = RandomForestImputationConfig(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_seed=random_seed,
            n_jobs=n_jobs,
            target_features=target_features,
            protected_features=[self.contract.id_column, self.contract.target_column],
        )

        imputer = RandomForestImputerModel(config=rf_config, contract=self.contract)
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
            "Random Forest imputation completed",
            experiment_id=experiment_id,
            total_cells_imputed=total_cells_imputed,
            features_imputed_count=len(metrics),
            trained_models_count=len(imputer.imputation_parameters.get("trained_models", [])),
        )

        return result
