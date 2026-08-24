"""Downstream ML Model Impact and End-to-End Validation Engine.

Orchestrates leakage-safe downstream model evaluation, comparing complete-data baselines,
candidate imputation methods, and fairness mitigations across demographic groups and missingness rates.
"""

import time
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from missing_data_platform.bias.engine import BiasAnalysisEngine
from missing_data_platform.downstream.config import (
    DownstreamBenchmarkConfig,
    DownstreamConfig,
    DownstreamTaskType,
)
from missing_data_platform.downstream.metrics import (
    calculate_classification_metrics,
    calculate_group_disparity,
    calculate_group_downstream_metrics,
    calculate_imputation_downstream_correlation,
    calculate_performance_delta,
    calculate_performance_recovery,
    calculate_regression_metrics,
)
from missing_data_platform.downstream.models import DownstreamModelWrapper
from missing_data_platform.downstream.report import (
    DownstreamBenchmarkReport,
    DownstreamEvaluationResult,
    GroupDownstreamMetric,
    MissingnessCurveReport,
    RepeatedDownstreamReport,
)
from missing_data_platform.evaluation.metrics import (
    calculate_mae,
    calculate_rmse,
    validate_and_filter_predictions,
)
from missing_data_platform.exceptions import DataQualityError, EvaluationError
from missing_data_platform.imputation.config import BaselineStrategy
from missing_data_platform.imputation.engine import BaselineImputationEngine
from missing_data_platform.ingestion.contract import DataType, RawDataContract
from missing_data_platform.logging import get_logger
from missing_data_platform.masking.config import MaskingConfig
from missing_data_platform.masking.engine import MaskingEngine
from missing_data_platform.masking.ground_truth import GroundTruthStore
from missing_data_platform.mitigation.config import MitigationConfig
from missing_data_platform.mitigation.engine import FairnessMitigationEngine

logger = get_logger("downstream.engine")


class DownstreamEvaluationEngine:
    """Production engine for benchmarking downstream predictive ML impact across imputation strategies."""

    def __init__(
        self,
        contract: RawDataContract | None = None,
        config: DownstreamConfig | None = None,
    ) -> None:
        self.contract = contract or RawDataContract.default_consumer_contract()
        self.config = config or DownstreamConfig()

    def split_dataset(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Perform deterministic train/test partition while strictly preserving target isolation."""
        if df.empty:
            raise DataQualityError("Cannot split an empty DataFrame.")

        target_col = self.contract.target_column
        if target_col not in df.columns:
            raise DataQualityError(f"Target column '{target_col}' not found in DataFrame.")

        # Check for Stratification if classification
        stratify = None
        if self.config.task_type == DownstreamTaskType.CLASSIFICATION:
            target_series = df[target_col].dropna()
            if len(target_series) == len(df):
                val_counts = target_series.value_counts()
                if (val_counts >= 2).all() and len(val_counts) > 1:
                    stratify = df[target_col]

        train_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
            stratify=stratify,
        )

        return train_df.copy(deep=True), test_df.copy(deep=True)

    def evaluate_complete_baseline(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        experiment_id: str = "complete_baseline_exp",
    ) -> DownstreamEvaluationResult:
        """Evaluate downstream model on complete / reference dataset (idealized baseline)."""
        t0 = time.perf_counter()
        target_col = self.contract.target_column
        group_col = self.config.group_column

        # Separate target from features
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        model = DownstreamModelWrapper(config=self.config, contract=self.contract)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # Primary metrics
        if self.config.task_type == DownstreamTaskType.CLASSIFICATION:
            metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
        else:
            metrics = calculate_regression_metrics(y_test, y_pred)

        # Group metrics
        bias_engine = BiasAnalysisEngine(contract=self.contract)
        group_series_test = (
            bias_engine.extract_group_series(test_df)
            if group_col in test_df.columns
            else pd.Series("Unknown", index=test_df.index)
        )

        raw_group_metrics = calculate_group_downstream_metrics(
            y_true=y_test,
            y_pred=y_pred,
            group_series=group_series_test,
            y_prob=y_prob,
            task_type=self.config.task_type,
            minimum_group_size=self.config.minimum_group_size,
        )

        group_metrics_objs = [
            GroupDownstreamMetric(
                group_value=g_val,
                sample_count=g_data["sample_count"],
                is_small_group=g_data["is_small_group"],
                metrics=g_data["metrics"],
            )
            for g_val, g_data in raw_group_metrics.items()
        ]

        group_disp = calculate_group_disparity(
            raw_group_metrics, metric_name=self.config.primary_metric
        )
        runtime = round(time.perf_counter() - t0, 4)

        return DownstreamEvaluationResult(
            experiment_id=experiment_id,
            dataset_version=self.config.dataset_version,
            missingness_rate=0.0,
            mask_seed=self.config.random_seed,
            imputation_method="complete_reference",
            mitigation_enabled=False,
            downstream_model=self.config.model_type.value,
            primary_metric=self.config.primary_metric,
            metrics=metrics,
            group_metrics=group_metrics_objs,
            group_disparities=group_disp,
            performance_delta=dict.fromkeys(metrics, 0.0),
            recovery=100.0,
            imputation_mae=0.0,
            imputation_rmse=0.0,
            runtime_seconds=runtime,
            warnings=[],
            reproducibility_metadata={
                "model_type": self.config.model_type.value,
                "task_type": self.config.task_type.value,
                "random_seed": self.config.random_seed,
                "test_size": self.config.test_size,
            },
        )

    def _calculate_imputation_error(
        self,
        imputed_test_df: pd.DataFrame,
        ground_truth_store: GroundTruthStore | None,
    ) -> tuple[float | None, float | None]:
        """Compute imputation MAE and RMSE on test masked cells for numerical features."""
        if ground_truth_store is None or ground_truth_store.total_masked_cells == 0:
            return None, None

        maes: list[float] = []
        rmses: list[float] = []

        for col in ground_truth_store.mask_matrix.columns:
            if col not in imputed_test_df.columns:
                continue

            gt_series = ground_truth_store.get_ground_truth(col)
            # Intersect with test set indices
            common_idx = gt_series.index.intersection(imputed_test_df.index)
            if len(common_idx) == 0:
                continue

            test_gt = gt_series.loc[common_idx]
            test_pred = imputed_test_df.loc[common_idx, col]

            col_defn = self.contract.get_column(col)
            is_num = pd.api.types.is_numeric_dtype(test_gt) or (
                col_defn is not None and col_defn.data_type in (DataType.FLOAT, DataType.INTEGER)
            )

            if is_num:
                clean_t, clean_p, _, _ = validate_and_filter_predictions(test_gt, test_pred)
                if clean_t.size > 0:
                    maes.append(calculate_mae(clean_t, clean_p))
                    rmses.append(calculate_rmse(clean_t, clean_p))

        mean_mae = round(float(np.mean(maes)), 4) if maes else None
        mean_rmse = round(float(np.mean(rmses)), 4) if rmses else None
        return mean_mae, mean_rmse

    def evaluate_imputed_pipeline(
        self,
        masked_train_df: pd.DataFrame,
        masked_test_df: pd.DataFrame,
        clean_train_df: pd.DataFrame,
        clean_test_df: pd.DataFrame,
        method: str,
        ground_truth_store: GroundTruthStore | None = None,
        complete_baseline: DownstreamEvaluationResult | None = None,
        baseline_missing_result: DownstreamEvaluationResult | None = None,
        experiment_id: str = "imputed_downstream_exp",
        mask_rate: float = 0.20,
        mask_seed: int = 42,
    ) -> DownstreamEvaluationResult:
        """Perform end-to-end imputation fitting on train, transformation, downstream ML fitting, and evaluation."""
        t0 = time.perf_counter()
        target_col = self.contract.target_column
        group_col = self.config.group_column

        # 1. Ensure target and ID are preserved untouched
        y_train = clean_train_df[target_col]
        y_test = clean_test_df[target_col]

        imputation_engine = BaselineImputationEngine(contract=self.contract)
        m_lower = method.lower()

        # 2. Impute training set and test set using models fitted STRICTLY on masked_train_df
        if m_lower in ("baseline_median", "median"):
            imp_train = imputation_engine.impute_dataset(
                masked_train_df,
                experiment_id=f"{experiment_id}_train",
                numeric_strategy=BaselineStrategy.MEDIAN,
                train_df=masked_train_df,
            ).imputed_dataset
            imp_test = imputation_engine.impute_dataset(
                masked_test_df,
                experiment_id=f"{experiment_id}_test",
                numeric_strategy=BaselineStrategy.MEDIAN,
                train_df=masked_train_df,
            ).imputed_dataset
        elif m_lower in ("baseline_mean", "mean"):
            imp_train = imputation_engine.impute_dataset(
                masked_train_df,
                experiment_id=f"{experiment_id}_train",
                numeric_strategy=BaselineStrategy.MEAN,
                train_df=masked_train_df,
            ).imputed_dataset
            imp_test = imputation_engine.impute_dataset(
                masked_test_df,
                experiment_id=f"{experiment_id}_test",
                numeric_strategy=BaselineStrategy.MEAN,
                train_df=masked_train_df,
            ).imputed_dataset
        elif m_lower == "knn":
            imp_train = imputation_engine.impute_knn_dataset(
                masked_train_df,
                experiment_id=f"{experiment_id}_train",
                train_df=masked_train_df,
            ).imputed_dataset
            imp_test = imputation_engine.impute_knn_dataset(
                masked_test_df,
                experiment_id=f"{experiment_id}_test",
                train_df=masked_train_df,
            ).imputed_dataset
        elif m_lower in ("iterative", "mice"):
            imp_train = imputation_engine.impute_iterative_dataset(
                masked_train_df,
                experiment_id=f"{experiment_id}_train",
                train_df=masked_train_df,
            ).imputed_dataset
            imp_test = imputation_engine.impute_iterative_dataset(
                masked_test_df,
                experiment_id=f"{experiment_id}_test",
                train_df=masked_train_df,
            ).imputed_dataset
        elif m_lower in ("random_forest", "rf"):
            imp_train = imputation_engine.impute_rf_dataset(
                masked_train_df,
                experiment_id=f"{experiment_id}_train",
                train_df=masked_train_df,
            ).imputed_dataset
            imp_test = imputation_engine.impute_rf_dataset(
                masked_test_df,
                experiment_id=f"{experiment_id}_test",
                train_df=masked_train_df,
            ).imputed_dataset
        else:
            raise EvaluationError(
                f"Unsupported imputation method for downstream evaluation: {method}"
            )

        # 3. Train downstream ML model on imputed_train + true y_train
        X_train_imp = (
            imp_train.drop(columns=[target_col]) if target_col in imp_train.columns else imp_train
        )
        X_test_imp = (
            imp_test.drop(columns=[target_col]) if target_col in imp_test.columns else imp_test
        )

        model = DownstreamModelWrapper(config=self.config, contract=self.contract)
        model.fit(X_train_imp, y_train)

        # 4. Predict on imputed_test
        y_pred = model.predict(X_test_imp)
        y_prob = model.predict_proba(X_test_imp)

        # 5. Calculate Metrics
        if self.config.task_type == DownstreamTaskType.CLASSIFICATION:
            metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
        else:
            metrics = calculate_regression_metrics(y_test, y_pred)

        # 6. Group Metrics & Disparity
        bias_engine = BiasAnalysisEngine(contract=self.contract)
        group_series_test = (
            bias_engine.extract_group_series(clean_test_df)
            if group_col in clean_test_df.columns
            else pd.Series("Unknown", index=clean_test_df.index)
        )

        raw_group_metrics = calculate_group_downstream_metrics(
            y_true=y_test,
            y_pred=y_pred,
            group_series=group_series_test,
            y_prob=y_prob,
            task_type=self.config.task_type,
            minimum_group_size=self.config.minimum_group_size,
        )

        group_metrics_objs = [
            GroupDownstreamMetric(
                group_value=g_val,
                sample_count=g_data["sample_count"],
                is_small_group=g_data["is_small_group"],
                metrics=g_data["metrics"],
            )
            for g_val, g_data in raw_group_metrics.items()
        ]

        group_disp = calculate_group_disparity(
            raw_group_metrics, metric_name=self.config.primary_metric
        )

        # 7. Performance Delta & Recovery
        deltas = (
            calculate_performance_delta(complete_baseline.metrics, metrics)
            if complete_baseline is not None
            else dict.fromkeys(metrics)
        )

        complete_primary_val = (
            complete_baseline.metrics.get(self.config.primary_metric)
            if complete_baseline is not None
            else None
        )
        base_missing_primary_val = (
            baseline_missing_result.metrics.get(self.config.primary_metric)
            if baseline_missing_result is not None
            else None
        )
        imputed_primary_val = metrics.get(self.config.primary_metric)

        recovery = calculate_performance_recovery(
            complete_val=complete_primary_val,
            baseline_missing_val=base_missing_primary_val,
            imputed_val=imputed_primary_val,
            metric_name=self.config.primary_metric,
        )

        # 8. Imputation error
        imp_mae, imp_rmse = self._calculate_imputation_error(imp_test, ground_truth_store)
        runtime = round(time.perf_counter() - t0, 4)

        return DownstreamEvaluationResult(
            experiment_id=experiment_id,
            dataset_version=self.config.dataset_version,
            missingness_rate=mask_rate,
            mask_seed=mask_seed,
            imputation_method=method,
            mitigation_enabled=False,
            downstream_model=self.config.model_type.value,
            primary_metric=self.config.primary_metric,
            metrics=metrics,
            group_metrics=group_metrics_objs,
            group_disparities=group_disp,
            performance_delta=deltas,
            recovery=recovery,
            imputation_mae=imp_mae,
            imputation_rmse=imp_rmse,
            runtime_seconds=runtime,
            warnings=[],
            reproducibility_metadata={
                "imputation_method": method,
                "mask_rate": mask_rate,
                "mask_seed": mask_seed,
                "random_seed": self.config.random_seed,
            },
        )

    def evaluate_mitigated_pipeline(
        self,
        masked_train_df: pd.DataFrame,
        masked_test_df: pd.DataFrame,
        clean_train_df: pd.DataFrame,
        clean_test_df: pd.DataFrame,
        mitigation_config: MitigationConfig | None = None,
        ground_truth_store: GroundTruthStore | None = None,
        complete_baseline: DownstreamEvaluationResult | None = None,
        baseline_missing_result: DownstreamEvaluationResult | None = None,
        experiment_id: str = "mitigated_downstream_exp",
        mask_rate: float = 0.20,
        mask_seed: int = 42,
    ) -> DownstreamEvaluationResult:
        """Evaluate fairness-mitigated imputation pipeline against downstream ML tasks."""
        t0 = time.perf_counter()
        target_col = self.contract.target_column
        group_col = self.config.group_column
        mit_cfg = mitigation_config or MitigationConfig(enabled=True, group_column=group_col)
        mit_cfg.enabled = True

        y_train = clean_train_df[target_col]
        y_test = clean_test_df[target_col]

        mit_engine = FairnessMitigationEngine(contract=self.contract, config=mit_cfg)

        imp_train = mit_engine.impute_with_mitigation(
            masked_train_df, method="random_forest", train_df=masked_train_df
        )
        imp_test = mit_engine.impute_with_mitigation(
            masked_test_df, method="random_forest", train_df=masked_train_df
        )

        X_train_imp = (
            imp_train.drop(columns=[target_col]) if target_col in imp_train.columns else imp_train
        )
        X_test_imp = (
            imp_test.drop(columns=[target_col]) if target_col in imp_test.columns else imp_test
        )

        model = DownstreamModelWrapper(config=self.config, contract=self.contract)
        model.fit(X_train_imp, y_train)

        y_pred = model.predict(X_test_imp)
        y_prob = model.predict_proba(X_test_imp)

        if self.config.task_type == DownstreamTaskType.CLASSIFICATION:
            metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
        else:
            metrics = calculate_regression_metrics(y_test, y_pred)

        bias_engine = BiasAnalysisEngine(contract=self.contract)
        group_series_test = (
            bias_engine.extract_group_series(clean_test_df)
            if group_col in clean_test_df.columns
            else pd.Series("Unknown", index=clean_test_df.index)
        )

        raw_group_metrics = calculate_group_downstream_metrics(
            y_true=y_test,
            y_pred=y_pred,
            group_series=group_series_test,
            y_prob=y_prob,
            task_type=self.config.task_type,
            minimum_group_size=self.config.minimum_group_size,
        )

        group_metrics_objs = [
            GroupDownstreamMetric(
                group_value=g_val,
                sample_count=g_data["sample_count"],
                is_small_group=g_data["is_small_group"],
                metrics=g_data["metrics"],
            )
            for g_val, g_data in raw_group_metrics.items()
        ]

        group_disp = calculate_group_disparity(
            raw_group_metrics, metric_name=self.config.primary_metric
        )

        deltas = (
            calculate_performance_delta(complete_baseline.metrics, metrics)
            if complete_baseline is not None
            else dict.fromkeys(metrics)
        )

        complete_primary_val = (
            complete_baseline.metrics.get(self.config.primary_metric)
            if complete_baseline is not None
            else None
        )
        base_missing_primary_val = (
            baseline_missing_result.metrics.get(self.config.primary_metric)
            if baseline_missing_result is not None
            else None
        )
        imputed_primary_val = metrics.get(self.config.primary_metric)

        recovery = calculate_performance_recovery(
            complete_val=complete_primary_val,
            baseline_missing_val=base_missing_primary_val,
            imputed_val=imputed_primary_val,
            metric_name=self.config.primary_metric,
        )

        imp_mae, imp_rmse = self._calculate_imputation_error(imp_test, ground_truth_store)
        runtime = round(time.perf_counter() - t0, 4)

        return DownstreamEvaluationResult(
            experiment_id=experiment_id,
            dataset_version=self.config.dataset_version,
            missingness_rate=mask_rate,
            mask_seed=mask_seed,
            imputation_method="fairness_weighted_rf",
            mitigation_enabled=True,
            downstream_model=self.config.model_type.value,
            primary_metric=self.config.primary_metric,
            metrics=metrics,
            group_metrics=group_metrics_objs,
            group_disparities=group_disp,
            performance_delta=deltas,
            recovery=recovery,
            imputation_mae=imp_mae,
            imputation_rmse=imp_rmse,
            runtime_seconds=runtime,
            warnings=[],
            reproducibility_metadata={
                "mitigation_strategy": mit_cfg.strategy.value,
                "mask_rate": mask_rate,
                "mask_seed": mask_seed,
                "random_seed": self.config.random_seed,
            },
        )

    def run_benchmark_suite(
        self,
        df: pd.DataFrame,
        mask_config: MaskingConfig | None = None,
        benchmark_config: DownstreamBenchmarkConfig | None = None,
    ) -> DownstreamBenchmarkReport:
        """Run comprehensive downstream benchmark across all candidate methods and mitigations."""
        bench_cfg = benchmark_config or DownstreamBenchmarkConfig(downstream_config=self.config)
        mask_cfg = mask_config or MaskingConfig(
            experiment_id=f"{bench_cfg.experiment_id}_mask",
            mask_rate=0.20,
            random_seed=self.config.random_seed,
        )

        logger.info(
            "Starting downstream benchmark suite",
            experiment_id=bench_cfg.experiment_id,
            methods=bench_cfg.methods,
            model_type=self.config.model_type.value,
            primary_metric=self.config.primary_metric,
        )

        # 1. Strict train/test partition on complete data
        train_df, test_df = self.split_dataset(df)

        # 2. Complete-Data Baseline (Reference Condition A)
        complete_baseline = self.evaluate_complete_baseline(
            train_df=train_df,
            test_df=test_df,
            experiment_id=f"{bench_cfg.experiment_id}_complete_baseline",
        )

        # 3. Mask train and test partitions
        masking_engine = MaskingEngine(contract=self.contract)
        mask_train_res = masking_engine.generate_benchmark_dataset(train_df, mask_cfg)

        mask_test_cfg = deepcopy(mask_cfg)
        mask_test_cfg.random_seed = mask_cfg.random_seed + 1000
        mask_test_res = masking_engine.generate_benchmark_dataset(test_df, mask_test_cfg)

        masked_train_df = mask_train_res.masked_dataset
        masked_test_df = mask_test_res.masked_dataset

        # 4. First run baseline missing condition (for recovery calculation)
        baseline_missing_res = self.evaluate_imputed_pipeline(
            masked_train_df=masked_train_df,
            masked_test_df=masked_test_df,
            clean_train_df=train_df,
            clean_test_df=test_df,
            method="baseline_median",
            ground_truth_store=mask_test_res.ground_truth_store,
            complete_baseline=complete_baseline,
            baseline_missing_result=None,
            experiment_id=f"{bench_cfg.experiment_id}_baseline_median",
            mask_rate=mask_cfg.mask_rate,
            mask_seed=mask_cfg.random_seed,
        )
        baseline_missing_res.recovery = 0.0

        method_results: dict[str, DownstreamEvaluationResult] = {}
        for method in bench_cfg.methods:
            if method.lower() in ("baseline_median", "median"):
                method_results[method] = baseline_missing_res
            else:
                m_res = self.evaluate_imputed_pipeline(
                    masked_train_df=masked_train_df,
                    masked_test_df=masked_test_df,
                    clean_train_df=train_df,
                    clean_test_df=test_df,
                    method=method,
                    ground_truth_store=mask_test_res.ground_truth_store,
                    complete_baseline=complete_baseline,
                    baseline_missing_result=baseline_missing_res,
                    experiment_id=f"{bench_cfg.experiment_id}_{method}",
                    mask_rate=mask_cfg.mask_rate,
                    mask_seed=mask_cfg.random_seed,
                )
                method_results[method] = m_res

        # 5. Evaluate mitigation if enabled
        mitigated_results: dict[str, DownstreamEvaluationResult] = {}
        if bench_cfg.include_mitigation:
            mit_res = self.evaluate_mitigated_pipeline(
                masked_train_df=masked_train_df,
                masked_test_df=masked_test_df,
                clean_train_df=train_df,
                clean_test_df=test_df,
                ground_truth_store=mask_test_res.ground_truth_store,
                complete_baseline=complete_baseline,
                baseline_missing_result=baseline_missing_res,
                experiment_id=f"{bench_cfg.experiment_id}_mitigated",
                mask_rate=mask_cfg.mask_rate,
                mask_seed=mask_cfg.random_seed,
            )
            mitigated_results["fairness_weighted_rf"] = mit_res

        # 6. Build structured comparison table
        comparison_table: list[dict[str, Any]] = []

        # Complete Baseline row
        comparison_table.append(
            {
                "method": "complete_reference",
                "mitigation": False,
                "missingness_rate": 0.0,
                "primary_metric": complete_baseline.metrics.get(self.config.primary_metric),
                "delta_from_complete": 0.0,
                "recovery": 100.0,
                "group_disparity": complete_baseline.group_disparities.get("max_disparity"),
                "imputation_mae": 0.0,
                "runtime_seconds": complete_baseline.runtime_seconds,
            }
        )

        for m_name, res in method_results.items():
            comparison_table.append(
                {
                    "method": m_name,
                    "mitigation": False,
                    "missingness_rate": res.missingness_rate,
                    "primary_metric": res.metrics.get(self.config.primary_metric),
                    "delta_from_complete": res.performance_delta.get(self.config.primary_metric),
                    "recovery": res.recovery,
                    "group_disparity": res.group_disparities.get("max_disparity"),
                    "imputation_mae": res.imputation_mae,
                    "runtime_seconds": res.runtime_seconds,
                }
            )

        for m_name, res in mitigated_results.items():
            comparison_table.append(
                {
                    "method": m_name,
                    "mitigation": True,
                    "missingness_rate": res.missingness_rate,
                    "primary_metric": res.metrics.get(self.config.primary_metric),
                    "delta_from_complete": res.performance_delta.get(self.config.primary_metric),
                    "recovery": res.recovery,
                    "group_disparity": res.group_disparities.get("max_disparity"),
                    "imputation_mae": res.imputation_mae,
                    "runtime_seconds": res.runtime_seconds,
                }
            )

        # 7. Imputation vs Downstream correlation summary
        maes: list[float] = []
        f1s: list[float] = []
        for res in method_results.values():
            if res.imputation_mae is not None:
                p_val = res.metrics.get(self.config.primary_metric)
                if p_val is not None:
                    maes.append(res.imputation_mae)
                    f1s.append(p_val)

        corr_summary = calculate_imputation_downstream_correlation(maes, f1s)
        corr_summary["sample_methods_count"] = len(maes)

        # 8. Group disparity summary
        group_disp_summary: dict[str, Any] = {
            "complete_disparity": complete_baseline.group_disparities.get("max_disparity"),
            "method_disparities": {
                m: res.group_disparities.get("max_disparity") for m, res in method_results.items()
            },
            "mitigated_disparities": {
                m: res.group_disparities.get("max_disparity")
                for m, res in mitigated_results.items()
            },
        }

        return DownstreamBenchmarkReport(
            experiment_id=bench_cfg.experiment_id,
            dataset_version=self.config.dataset_version,
            downstream_model=self.config.model_type.value,
            primary_metric=self.config.primary_metric,
            complete_baseline=complete_baseline,
            method_results=method_results,
            mitigated_results=mitigated_results,
            comparison_table=comparison_table,
            imputation_vs_downstream_summary=corr_summary,
            group_disparity_summary=group_disp_summary,
            warnings=[],
        )

    def run_missingness_rate_curve(
        self,
        df: pd.DataFrame,
        benchmark_config: DownstreamBenchmarkConfig | None = None,
    ) -> MissingnessCurveReport:
        """Sweep missingness rates (e.g. 10%, 20%, 30%, 40%, 50%) to track degradation curve."""
        bench_cfg = benchmark_config or DownstreamBenchmarkConfig(downstream_config=self.config)
        curve_points: list[dict[str, Any]] = []

        train_df, test_df = self.split_dataset(df)
        complete_baseline = self.evaluate_complete_baseline(
            train_df, test_df, experiment_id=f"{bench_cfg.experiment_id}_curve_comp"
        )
        comp_primary = complete_baseline.metrics.get(self.config.primary_metric)

        for rate in bench_cfg.missingness_rates:
            mask_cfg = MaskingConfig(
                experiment_id=f"{bench_cfg.experiment_id}_rate_{int(rate * 100)}",
                mask_rate=rate,
                random_seed=self.config.random_seed,
            )

            masking_engine = MaskingEngine(contract=self.contract)
            mask_train = masking_engine.generate_benchmark_dataset(train_df, mask_cfg)

            mask_test_cfg = deepcopy(mask_cfg)
            mask_test_cfg.random_seed = mask_cfg.random_seed + 1000
            mask_test = masking_engine.generate_benchmark_dataset(test_df, mask_test_cfg)

            # Baseline missing
            base_missing = self.evaluate_imputed_pipeline(
                masked_train_df=mask_train.masked_dataset,
                masked_test_df=mask_test.masked_dataset,
                clean_train_df=train_df,
                clean_test_df=test_df,
                method="baseline_median",
                ground_truth_store=mask_test.ground_truth_store,
                complete_baseline=complete_baseline,
                baseline_missing_result=None,
                experiment_id=f"{bench_cfg.experiment_id}_median_r{int(rate * 100)}",
                mask_rate=rate,
                mask_seed=mask_cfg.random_seed,
            )

            for method in bench_cfg.methods:
                m_res = self.evaluate_imputed_pipeline(
                    masked_train_df=mask_train.masked_dataset,
                    masked_test_df=mask_test.masked_dataset,
                    clean_train_df=train_df,
                    clean_test_df=test_df,
                    method=method,
                    ground_truth_store=mask_test.ground_truth_store,
                    complete_baseline=complete_baseline,
                    baseline_missing_result=base_missing,
                    experiment_id=f"{bench_cfg.experiment_id}_{method}_r{int(rate * 100)}",
                    mask_rate=rate,
                    mask_seed=mask_cfg.random_seed,
                )
                m_val = m_res.metrics.get(self.config.primary_metric)
                delta = (
                    round(float(m_val - comp_primary), 4)
                    if m_val is not None and comp_primary is not None
                    else None
                )

                curve_points.append(
                    {
                        "missingness_rate": rate,
                        "method": method,
                        "primary_metric_name": self.config.primary_metric,
                        "primary_metric_value": m_val,
                        "metric_delta": delta,
                        "recovery": m_res.recovery,
                        "imputation_mae": m_res.imputation_mae,
                    }
                )

        return MissingnessCurveReport(
            experiment_id=bench_cfg.experiment_id,
            primary_metric=self.config.primary_metric,
            missingness_rates=bench_cfg.missingness_rates,
            curve_points=curve_points,
        )

    def run_repeated_benchmark(
        self,
        df: pd.DataFrame,
        seeds: list[int] | None = None,
        benchmark_config: DownstreamBenchmarkConfig | None = None,
    ) -> RepeatedDownstreamReport:
        """Run repeated downstream evaluations across multiple random seeds for statistical stability."""
        bench_cfg = benchmark_config or DownstreamBenchmarkConfig(downstream_config=self.config)
        eval_seeds = seeds or bench_cfg.repeated_seeds

        method_scores: dict[str, list[float]] = {m: [] for m in bench_cfg.methods}

        for seed in eval_seeds:
            mask_cfg = MaskingConfig(
                experiment_id=f"{bench_cfg.experiment_id}_seed_{seed}",
                mask_rate=0.20,
                random_seed=seed,
            )
            report = self.run_benchmark_suite(
                df=df,
                mask_config=mask_cfg,
                benchmark_config=bench_cfg,
            )
            for m in bench_cfg.methods:
                res = report.method_results.get(m)
                if res is not None:
                    v = res.metrics.get(self.config.primary_metric)
                    if v is not None:
                        method_scores[m].append(float(v))

        stats: dict[str, dict[str, float]] = {}
        for m, scores in method_scores.items():
            if scores:
                mean_v = float(np.mean(scores))
                std_v = float(np.std(scores)) if len(scores) > 1 else 0.0
                # 95% confidence interval half-width: 1.96 * (std / sqrt(n))
                ci_margin = float(1.96 * (std_v / np.sqrt(len(scores)))) if len(scores) > 1 else 0.0
                stats[m] = {
                    "mean": round(mean_v, 4),
                    "std": round(std_v, 4),
                    "ci_95_low": round(mean_v - ci_margin, 4),
                    "ci_95_high": round(mean_v + ci_margin, 4),
                    "repetitions": len(scores),
                }
            else:
                stats[m] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "ci_95_low": 0.0,
                    "ci_95_high": 0.0,
                    "repetitions": 0,
                }

        return RepeatedDownstreamReport(
            experiment_id=bench_cfg.experiment_id,
            repeated_seeds=eval_seeds,
            total_repetitions=len(eval_seeds),
            primary_metric=self.config.primary_metric,
            method_stats=stats,
        )
