"""Imputation Evaluation and Benchmark Engine.

Orchestrates leakage-safe model evaluation against ground truth masks, computes per-feature
and aggregate metrics, ranks imputation algorithms, and performs multi-seed stability benchmarks.
"""

from copy import deepcopy

import numpy as np
import pandas as pd

from missing_data_platform.evaluation.config import EvaluationConfig
from missing_data_platform.evaluation.metrics import (
    calculate_accuracy,
    calculate_mae,
    calculate_nrmse,
    calculate_rmse,
    validate_and_filter_predictions,
)
from missing_data_platform.evaluation.report import (
    BenchmarkComparisonReport,
    FeatureEvaluationResult,
    MethodEvaluationResult,
    RepeatedExperimentReport,
)
from missing_data_platform.exceptions import DataQualityError, EvaluationError
from missing_data_platform.imputation.config import BaselineStrategy
from missing_data_platform.imputation.engine import BaselineImputationEngine
from missing_data_platform.imputation.report import ImputationResult
from missing_data_platform.ingestion.contract import DataType, RawDataContract
from missing_data_platform.logging import get_logger
from missing_data_platform.masking.config import MaskingConfig
from missing_data_platform.masking.engine import MaskingEngine
from missing_data_platform.masking.ground_truth import GroundTruthStore

logger = get_logger("evaluation.engine")


class ImputationEvaluator:
    """Production evaluation engine for benchmarking and comparing imputation strategies."""

    def __init__(
        self,
        contract: RawDataContract | None = None,
        config: EvaluationConfig | None = None,
    ) -> None:
        self.contract = contract or RawDataContract.default_consumer_contract()
        self.config = config or EvaluationConfig()

    def evaluate_method(
        self,
        imputed_df: pd.DataFrame,
        ground_truth_store: GroundTruthStore,
        method_name: str,
        experiment_id: str = "eval_exp",
    ) -> MethodEvaluationResult:
        """Evaluate an imputation output strictly against hidden ground-truth cells.

        Args:
            imputed_df: DataFrame output produced by the imputation algorithm.
            ground_truth_store: Encapsulated ground truth and boolean mask matrix.
            method_name: Identifier of the imputation method being evaluated.
            experiment_id: Experiment tracking ID.

        Raises:
            DataQualityError: If imputed DataFrame is empty.
            EvaluationError: If ground truth store is invalid or column alignment fails.
        """
        if imputed_df.empty:
            raise DataQualityError("Cannot evaluate an empty imputed DataFrame.")

        if ground_truth_store.total_masked_cells == 0:
            raise EvaluationError("GroundTruthStore contains 0 masked cells to evaluate.")

        feature_results: list[FeatureEvaluationResult] = []
        total_cells_all = 0
        missing_preds_all = 0

        # Evaluate only features that were artificially masked
        for col in ground_truth_store.mask_matrix.columns:
            gt_series = ground_truth_store.get_ground_truth(col)
            if gt_series.empty:
                continue

            if col not in imputed_df.columns:
                raise EvaluationError(
                    f"Evaluated feature '{col}' not found in imputed DataFrame.",
                    context={"feature": col},
                )

            # Extract imputed values at the exact masked indices
            pred_series = imputed_df.loc[gt_series.index, col]
            evaluated_count = len(gt_series)
            total_cells_all += evaluated_count

            col_defn = self.contract.get_column(col)
            is_numeric = pd.api.types.is_numeric_dtype(gt_series) or (
                col_defn is not None and col_defn.data_type in (DataType.FLOAT, DataType.INTEGER)
            )

            if is_numeric:
                clean_true, clean_pred, missing_count, invalid_count = (
                    validate_and_filter_predictions(gt_series, pred_series)
                )
                missing_preds_all += missing_count + invalid_count

                if clean_true.size > 0:
                    mae = calculate_mae(clean_true, clean_pred)
                    rmse = calculate_rmse(clean_true, clean_pred)
                    nrmse = calculate_nrmse(clean_true, clean_pred)
                else:
                    mae = None
                    rmse = None
                    nrmse = None

                feature_results.append(
                    FeatureEvaluationResult(
                        feature_name=col,
                        feature_type="numeric",
                        method=method_name,
                        evaluated_count=evaluated_count,
                        missing_prediction_count=missing_count + invalid_count,
                        mae=round(mae, 4) if mae is not None else None,
                        rmse=round(rmse, 4) if rmse is not None else None,
                        nrmse=round(nrmse, 4) if nrmse is not None else None,
                    )
                )
            else:
                clean_true, clean_pred, missing_count, invalid_count = (
                    validate_and_filter_predictions(gt_series, pred_series)
                )
                missing_preds_all += missing_count + invalid_count
                acc = calculate_accuracy(clean_true, clean_pred) if clean_true.size > 0 else None

                feature_results.append(
                    FeatureEvaluationResult(
                        feature_name=col,
                        feature_type="categorical",
                        method=method_name,
                        evaluated_count=evaluated_count,
                        missing_prediction_count=missing_count + invalid_count,
                        mae=None,
                        rmse=None,
                        accuracy=round(acc, 4) if acc is not None else None,
                    )
                )

        # Aggregate metrics across numeric features
        num_features = [
            f for f in feature_results if f.feature_type == "numeric" and f.mae is not None
        ]
        if num_features:
            macro_mae = float(np.mean([f.mae for f in num_features if f.mae is not None]))
            macro_rmse = float(np.mean([f.rmse for f in num_features if f.rmse is not None]))

            # Weighted by evaluated cell count per feature
            tot_num_cells = sum(f.evaluated_count for f in num_features)
            if tot_num_cells > 0:
                weighted_mae = float(
                    sum(f.mae * f.evaluated_count for f in num_features if f.mae is not None)
                    / tot_num_cells
                )
                weighted_rmse = float(
                    sum(f.rmse * f.evaluated_count for f in num_features if f.rmse is not None)
                    / tot_num_cells
                )
            else:
                weighted_mae = macro_mae
                weighted_rmse = macro_rmse
        else:
            macro_mae = None
            macro_rmse = None
            weighted_mae = None
            weighted_rmse = None

        return MethodEvaluationResult(
            method=method_name,
            experiment_id=experiment_id,
            total_evaluated_cells=total_cells_all,
            missing_prediction_count=missing_preds_all,
            macro_mae=round(macro_mae, 4) if macro_mae is not None else None,
            macro_rmse=round(macro_rmse, 4) if macro_rmse is not None else None,
            weighted_mae=round(weighted_mae, 4) if weighted_mae is not None else None,
            weighted_rmse=round(weighted_rmse, 4) if weighted_rmse is not None else None,
            feature_results=feature_results,
        )

    def compare_methods(
        self,
        imputed_results: dict[str, pd.DataFrame | ImputationResult],
        ground_truth_store: GroundTruthStore,
        experiment_id: str = "comp_exp",
        dataset_version: str = "v1.0",
        mask_strategy: str = "uniform_random",
        mask_rate: float = 0.2,
        mask_seed: int = 42,
    ) -> BenchmarkComparisonReport:
        """Compare multiple imputation methods against identical ground-truth masks and calculate ranks."""
        if not imputed_results:
            raise EvaluationError("Cannot compare methods: imputed_results dictionary is empty.")

        method_evals: dict[str, MethodEvaluationResult] = {}
        for method_name, imp_data in imputed_results.items():
            df = imp_data.imputed_dataset if isinstance(imp_data, ImputationResult) else imp_data
            method_evals[method_name] = self.evaluate_method(
                imputed_df=df,
                ground_truth_store=ground_truth_store,
                method_name=method_name,
                experiment_id=experiment_id,
            )

        # Calculate rankings based on Weighted MAE (and Weighted RMSE as secondary)
        # Handle None values by placing them last
        def mae_sort_key(res: MethodEvaluationResult) -> float:
            return res.weighted_mae if res.weighted_mae is not None else float("inf")

        def rmse_sort_key(res: MethodEvaluationResult) -> float:
            return res.weighted_rmse if res.weighted_rmse is not None else float("inf")

        # Rank by MAE
        sorted_by_mae = sorted(method_evals.values(), key=mae_sort_key)
        for rank_idx, m_res in enumerate(sorted_by_mae, 1):
            m_res.rank_mae = rank_idx

        # Rank by RMSE
        sorted_by_rmse = sorted(method_evals.values(), key=rmse_sort_key)
        for rank_idx, m_res in enumerate(sorted_by_rmse, 1):
            m_res.rank_rmse = rank_idx

        rankings_summary = []
        for m_res in sorted_by_mae:
            rankings_summary.append(
                {
                    "rank_mae": m_res.rank_mae,
                    "rank_rmse": m_res.rank_rmse,
                    "method": m_res.method,
                    "total_evaluated_cells": m_res.total_evaluated_cells,
                    "missing_prediction_count": m_res.missing_prediction_count,
                    "weighted_mae": m_res.weighted_mae,
                    "weighted_rmse": m_res.weighted_rmse,
                    "macro_mae": m_res.macro_mae,
                    "macro_rmse": m_res.macro_rmse,
                }
            )

        return BenchmarkComparisonReport(
            experiment_id=experiment_id,
            dataset_version=dataset_version,
            mask_strategy=mask_strategy,
            mask_rate=mask_rate,
            mask_seed=mask_seed,
            method_results=method_evals,
            method_rankings=rankings_summary,
        )

    def run_benchmark_suite(
        self,
        df: pd.DataFrame,
        mask_config: MaskingConfig,
        methods: list[str] | None = None,
        train_df: pd.DataFrame | None = None,
    ) -> BenchmarkComparisonReport:
        """Execute end-to-end benchmark masking, multi-algorithm imputation, and comparative evaluation."""
        methods_to_run = methods or self.config.supported_methods
        logger.info(
            "Starting automated imputation benchmark suite",
            experiment_id=mask_config.experiment_id,
            methods=methods_to_run,
            mask_rate=mask_config.mask_rate,
            random_seed=mask_config.random_seed,
        )

        # 1. Mask reference dataset
        masking_engine = MaskingEngine(contract=self.contract)
        bench_mask_result = masking_engine.generate_benchmark_dataset(df, mask_config)

        # 2. Impute with each requested method
        imputation_engine = BaselineImputationEngine(contract=self.contract)
        imputed_dict: dict[str, pd.DataFrame] = {}

        for method in methods_to_run:
            m_lower = method.lower()
            if m_lower in ("baseline_median", "median"):
                res = imputation_engine.impute_dataset(
                    bench_mask_result.masked_dataset,
                    experiment_id=f"{mask_config.experiment_id}_median",
                    numeric_strategy=BaselineStrategy.MEDIAN,
                    train_df=train_df,
                )
                imputed_dict[method] = res.imputed_dataset
            elif m_lower in ("baseline_mean", "mean"):
                res = imputation_engine.impute_dataset(
                    bench_mask_result.masked_dataset,
                    experiment_id=f"{mask_config.experiment_id}_mean",
                    numeric_strategy=BaselineStrategy.MEAN,
                    train_df=train_df,
                )
                imputed_dict[method] = res.imputed_dataset
            elif m_lower == "knn":
                res = imputation_engine.impute_knn_dataset(
                    bench_mask_result.masked_dataset,
                    experiment_id=f"{mask_config.experiment_id}_knn",
                    train_df=train_df,
                )
                imputed_dict[method] = res.imputed_dataset
            elif m_lower in ("iterative", "mice"):
                res = imputation_engine.impute_iterative_dataset(
                    bench_mask_result.masked_dataset,
                    experiment_id=f"{mask_config.experiment_id}_iterative",
                    train_df=train_df,
                )
                imputed_dict[method] = res.imputed_dataset
            elif m_lower in ("random_forest", "rf"):
                res = imputation_engine.impute_rf_dataset(
                    bench_mask_result.masked_dataset,
                    experiment_id=f"{mask_config.experiment_id}_rf",
                    train_df=train_df,
                )
                imputed_dict[method] = res.imputed_dataset
            else:
                raise EvaluationError(
                    f"Unsupported benchmark imputation method: {method}",
                    context={"supported_methods": self.config.supported_methods},
                )

        # 3. Compare all algorithms against the identical GroundTruthStore
        return self.compare_methods(
            imputed_results=imputed_dict,
            ground_truth_store=bench_mask_result.ground_truth_store,
            experiment_id=mask_config.experiment_id,
            dataset_version=mask_config.dataset_version,
            mask_strategy=mask_config.strategy.value,
            mask_rate=mask_config.mask_rate,
            mask_seed=mask_config.random_seed,
        )

    def run_repeated_benchmark(
        self,
        df: pd.DataFrame,
        base_mask_config: MaskingConfig,
        seeds: list[int],
        methods: list[str] | None = None,
        train_df: pd.DataFrame | None = None,
    ) -> RepeatedExperimentReport:
        """Run repeated masking experiments across multiple random seeds to measure algorithm stability."""
        if not seeds:
            raise EvaluationError(
                "Must provide at least one random seed for repeated benchmarking."
            )

        methods_to_run = methods or self.config.supported_methods
        method_maes: dict[str, list[float]] = {m: [] for m in methods_to_run}
        method_rmses: dict[str, list[float]] = {m: [] for m in methods_to_run}

        for seed in seeds:
            config_copy = deepcopy(base_mask_config)
            config_copy.random_seed = seed
            config_copy.experiment_id = f"{base_mask_config.experiment_id}_seed_{seed}"

            report = self.run_benchmark_suite(
                df=df,
                mask_config=config_copy,
                methods=methods_to_run,
                train_df=train_df,
            )

            for m in methods_to_run:
                m_res = report.method_results.get(m)
                if m_res and m_res.weighted_mae is not None and m_res.weighted_rmse is not None:
                    method_maes[m].append(m_res.weighted_mae)
                    method_rmses[m].append(m_res.weighted_rmse)

        stats: dict[str, dict[str, float]] = {}
        for m in methods_to_run:
            maes = method_maes[m]
            rmses = method_rmses[m]
            stats[m] = {
                "mean_mae": round(float(np.mean(maes)), 4) if maes else 0.0,
                "std_mae": round(float(np.std(maes)), 4) if len(maes) > 1 else 0.0,
                "mean_rmse": round(float(np.mean(rmses)), 4) if rmses else 0.0,
                "std_rmse": round(float(np.std(rmses)), 4) if len(rmses) > 1 else 0.0,
                "repetitions_count": len(maes),
            }

        return RepeatedExperimentReport(
            experiment_id=base_mask_config.experiment_id,
            repeated_seeds=seeds,
            total_repetitions=len(seeds),
            method_stats=stats,
        )
