"""Bias and Representation Analysis Engine Orchestrator.

Measures demographic and segment group missingness disparities, representation balance,
and imputation accuracy variance across population groups without modifying imputation algorithms.
"""

import numpy as np
import pandas as pd

from missing_data_platform.bias.config import GroupDefinitionConfig, MissingGroupPolicy
from missing_data_platform.bias.report import (
    BiasAnalysisResult,
    DisparityResult,
    GroupImputationPerformance,
    GroupMissingness,
    GroupRepresentation,
)
from missing_data_platform.evaluation.metrics import (
    calculate_accuracy,
    calculate_mae,
    calculate_rmse,
    validate_and_filter_predictions,
)
from missing_data_platform.exceptions import ConfigurationError, DataQualityError
from missing_data_platform.imputation.report import ImputationResult
from missing_data_platform.ingestion.contract import DataType, RawDataContract
from missing_data_platform.logging import get_logger
from missing_data_platform.masking.ground_truth import GroundTruthStore
from missing_data_platform.missingness.group_analysis import create_age_bands

logger = get_logger("bias.engine")


class BiasAnalysisEngine:
    """Production engine for measuring group-level representation and imputation disparities."""

    def __init__(
        self,
        contract: RawDataContract | None = None,
        config: GroupDefinitionConfig | None = None,
    ) -> None:
        self.contract = contract or RawDataContract.default_consumer_contract()
        self.config = config or GroupDefinitionConfig()

    def extract_group_series(self, df: pd.DataFrame) -> pd.Series:
        """Extract and resolve grouping column series according to configured policies."""
        group_col = self.config.group_column
        if group_col not in df.columns:
            raise ConfigurationError(
                f"Grouping column '{group_col}' not found in DataFrame.",
                context={"group_column": group_col, "available_columns": list(df.columns)},
            )

        group_series = create_age_bands(df["age"]) if group_col == "age" else df[group_col].copy()

        # Handle missing group values
        if self.config.missing_group_policy == MissingGroupPolicy.UNKNOWN:
            group_series = group_series.fillna("Unknown").astype(str)
        else:
            # If policy is EXCLUDE, leave missing as None/NaN
            group_series = group_series.astype(object)

        return group_series

    def analyze_representation(
        self,
        df: pd.DataFrame,
        ground_truth_store: GroundTruthStore | None = None,
        group_series: pd.Series | None = None,
    ) -> list[GroupRepresentation]:
        """Measure demographic/segment group population counts and evaluation cell distribution."""
        if group_series is None:
            group_series = self.extract_group_series(df)

        valid_mask = group_series.notna()
        clean_groups = group_series[valid_mask]
        total_pop = len(clean_groups)

        if total_pop == 0:
            return []

        # Count evaluation cells per group if ground_truth_store is available
        total_eval_cells = ground_truth_store.total_masked_cells if ground_truth_store else 0
        group_eval_counts: dict[str, int] = {str(g): 0 for g in clean_groups.unique()}

        if ground_truth_store and total_eval_cells > 0:
            for col in ground_truth_store.mask_matrix.columns:
                gt_series = ground_truth_store.get_ground_truth(col)
                if gt_series.empty:
                    continue
                # For each masked row index, increment group count
                for idx in gt_series.index:
                    if idx in group_series.index and pd.notna(group_series.loc[idx]):
                        g_name = str(group_series.loc[idx])
                        group_eval_counts[g_name] = group_eval_counts.get(g_name, 0) + 1

        rep_results: list[GroupRepresentation] = []
        for g_val, count in clean_groups.value_counts().items():
            g_str = str(g_val)
            pop_pct = round((count / total_pop) * 100.0, 2)
            eval_cells = group_eval_counts.get(g_str, 0)
            eval_pct = (
                round((eval_cells / total_eval_cells) * 100.0, 2) if total_eval_cells > 0 else 0.0
            )

            rep_results.append(
                GroupRepresentation(
                    group_value=g_str,
                    population_count=int(count),
                    population_percentage=pop_pct,
                    eligible_evaluation_cells=eval_cells,
                    evaluation_percentage=eval_pct,
                    is_small_group=bool(count < self.config.minimum_group_size),
                )
            )

        return rep_results

    def analyze_group_missingness(
        self,
        df: pd.DataFrame,
        group_series: pd.Series | None = None,
        target_features: list[str] | None = None,
    ) -> list[GroupMissingness]:
        """Measure natural missingness rates across groups for target features."""
        if group_series is None:
            group_series = self.extract_group_series(df)

        if target_features is not None:
            features = target_features
        else:
            features = [
                col
                for col in df.columns
                if col != self.config.group_column
                and col != self.contract.id_column
                and col != self.contract.target_column
            ]

        missing_results: list[GroupMissingness] = []
        valid_mask = group_series.notna()
        unique_groups = group_series[valid_mask].unique()

        for feat in features:
            if feat not in df.columns:
                continue

            for g_val in unique_groups:
                g_str = str(g_val)
                group_row_mask = valid_mask & (group_series == g_val)
                sub_series = df.loc[group_row_mask, feat]
                pop = len(sub_series)
                if pop == 0:
                    continue

                miss_count = int(sub_series.isna().sum())
                obs_count = pop - miss_count
                miss_rate = round((miss_count / pop) * 100.0, 2)

                missing_results.append(
                    GroupMissingness(
                        group_value=g_str,
                        feature_name=feat,
                        missing_count=miss_count,
                        observed_count=obs_count,
                        missing_rate=miss_rate,
                        is_small_group=bool(pop < self.config.minimum_group_size),
                    )
                )

        return missing_results

    def analyze_imputation_by_group(
        self,
        imputed_results: dict[str, pd.DataFrame | ImputationResult],
        ground_truth_store: GroundTruthStore,
        group_series: pd.Series,
    ) -> list[GroupImputationPerformance]:
        """Measure imputation error metrics (MAE, RMSE, Accuracy) separately for each group."""
        perf_results: list[GroupImputationPerformance] = []
        valid_mask = group_series.notna()
        unique_groups = group_series[valid_mask].unique()

        for method_name, imp_data in imputed_results.items():
            imputed_df = (
                imp_data.imputed_dataset if isinstance(imp_data, ImputationResult) else imp_data
            )

            for col in ground_truth_store.mask_matrix.columns:
                gt_series = ground_truth_store.get_ground_truth(col)
                if gt_series.empty or col not in imputed_df.columns:
                    continue

                col_defn = self.contract.get_column(col)
                is_numeric = pd.api.types.is_numeric_dtype(gt_series) or (
                    col_defn is not None
                    and col_defn.data_type in (DataType.FLOAT, DataType.INTEGER)
                )

                for g_val in unique_groups:
                    g_str = str(g_val)
                    # Filter masked indices belonging to this group
                    group_indices = [
                        idx
                        for idx in gt_series.index
                        if idx in group_series.index and group_series.loc[idx] == g_val
                    ]

                    sample_count = len(group_indices)
                    if sample_count == 0:
                        continue

                    is_suppressed = sample_count < self.config.minimum_group_size
                    warning = "INSUFFICIENT_SAMPLE_SIZE" if is_suppressed else None

                    sub_gt = gt_series.loc[group_indices]
                    sub_pred = imputed_df.loc[group_indices, col]

                    clean_true, clean_pred, missing_count, invalid_count = (
                        validate_and_filter_predictions(sub_gt, sub_pred)
                    )
                    valid_preds = len(clean_true)
                    total_missing_preds = missing_count + invalid_count

                    if is_numeric:
                        mae_val = (
                            round(calculate_mae(clean_true, clean_pred), 4)
                            if valid_preds > 0
                            else None
                        )
                        rmse_val = (
                            round(calculate_rmse(clean_true, clean_pred), 4)
                            if valid_preds > 0
                            else None
                        )

                        perf_results.append(
                            GroupImputationPerformance(
                                group_value=g_str,
                                method=method_name,
                                feature_name=col,
                                metric_name="MAE",
                                metric_value=mae_val if not is_suppressed else None,
                                sample_count=sample_count,
                                valid_prediction_count=valid_preds,
                                missing_prediction_count=total_missing_preds,
                                is_suppressed=is_suppressed,
                                warning=warning,
                            )
                        )
                        perf_results.append(
                            GroupImputationPerformance(
                                group_value=g_str,
                                method=method_name,
                                feature_name=col,
                                metric_name="RMSE",
                                metric_value=rmse_val if not is_suppressed else None,
                                sample_count=sample_count,
                                valid_prediction_count=valid_preds,
                                missing_prediction_count=total_missing_preds,
                                is_suppressed=is_suppressed,
                                warning=warning,
                            )
                        )
                    else:
                        acc_val = (
                            round(calculate_accuracy(clean_true, clean_pred), 4)
                            if valid_preds > 0
                            else None
                        )
                        perf_results.append(
                            GroupImputationPerformance(
                                group_value=g_str,
                                method=method_name,
                                feature_name=col,
                                metric_name="Accuracy",
                                metric_value=acc_val if not is_suppressed else None,
                                sample_count=sample_count,
                                valid_prediction_count=valid_preds,
                                missing_prediction_count=total_missing_preds,
                                is_suppressed=is_suppressed,
                                warning=warning,
                            )
                        )

        return perf_results

    def calculate_disparities(
        self,
        performance_results: list[GroupImputationPerformance],
    ) -> list[DisparityResult]:
        """Compute pairwise disparity metrics (absolute and relative differences) between groups."""
        # Index performance results by (method, feature_name, metric_name)
        grouped_entries: dict[tuple[str, str, str], list[GroupImputationPerformance]] = {}
        for p in performance_results:
            key = (p.method, p.feature_name, p.metric_name)
            grouped_entries.setdefault(key, []).append(p)

        disparity_results: list[DisparityResult] = []

        for (method, feat, metric), entries in grouped_entries.items():
            # Consider only non-suppressed entries with valid metric values
            valid_entries = [
                e for e in entries if not e.is_suppressed and e.metric_value is not None
            ]
            n_entries = len(valid_entries)

            for i in range(n_entries):
                for j in range(i + 1, n_entries):
                    item_a = valid_entries[i]
                    item_b = valid_entries[j]

                    val_a = item_a.metric_value
                    val_b = item_b.metric_value

                    if val_a is not None and val_b is not None:
                        abs_diff = round(abs(val_a - val_b), 4)
                        # Relative disparity = (val_a - val_b) / val_b if denominator is valid
                        rel_diff = round((val_a - val_b) / val_b, 4) if abs(val_b) > 1e-06 else None
                    else:
                        abs_diff = None
                        rel_diff = None

                    comp_name = f"{item_a.group_value} vs {item_b.group_value}"
                    disparity_results.append(
                        DisparityResult(
                            comparison_name=comp_name,
                            method=method,
                            feature_name=feat,
                            metric_name=metric,
                            group_a=item_a.group_value,
                            group_b=item_b.group_value,
                            value_group_a=val_a,
                            value_group_b=val_b,
                            absolute_disparity=abs_diff,
                            relative_disparity=rel_diff,
                            sample_count_a=item_a.sample_count,
                            sample_count_b=item_b.sample_count,
                        )
                    )

        return disparity_results

    def run_bias_analysis(
        self,
        df: pd.DataFrame,
        imputed_results: dict[str, pd.DataFrame | ImputationResult],
        ground_truth_store: GroundTruthStore,
        experiment_id: str = "bias_analysis_exp",
        dataset_version: str = "v1.0",
    ) -> BiasAnalysisResult:
        """Orchestrate complete group representation, missingness, and performance disparity analysis."""
        if df.empty:
            raise DataQualityError("Cannot run bias analysis on an empty DataFrame.")

        logger.info(
            "Starting bias and representation analysis",
            experiment_id=experiment_id,
            grouping_column=self.config.group_column,
            minimum_group_size=self.config.minimum_group_size,
            methods_count=len(imputed_results),
        )

        warnings: list[str] = []
        group_series = self.extract_group_series(df)

        # 1. Representation analysis
        rep_results = self.analyze_representation(
            df=df,
            ground_truth_store=ground_truth_store,
            group_series=group_series,
        )

        for rep in rep_results:
            if rep.is_small_group:
                warnings.append(
                    f"Group '{rep.group_value}' has population {rep.population_count} < minimum_group_size={self.config.minimum_group_size}."
                )

        # 2. Missingness analysis
        missing_results = self.analyze_group_missingness(
            df=df,
            group_series=group_series,
            target_features=self.config.target_features,
        )

        # 3. Performance analysis by group
        perf_results = self.analyze_imputation_by_group(
            imputed_results=imputed_results,
            ground_truth_store=ground_truth_store,
            group_series=group_series,
        )

        # 4. Pairwise disparities
        disparity_results = self.calculate_disparities(perf_results)

        # 5. Determine best method per group and global best
        best_method_per_group: dict[str, str] = {}
        unique_groups = [r.group_value for r in rep_results if not r.is_small_group]

        for g_val in unique_groups:
            group_mae_scores: dict[str, list[float]] = {}
            for p in perf_results:
                if p.group_value == g_val and p.metric_name == "MAE" and p.metric_value is not None:
                    group_mae_scores.setdefault(p.method, []).append(p.metric_value)

            if group_mae_scores:
                mean_group_maes = {
                    m: float(np.mean(scores)) for m, scores in group_mae_scores.items()
                }
                best_m = min(mean_group_maes, key=lambda m: mean_group_maes[m])
                best_method_per_group[g_val] = best_m

        # Global best across all non-suppressed MAE scores
        global_mae_scores: dict[str, list[float]] = {}
        for p in perf_results:
            if p.metric_name == "MAE" and p.metric_value is not None:
                global_mae_scores.setdefault(p.method, []).append(p.metric_value)

        if global_mae_scores:
            global_mean_maes = {
                m: float(np.mean(scores)) for m, scores in global_mae_scores.items()
            }
            global_best = min(global_mean_maes, key=lambda m: global_mean_maes[m])
        else:
            global_best = None

        logger.info(
            "Bias and representation analysis completed",
            experiment_id=experiment_id,
            groups_analyzed=len(rep_results),
            disparities_computed=len(disparity_results),
            warnings_count=len(warnings),
        )

        return BiasAnalysisResult(
            experiment_id=experiment_id,
            dataset_version=dataset_version,
            grouping_column=self.config.group_column,
            minimum_group_size=self.config.minimum_group_size,
            representation_results=rep_results,
            missingness_results=missing_results,
            performance_results=perf_results,
            disparity_results=disparity_results,
            best_method_per_group=best_method_per_group,
            global_best_method=global_best,
            warnings=warnings,
        )
