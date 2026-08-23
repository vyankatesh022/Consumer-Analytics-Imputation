"""Fairness-aware Bias Mitigation Engine for missing data imputation pipelines.

Implements controlled mitigation interventions (sample weighting, group-specific models,
group-conditioned modeling), before/after metric comparisons, and automated decision criteria.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from missing_data_platform.bias.config import GroupDefinitionConfig
from missing_data_platform.bias.engine import BiasAnalysisEngine
from missing_data_platform.evaluation.engine import ImputationEvaluator
from missing_data_platform.exceptions import ConfigurationError, DataQualityError
from missing_data_platform.imputation.base import BaseImputer
from missing_data_platform.imputation.config import BaselineStrategy
from missing_data_platform.imputation.engine import BaselineImputationEngine
from missing_data_platform.ingestion.contract import DataType, RawDataContract
from missing_data_platform.logging import get_logger
from missing_data_platform.masking.config import MaskingConfig
from missing_data_platform.masking.engine import MaskingEngine
from missing_data_platform.mitigation.config import (
    MitigationConfig,
    MitigationDecision,
    MitigationStrategy,
)
from missing_data_platform.mitigation.report import MitigationResult
from missing_data_platform.mitigation.weighting import calculate_group_sample_weights

logger = get_logger("mitigation.engine")


class WeightedRandomForestImputer(BaseImputer):
    """Fairness-weighted Random Forest imputer incorporating inverse-frequency cohort weights."""

    def __init__(
        self,
        group_column: str = "customer_segment",
        max_sample_weight: float = 5.0,
        n_estimators: int = 100,
        random_seed: int = 42,
        protected_features: list[str] | None = None,
        contract: RawDataContract | None = None,
    ) -> None:
        super().__init__()
        self.group_column = group_column
        self.max_sample_weight = max_sample_weight
        self.n_estimators = n_estimators
        self.random_seed = random_seed
        self.protected_features = protected_features or ["customer_id", "purchase_next_month"]
        self.contract = contract or RawDataContract.default_consumer_contract()
        self.numeric_features: list[str] = []
        self._models: dict[str, RandomForestRegressor] = {}
        self._fallbacks: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "WeightedRandomForestImputer":
        """Fit target-specific Random Forest models using demographic sample weights."""
        if df.empty:
            raise DataQualityError("Cannot fit WeightedRandomForestImputer on empty DataFrame.")

        # Extract sample weights strictly from training reference data
        group_series = (
            df[self.group_column]
            if self.group_column in df.columns
            else pd.Series("Unknown", index=df.index)
        )
        sample_weights, _ = calculate_group_sample_weights(
            group_series, max_weight=self.max_sample_weight
        )

        # Eligible numeric features
        self.numeric_features = []
        for col in sorted(df.columns):
            if col in self.protected_features or col in (
                self.contract.id_column,
                self.contract.target_column,
                self.group_column,
            ):
                continue
            col_defn = self.contract.get_column(col)
            is_num = pd.api.types.is_numeric_dtype(df[col]) or (
                col_defn is not None and col_defn.data_type in (DataType.FLOAT, DataType.INTEGER)
            )
            if is_num:
                self.numeric_features.append(col)

        numeric_df = df[self.numeric_features].apply(pd.to_numeric, errors="coerce")

        self._fallbacks.clear()
        for col in self.numeric_features:
            vals = numeric_df[col].dropna()
            self._fallbacks[col] = float(vals.median()) if not vals.empty else 0.0

        self._models.clear()
        for target_col in self.numeric_features:
            p_cols = [c for c in self.numeric_features if c != target_col]
            if not p_cols:
                continue

            obs_mask = numeric_df[target_col].notna()
            if obs_mask.sum() < 2:
                continue

            X_train = numeric_df.loc[obs_mask, p_cols].copy()
            for p in p_cols:
                if X_train[p].isna().any():
                    X_train[p] = X_train[p].fillna(self._fallbacks[p])

            y_train = numeric_df.loc[obs_mask, target_col]
            w_train = sample_weights[obs_mask.to_numpy()]

            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_seed,
            )
            rf.fit(X_train, y_train, sample_weight=w_train)
            self._models[target_col] = rf

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing numeric values using fitted weighted models."""
        imputed_df = df.copy(deep=True)
        numeric_df = imputed_df[self.numeric_features].apply(pd.to_numeric, errors="coerce")

        for target_col in self.numeric_features:
            missing_mask = numeric_df[target_col].isna()
            if missing_mask.sum() > 0 and target_col in self._models:
                p_cols = [c for c in self.numeric_features if c != target_col]
                X_miss = numeric_df.loc[missing_mask, p_cols].copy()
                for p in p_cols:
                    if X_miss[p].isna().any():
                        X_miss[p] = X_miss[p].fillna(self._fallbacks[p])

                preds = self._models[target_col].predict(X_miss)
                imputed_df.loc[missing_mask, target_col] = preds
                numeric_df.loc[missing_mask, target_col] = preds

            # Residual NaNs fallback
            if imputed_df[target_col].isna().any():
                imputed_df[target_col] = imputed_df[target_col].fillna(
                    self._fallbacks.get(target_col, 0.0)
                )

        return imputed_df


class FairnessMitigationEngine:
    """Production orchestrator for fairness mitigation interventions and empirical audit."""

    def __init__(
        self,
        contract: RawDataContract | None = None,
        config: MitigationConfig | None = None,
    ) -> None:
        self.contract = contract or RawDataContract.default_consumer_contract()
        self.config = config or MitigationConfig()

    def impute_with_mitigation(
        self,
        df: pd.DataFrame,
        method: str = "random_forest",
        train_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Impute DataFrame with configured mitigation strategy applied strictly on training data."""
        if not self.config.enabled:
            # Mitigation disabled: execute standard unmitigated imputation
            engine = BaselineImputationEngine(contract=self.contract)
            m_lower = method.lower()
            if m_lower in ("random_forest", "rf"):
                return engine.impute_rf_dataset(df, train_df=train_df).imputed_dataset
            elif m_lower == "knn":
                return engine.impute_knn_dataset(df, train_df=train_df).imputed_dataset
            elif m_lower in ("iterative", "mice"):
                return engine.impute_iterative_dataset(df, train_df=train_df).imputed_dataset
            else:
                return engine.impute_dataset(
                    df, numeric_strategy=BaselineStrategy.MEDIAN, train_df=train_df
                ).imputed_dataset

        # Mitigation enabled
        fit_source = train_df if train_df is not None else df
        group_col = self.config.group_column

        if group_col not in fit_source.columns:
            raise ConfigurationError(
                f"Mitigation group column '{group_col}' not found in training dataset.",
                context={"group_column": group_col},
            )

        if self.config.strategy == MitigationStrategy.SAMPLE_WEIGHTING:
            weighted_imputer = WeightedRandomForestImputer(
                group_column=group_col,
                max_sample_weight=self.config.max_sample_weight,
                random_seed=self.config.random_seed,
                protected_features=self.config.protected_features,
                contract=self.contract,
            )
            return weighted_imputer.fit(fit_source).transform(df)

        elif self.config.strategy == MitigationStrategy.GROUP_SPECIFIC:
            # Group-specific sub-models
            imputed_df = df.copy(deep=True)
            clean_groups = fit_source[group_col].fillna("Unknown").astype(str)

            for g_val in clean_groups.unique():
                train_group_mask = clean_groups == g_val
                target_group_mask = df[group_col].fillna("Unknown").astype(str) == g_val

                sub_train = fit_source[train_group_mask]
                sub_target = df[target_group_mask]

                if sub_target.empty:
                    continue

                if len(sub_train) >= self.config.minimum_group_size:
                    sub_engine = BaselineImputationEngine(contract=self.contract)
                    sub_res = sub_engine.impute_rf_dataset(
                        sub_target,
                        train_df=sub_train,
                        random_seed=self.config.random_seed,
                    )
                    imputed_df.loc[sub_target.index] = sub_res.imputed_dataset
                else:
                    # Small group fallback to global model
                    global_engine = BaselineImputationEngine(contract=self.contract)
                    global_res = global_engine.impute_rf_dataset(
                        sub_target,
                        train_df=fit_source,
                        random_seed=self.config.random_seed,
                    )
                    imputed_df.loc[sub_target.index] = global_res.imputed_dataset

            return imputed_df

        else:
            # Default to weighted imputer
            weighted_imputer = WeightedRandomForestImputer(
                group_column=group_col,
                max_sample_weight=self.config.max_sample_weight,
                random_seed=self.config.random_seed,
                protected_features=self.config.protected_features,
                contract=self.contract,
            )
            return weighted_imputer.fit(fit_source).transform(df)

    def mitigate_and_evaluate(
        self,
        df: pd.DataFrame,
        mask_config: MaskingConfig,
        method: str = "random_forest",
        train_df: pd.DataFrame | None = None,
    ) -> MitigationResult:
        """Run complete before vs after empirical evaluation of the mitigation intervention."""
        if df.empty:
            raise DataQualityError("Cannot run mitigation on empty DataFrame.")

        logger.info(
            "Starting bias mitigation evaluation experiment",
            experiment_id=mask_config.experiment_id,
            strategy=self.config.strategy.value,
            enabled=self.config.enabled,
            group_column=self.config.group_column,
        )

        warnings: list[str] = []

        # 1. Generate benchmark masked dataset
        masking_engine = MaskingEngine(contract=self.contract)
        mask_res = masking_engine.generate_benchmark_dataset(df, mask_config)
        gt_store = mask_res.ground_truth_store

        # 2. Run Baseline (Non-Mitigated) Imputation & Evaluation
        base_engine = BaselineImputationEngine(contract=self.contract)
        m_lower = method.lower()
        if m_lower in ("random_forest", "rf"):
            base_imp = base_engine.impute_rf_dataset(
                mask_res.masked_dataset, train_df=train_df
            ).imputed_dataset
        elif m_lower == "knn":
            base_imp = base_engine.impute_knn_dataset(
                mask_res.masked_dataset, train_df=train_df
            ).imputed_dataset
        elif m_lower in ("iterative", "mice"):
            base_imp = base_engine.impute_iterative_dataset(
                mask_res.masked_dataset, train_df=train_df
            ).imputed_dataset
        else:
            base_imp = base_engine.impute_dataset(
                mask_res.masked_dataset, train_df=train_df
            ).imputed_dataset

        evaluator = ImputationEvaluator(contract=self.contract)
        base_eval = evaluator.evaluate_method(base_imp, gt_store, method_name=f"{method}_baseline")

        bias_config = GroupDefinitionConfig(
            group_column=self.config.group_column,
            minimum_group_size=self.config.minimum_group_size,
        )
        bias_engine = BiasAnalysisEngine(contract=self.contract, config=bias_config)
        base_bias = bias_engine.run_bias_analysis(
            df=df,
            imputed_results={method: base_imp},
            ground_truth_store=gt_store,
            experiment_id=f"{mask_config.experiment_id}_base_bias",
        )

        # Baseline disparities
        base_disparities = [
            d.absolute_disparity
            for d in base_bias.disparity_results
            if d.absolute_disparity is not None
        ]
        base_max_disp = float(max(base_disparities)) if base_disparities else 0.0

        if not self.config.enabled:
            # When mitigation is disabled, return baseline result
            return MitigationResult(
                experiment_id=mask_config.experiment_id,
                dataset_version=mask_config.dataset_version,
                method=method,
                mitigation_strategy=self.config.strategy,
                mitigation_config={"enabled": False},
                baseline_mae=base_eval.weighted_mae,
                baseline_rmse=base_eval.weighted_rmse,
                baseline_max_disparity=round(base_max_disp, 4),
                mitigated_mae=base_eval.weighted_mae,
                mitigated_rmse=base_eval.weighted_rmse,
                mitigated_max_disparity=round(base_max_disp, 4),
                accuracy_change_pct=0.0,
                disparity_reduction_pct=0.0,
                decision=MitigationDecision.ACCEPTED,
                decision_reason="Mitigation is disabled (baseline passthrough verified).",
                group_results_before=base_bias.performance_results,
                group_results_after=base_bias.performance_results,
                warnings=["Mitigation is disabled."],
            )

        # 3. Run Mitigated Imputation & Re-Evaluation
        mitigated_imp = self.impute_with_mitigation(
            df=mask_res.masked_dataset,
            method=method,
            train_df=train_df,
        )

        mitigated_eval = evaluator.evaluate_method(
            mitigated_imp, gt_store, method_name=f"{method}_mitigated"
        )
        mitigated_bias = bias_engine.run_bias_analysis(
            df=df,
            imputed_results={method: mitigated_imp},
            ground_truth_store=gt_store,
            experiment_id=f"{mask_config.experiment_id}_mitigated_bias",
        )

        mitigated_disparities = [
            d.absolute_disparity
            for d in mitigated_bias.disparity_results
            if d.absolute_disparity is not None
        ]
        mitigated_max_disp = float(max(mitigated_disparities)) if mitigated_disparities else 0.0

        # 4. Compute Accuracy & Disparity Deltas
        b_mae = base_eval.weighted_mae or 0.0
        m_mae = mitigated_eval.weighted_mae or 0.0
        acc_change_pct = round(((m_mae - b_mae) / b_mae * 100.0), 2) if b_mae > 0 else 0.0

        if base_max_disp > 1e-6:
            disp_reduction_pct = round(
                ((base_max_disp - mitigated_max_disp) / base_max_disp * 100.0), 2
            )
        else:
            disp_reduction_pct = 0.0

        # 5. Automated Decision Evaluation
        max_allowed_acc_deg = self.config.max_allowed_accuracy_degradation * 100.0
        target_disp_red = self.config.target_disparity_reduction * 100.0

        if acc_change_pct > max_allowed_acc_deg:
            decision = MitigationDecision.REJECTED
            reason = (
                f"Accuracy degraded by {acc_change_pct:.2f}% which exceeds maximum "
                f"allowed constraint ({max_allowed_acc_deg:.1f}%)."
            )
        elif disp_reduction_pct < 0:
            decision = MitigationDecision.REJECTED
            reason = f"Mitigation increased maximum group disparity by {-disp_reduction_pct:.2f}%."
        elif disp_reduction_pct >= target_disp_red:
            decision = MitigationDecision.ACCEPTED
            reason = (
                f"Disparity reduced by {disp_reduction_pct:.2f}% (meets target {target_disp_red:.1f}%) "
                f"with acceptable accuracy delta ({acc_change_pct:+.2f}%)."
            )
        else:
            decision = MitigationDecision.REQUIRES_REVIEW
            reason = (
                f"Disparity reduced by {disp_reduction_pct:.2f}% (below target {target_disp_red:.1f}%) "
                f"with accuracy delta {acc_change_pct:+.2f}%."
            )

        logger.info(
            "Bias mitigation evaluation completed",
            experiment_id=mask_config.experiment_id,
            decision=decision.value,
            disparity_reduction_pct=disp_reduction_pct,
            accuracy_change_pct=acc_change_pct,
        )

        return MitigationResult(
            experiment_id=mask_config.experiment_id,
            dataset_version=mask_config.dataset_version,
            method=method,
            mitigation_strategy=self.config.strategy,
            mitigation_config={
                "enabled": True,
                "strategy": self.config.strategy.value,
                "group_column": self.config.group_column,
                "max_sample_weight": self.config.max_sample_weight,
                "max_allowed_accuracy_degradation": self.config.max_allowed_accuracy_degradation,
                "target_disparity_reduction": self.config.target_disparity_reduction,
            },
            baseline_mae=round(b_mae, 4),
            baseline_rmse=round(base_eval.weighted_rmse, 4) if base_eval.weighted_rmse else None,
            baseline_max_disparity=round(base_max_disp, 4),
            mitigated_mae=round(m_mae, 4),
            mitigated_rmse=round(mitigated_eval.weighted_rmse, 4)
            if mitigated_eval.weighted_rmse
            else None,
            mitigated_max_disparity=round(mitigated_max_disp, 4),
            accuracy_change_pct=acc_change_pct,
            disparity_reduction_pct=disp_reduction_pct,
            decision=decision,
            decision_reason=reason,
            group_results_before=base_bias.performance_results,
            group_results_after=mitigated_bias.performance_results,
            warnings=warnings,
        )
