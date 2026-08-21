"""Artificial Missingness Masking Engine Orchestrator.

Applies deterministic, reproducible artificial masking to reference datasets
to create benchmark evaluation datasets with ground-truth preservation.
"""

import numpy as np
import pandas as pd

from missing_data_platform.exceptions import ConfigurationError, DataQualityError
from missing_data_platform.ingestion.contract import RawDataContract
from missing_data_platform.logging import get_logger
from missing_data_platform.masking.config import MaskingConfig, MaskingStrategy
from missing_data_platform.masking.ground_truth import GroundTruthStore
from missing_data_platform.masking.report import (
    FeatureMaskingSummary,
    MaskingExperimentResult,
)
from missing_data_platform.masking.strategies import (
    mask_mar_covariate_conditioned,
    mask_stratified_by_group,
    mask_uniform_random,
)

logger = get_logger("masking.engine")


class MaskingEngine:
    """Orchestrates controlled artificial missingness experiments for benchmarking."""

    def __init__(
        self,
        contract: RawDataContract | None = None,
    ) -> None:
        self.contract = contract or RawDataContract.default_consumer_contract()

    def generate_benchmark_dataset(
        self,
        df: pd.DataFrame,
        config: MaskingConfig,
    ) -> MaskingExperimentResult:
        """Create an artificially masked dataset and associated ground-truth store.

        Raises:
            DataQualityError: If reference dataset is empty.
            ConfigurationError: If configuration fails validation or targets invalid columns.
        """
        if df.empty:
            raise DataQualityError(
                "Cannot apply artificial masking to an empty DataFrame.",
                context={"experiment_id": config.experiment_id},
            )

        logger.info(
            "Starting artificial missingness masking experiment",
            experiment_id=config.experiment_id,
            strategy=config.strategy.value,
            mask_rate=config.mask_rate,
            random_seed=config.random_seed,
            total_records=len(df),
        )

        # Determine target features for masking
        if config.target_features is not None:
            target_cols = config.target_features
            # Validate presence
            missing_cols = [c for c in target_cols if c not in df.columns]
            if missing_cols:
                raise ConfigurationError(
                    f"Target features not found in dataset: {missing_cols}",
                    context={"missing_columns": missing_cols},
                )
        else:
            # Default to all eligible columns (non-ID, non-target)
            target_cols = [
                col
                for col in df.columns
                if col not in config.protected_features
                and col != self.contract.id_column
                and col != self.contract.target_column
            ]

        # Enforce protected columns are never masked
        for protected_col in config.protected_features:
            if protected_col in target_cols:
                raise ConfigurationError(
                    f"Protected column '{protected_col}' cannot be targeted for artificial masking.",
                    context={"protected_column": protected_col},
                )

        rng = np.random.default_rng(config.random_seed)
        masked_df = df.copy(deep=True)
        mask_matrix = pd.DataFrame(False, index=df.index, columns=target_cols)
        ground_truth_values: dict[str, pd.Series] = {}
        feature_summaries: list[FeatureMaskingSummary] = []

        total_eligible_all = 0
        total_masked_all = 0

        for col in target_cols:
            series = df[col]
            natural_null_count = int(series.isna().sum())
            eligible_obs_count = len(series) - natural_null_count
            total_eligible_all += eligible_obs_count

            if eligible_obs_count == 0 or config.mask_rate == 0.0:
                # Column is entirely naturally missing or mask rate is 0
                feature_summaries.append(
                    FeatureMaskingSummary(
                        feature_name=col,
                        total_records=len(series),
                        natural_missing_count=natural_null_count,
                        eligible_observed_count=eligible_obs_count,
                        requested_mask_rate=config.mask_rate,
                        artificially_masked_count=0,
                        actual_mask_rate=0.0,
                        total_missing_after_masking=natural_null_count,
                    )
                )
                continue

            # Execute strategy sampling
            if config.strategy == MaskingStrategy.UNIFORM_RANDOM:
                mask_array = mask_uniform_random(
                    series=series,
                    mask_rate=config.mask_rate,
                    rng=rng,
                )
            elif config.strategy == MaskingStrategy.MAR_COVARIATE:
                covariate_col = config.conditioning_covariate
                if not covariate_col or covariate_col not in df.columns:
                    raise ConfigurationError(
                        f"Conditioning covariate '{covariate_col}' not found in DataFrame."
                    )
                mask_array = mask_mar_covariate_conditioned(
                    series=series,
                    covariate_series=df[covariate_col],
                    base_mask_rate=config.mask_rate,
                    rng=rng,
                )
            elif config.strategy == MaskingStrategy.GROUP_STRATIFIED:
                group_col = config.conditioning_covariate or "region"
                if group_col not in df.columns:
                    raise ConfigurationError(
                        f"Grouping column '{group_col}' for stratification not found in DataFrame."
                    )
                mask_array = mask_stratified_by_group(
                    series=series,
                    group_series=df[group_col],
                    mask_rate=config.mask_rate,
                    rng=rng,
                )
            else:
                raise ConfigurationError(f"Unsupported masking strategy: {config.strategy}")

            # Record mask matrix and ground truth
            mask_matrix[col] = mask_array
            masked_indices = np.where(mask_array)[0]
            masked_count = len(masked_indices)
            total_masked_all += masked_count

            # Extract ground truth original observed values
            ground_truth_values[col] = series.iloc[masked_indices].copy()

            # Apply mask to dataset copy (inserting NaN)
            masked_df.loc[df.index[masked_indices], col] = np.nan

            actual_rate = (masked_count / eligible_obs_count) if eligible_obs_count > 0 else 0.0
            feature_summaries.append(
                FeatureMaskingSummary(
                    feature_name=col,
                    total_records=len(series),
                    natural_missing_count=natural_null_count,
                    eligible_observed_count=eligible_obs_count,
                    requested_mask_rate=config.mask_rate,
                    artificially_masked_count=masked_count,
                    actual_mask_rate=round(actual_rate, 4),
                    total_missing_after_masking=natural_null_count + masked_count,
                )
            )

        overall_actual_rate = (
            (total_masked_all / total_eligible_all) if total_eligible_all > 0 else 0.0
        )

        gt_store = GroundTruthStore(
            experiment_id=config.experiment_id,
            mask_matrix=mask_matrix,
            original_values=ground_truth_values,
        )

        result = MaskingExperimentResult(
            experiment_id=config.experiment_id,
            dataset_version=config.dataset_version,
            strategy=config.strategy,
            random_seed=config.random_seed,
            requested_mask_rate=config.mask_rate,
            total_records=len(df),
            total_artificially_masked_cells=total_masked_all,
            overall_actual_mask_rate=round(overall_actual_rate, 4),
            feature_summaries=feature_summaries,
            masked_dataset=masked_df,
            ground_truth_mask=mask_matrix,
            ground_truth_store=gt_store,
        )

        logger.info(
            "Artificial missingness masking completed",
            experiment_id=config.experiment_id,
            total_masked_cells=total_masked_all,
            overall_actual_rate=round(overall_actual_rate, 4),
        )

        return result
