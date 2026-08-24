"""Production ML Pipeline Orchestrator with stage boundaries, checkpointing, and fault tolerance."""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from missing_data_platform.bias.engine import BiasAnalysisEngine
from missing_data_platform.downstream.engine import DownstreamEvaluationEngine
from missing_data_platform.evaluation.engine import ImputationEvaluator
from missing_data_platform.exceptions import (
    ConfigurationError,
    DataQualityError,
    ImputationError,
    PlatformError,
)
from missing_data_platform.imputation.config import BaselineStrategy
from missing_data_platform.imputation.engine import BaselineImputationEngine
from missing_data_platform.ingestion.contract import RawDataContract
from missing_data_platform.ingestion.validator import SchemaValidator
from missing_data_platform.logging import get_logger
from missing_data_platform.masking.engine import MaskingEngine
from missing_data_platform.mitigation.engine import FairnessMitigationEngine
from missing_data_platform.orchestration.checkpoints import CheckpointManager
from missing_data_platform.orchestration.config import ExperimentPipelineConfig
from missing_data_platform.orchestration.fingerprint import (
    calculate_dataset_fingerprint,
    get_environment_info,
)
from missing_data_platform.orchestration.manifest import (
    ArtifactCategory,
    ArtifactReference,
    ExperimentManifest,
)
from missing_data_platform.orchestration.stages import (
    ErrorCode,
    MethodExecutionStatus,
    PipelineStage,
    StageRecord,
    StageStateMachine,
    StageStatus,
)
from missing_data_platform.quality.engine import DataQualityEngine

logger = get_logger("orchestration.engine")


class PipelineOrchestrator:
    """Enterprise-grade experiment pipeline orchestrator enforcing strict reproducibility and validation boundaries."""

    def __init__(
        self,
        contract: RawDataContract | None = None,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> None:
        self.contract = contract or RawDataContract.default_consumer_contract()
        self._custom_checkpoint_manager = checkpoint_manager

    def _create_artifact_ref(
        self, name: str, category: ArtifactCategory, payload: Any, output_dir: Path
    ) -> ArtifactReference:
        """Persist payload to output directory and generate cryptographic ArtifactReference."""
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"{name}.json"

        if isinstance(payload, str):
            content_str = payload
        elif hasattr(payload, "to_json"):
            content_str = payload.to_json()
        elif hasattr(payload, "to_dict"):
            content_str = json.dumps(payload.to_dict(), indent=2, default=str)
        else:
            content_str = json.dumps(payload, indent=2, default=str)

        file_path.write_text(content_str, encoding="utf-8")
        sha = hashlib.sha256(content_str.encode("utf-8")).hexdigest()
        size_bytes = file_path.stat().st_size

        return ArtifactReference(
            name=name,
            category=category,
            path=str(file_path),
            sha256_hash=sha,
            size_bytes=size_bytes,
        )

    def execute_pipeline(
        self,
        df: pd.DataFrame,
        config: ExperimentPipelineConfig,
        run_id: str | None = None,
    ) -> ExperimentManifest:
        """Execute the end-to-end 11-stage hardened experiment pipeline."""
        start_time_utc = datetime.now(UTC).isoformat()
        t0_total = time.perf_counter()

        actual_run_id = run_id or f"run_{int(time.time())}"
        config_fp = config.get_config_fingerprint()
        env_info = get_environment_info()

        checkpoint_mgr = self._custom_checkpoint_manager or CheckpointManager(
            checkpoint_dir=config.execution.checkpoint_dir
        )
        self.checkpoint_manager = checkpoint_mgr

        logger.info(
            "Starting production experiment pipeline",
            experiment_id=config.experiment_id,
            run_id=actual_run_id,
            dataset_version=config.dataset_version,
        )

        stage_records: dict[PipelineStage, StageRecord] = {
            s: StageRecord(stage=s) for s in PipelineStage
        }
        artifact_refs: list[ArtifactReference] = []
        warnings_list: list[str] = []
        method_statuses: dict[str, MethodExecutionStatus] = {}
        partial_failure = False

        output_dir = Path(config.execution.output_dir) / config.experiment_id / actual_run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        current_stage = PipelineStage.ENVIRONMENT_VALIDATION
        dataset_fp = "uncomputed"

        try:
            # =========================================================================
            # STAGE 1: ENVIRONMENT_VALIDATION
            # =========================================================================
            current_stage = PipelineStage.ENVIRONMENT_VALIDATION
            st_rec = stage_records[current_stage]
            sm = StageStateMachine()
            sm.transition_to(StageStatus.RUNNING)
            st_rec.status = StageStatus.RUNNING
            st_rec.start_time_utc = datetime.now(UTC).isoformat()
            t0_stage = time.perf_counter()

            # Resource Quotas check
            if len(df) > config.resource_limits.max_records:
                raise ConfigurationError(
                    f"Dataset records ({len(df)}) exceed max_records limit ({config.resource_limits.max_records})."
                )
            if len(df.columns) > config.resource_limits.max_features:
                raise ConfigurationError(
                    f"Dataset features ({len(df.columns)}) exceed max_features limit ({config.resource_limits.max_features})."
                )
            if config.execution.clean_worktree_required and not env_info.get(
                "clean_worktree", False
            ):
                raise PlatformError("Clean worktree required but uncommitted changes detected.")

            st_rec.duration_seconds = round(time.perf_counter() - t0_stage, 4)
            st_rec.end_time_utc = datetime.now(UTC).isoformat()
            sm.transition_to(StageStatus.COMPLETED)
            st_rec.status = StageStatus.COMPLETED

            # =========================================================================
            # STAGE 2: DATASET_VALIDATION
            # =========================================================================
            current_stage = PipelineStage.DATASET_VALIDATION
            st_rec = stage_records[current_stage]
            sm = StageStateMachine()
            sm.transition_to(StageStatus.RUNNING)
            st_rec.status = StageStatus.RUNNING
            st_rec.start_time_utc = datetime.now(UTC).isoformat()
            t0_stage = time.perf_counter()

            if df.empty:
                raise DataQualityError("Cannot run experiment on an empty DataFrame.")

            # Schema validation
            validator = SchemaValidator(contract=self.contract)
            val_res = validator.validate(df)
            if not val_res.is_valid:
                raise DataQualityError(
                    "Schema validation failed during dataset validation stage.",
                    context={"errors": val_res.validation_errors},
                )

            clean_df = val_res.valid_df

            # Quality audit
            qual_engine = DataQualityEngine(contract=self.contract)
            qual_report = qual_engine.audit_dataset(clean_df, dataset_id=config.dataset_version)
            if qual_report.has_failures:
                raise DataQualityError(
                    "Data quality audit failed with critical severity errors.",
                    context={"failed_checks": qual_report.failed_checks},
                )

            dataset_fp = calculate_dataset_fingerprint(clean_df, self.contract)
            artifact_refs.append(
                self._create_artifact_ref(
                    "data_quality_report", ArtifactCategory.EVALUATION, qual_report, output_dir
                )
            )

            st_rec.duration_seconds = round(time.perf_counter() - t0_stage, 4)
            st_rec.end_time_utc = datetime.now(UTC).isoformat()
            sm.transition_to(StageStatus.COMPLETED)
            st_rec.status = StageStatus.COMPLETED

            # =========================================================================
            # STAGE 3: EXPERIMENT_INITIALIZATION
            # =========================================================================
            current_stage = PipelineStage.EXPERIMENT_INITIALIZATION
            st_rec = stage_records[current_stage]
            sm = StageStateMachine()
            sm.transition_to(StageStatus.RUNNING)
            st_rec.status = StageStatus.RUNNING
            st_rec.start_time_utc = datetime.now(UTC).isoformat()
            t0_stage = time.perf_counter()

            # Save immutable config snapshot
            config_snapshot = config.to_sanitized_dict()
            artifact_refs.append(
                self._create_artifact_ref(
                    "config_snapshot", ArtifactCategory.CONFIG, config_snapshot, output_dir
                )
            )

            st_rec.duration_seconds = round(time.perf_counter() - t0_stage, 4)
            st_rec.end_time_utc = datetime.now(UTC).isoformat()
            sm.transition_to(StageStatus.COMPLETED)
            st_rec.status = StageStatus.COMPLETED

            # =========================================================================
            # STAGE 4: MASKING
            # =========================================================================
            current_stage = PipelineStage.MASKING
            st_rec = stage_records[current_stage]
            sm = StageStateMachine()
            st_rec.start_time_utc = datetime.now(UTC).isoformat()
            t0_stage = time.perf_counter()

            # Checkpoint resume check
            resumed_mask = None
            if config.execution.resume_from_checkpoint:
                chk = self.checkpoint_manager.load_checkpoint(
                    PipelineStage.MASKING, config.experiment_id, actual_run_id
                )
                if chk:
                    valid, reason = self.checkpoint_manager.verify_integrity(
                        chk, dataset_fp, config_fp
                    )
                    if valid:
                        resumed_mask = chk
                        logger.info(
                            "Resuming MASKING stage from verified checkpoint", reason=reason
                        )

            if resumed_mask:
                sm.transition_to(StageStatus.SKIPPED)
                st_rec.status = StageStatus.SKIPPED
                st_rec.details["resumed_from_checkpoint"] = True
                mask_res = resumed_mask.payload.get("mask_result")
                # Re-generate if payload was metadata only
                masking_engine = MaskingEngine(contract=self.contract)
                mask_res = masking_engine.generate_benchmark_dataset(clean_df, config.masking)
            else:
                sm.transition_to(StageStatus.RUNNING)
                st_rec.status = StageStatus.RUNNING
                masking_engine = MaskingEngine(contract=self.contract)
                mask_res = masking_engine.generate_benchmark_dataset(clean_df, config.masking)

                # Output validation
                if mask_res.ground_truth_store.total_masked_cells == 0:
                    raise PlatformError("Masking generated 0 masked cells for evaluation.")

                if config.execution.enable_checkpointing:
                    self.checkpoint_manager.save_checkpoint(
                        stage=PipelineStage.MASKING,
                        experiment_id=config.experiment_id,
                        run_id=actual_run_id,
                        dataset_fingerprint=dataset_fp,
                        config_fingerprint=config_fp,
                        payload={
                            "total_masked_cells": mask_res.ground_truth_store.total_masked_cells,
                            "mask_rate": config.masking.mask_rate,
                        },
                    )
                sm.transition_to(StageStatus.COMPLETED)
                st_rec.status = StageStatus.COMPLETED

            st_rec.duration_seconds = round(time.perf_counter() - t0_stage, 4)
            st_rec.end_time_utc = datetime.now(UTC).isoformat()

            # =========================================================================
            # STAGE 5: IMPUTATION
            # =========================================================================
            current_stage = PipelineStage.IMPUTATION
            st_rec = stage_records[current_stage]
            sm = StageStateMachine()
            sm.transition_to(StageStatus.RUNNING)
            st_rec.status = StageStatus.RUNNING
            st_rec.start_time_utc = datetime.now(UTC).isoformat()
            t0_stage = time.perf_counter()

            imputed_datasets: dict[str, pd.DataFrame] = {}
            imputation_engine = BaselineImputationEngine(contract=self.contract)

            for method in config.imputation_methods:
                m_status = MethodExecutionStatus(method=method, status=StageStatus.RUNNING)
                t0_m = time.perf_counter()
                try:
                    m_lower = method.lower()
                    if m_lower in ("baseline_median", "median"):
                        imp_out = imputation_engine.impute_dataset(
                            mask_res.masked_dataset,
                            experiment_id=f"{config.experiment_id}_median",
                            numeric_strategy=BaselineStrategy.MEDIAN,
                        ).imputed_dataset
                    elif m_lower in ("baseline_mean", "mean"):
                        imp_out = imputation_engine.impute_dataset(
                            mask_res.masked_dataset,
                            experiment_id=f"{config.experiment_id}_mean",
                            numeric_strategy=BaselineStrategy.MEAN,
                        ).imputed_dataset
                    elif m_lower == "knn":
                        imp_out = imputation_engine.impute_knn_dataset(
                            mask_res.masked_dataset,
                            experiment_id=f"{config.experiment_id}_knn",
                        ).imputed_dataset
                    elif m_lower in ("iterative", "mice"):
                        imp_out = imputation_engine.impute_iterative_dataset(
                            mask_res.masked_dataset,
                            experiment_id=f"{config.experiment_id}_iterative",
                            random_seed=config.random_seed,
                        ).imputed_dataset
                    elif m_lower in ("random_forest", "rf"):
                        imp_out = imputation_engine.impute_rf_dataset(
                            mask_res.masked_dataset,
                            experiment_id=f"{config.experiment_id}_rf",
                            random_seed=config.random_seed,
                        ).imputed_dataset
                    else:
                        raise ImputationError(f"Unsupported algorithm: '{method}'")

                    # Validate stage output
                    if len(imp_out) != len(clean_df):
                        raise ImputationError(
                            f"Imputed row count ({len(imp_out)}) does not match source ({len(clean_df)})."
                        )

                    imputed_datasets[method] = imp_out
                    dur = time.perf_counter() - t0_m
                    m_status.mark_completed(dur)
                except Exception as e:
                    dur = time.perf_counter() - t0_m
                    m_status.mark_failed(dur, ErrorCode.METHOD_FAILURE, str(e))
                    partial_failure = True
                    warnings_list.append(f"Method '{method}' failed: {e}")
                    if (
                        config.execution.fail_fast
                        and not config.execution.allow_partial_imputation_failure
                    ):
                        raise

                method_statuses[method] = m_status

            if not imputed_datasets:
                raise ImputationError("All requested imputation methods failed during execution.")

            if config.execution.enable_checkpointing:
                self.checkpoint_manager.save_checkpoint(
                    stage=PipelineStage.IMPUTATION,
                    experiment_id=config.experiment_id,
                    run_id=actual_run_id,
                    dataset_fingerprint=dataset_fp,
                    config_fingerprint=config_fp,
                    payload={"successful_methods": list(imputed_datasets.keys())},
                )

            st_rec.duration_seconds = round(time.perf_counter() - t0_stage, 4)
            st_rec.end_time_utc = datetime.now(UTC).isoformat()
            sm.transition_to(StageStatus.COMPLETED)
            st_rec.status = StageStatus.COMPLETED

            # =========================================================================
            # STAGE 6: MITIGATION
            # =========================================================================
            current_stage = PipelineStage.MITIGATION
            st_rec = stage_records[current_stage]
            sm = StageStateMachine()
            st_rec.start_time_utc = datetime.now(UTC).isoformat()
            t0_stage = time.perf_counter()

            mitigated_df = None
            if config.mitigation.enabled:
                sm.transition_to(StageStatus.RUNNING)
                st_rec.status = StageStatus.RUNNING
                mit_engine = FairnessMitigationEngine(
                    contract=self.contract, config=config.mitigation
                )
                mitigated_df = mit_engine.impute_with_mitigation(mask_res.masked_dataset)

                if config.execution.enable_checkpointing:
                    self.checkpoint_manager.save_checkpoint(
                        stage=PipelineStage.MITIGATION,
                        experiment_id=config.experiment_id,
                        run_id=actual_run_id,
                        dataset_fingerprint=dataset_fp,
                        config_fingerprint=config_fp,
                        payload={"mitigation_strategy": config.mitigation.strategy.value},
                    )
                sm.transition_to(StageStatus.COMPLETED)
                st_rec.status = StageStatus.COMPLETED
            else:
                sm.transition_to(StageStatus.SKIPPED)
                st_rec.status = StageStatus.SKIPPED

            st_rec.duration_seconds = round(time.perf_counter() - t0_stage, 4)
            st_rec.end_time_utc = datetime.now(UTC).isoformat()

            # =========================================================================
            # STAGE 7: IMPUTATION_EVALUATION
            # =========================================================================
            current_stage = PipelineStage.IMPUTATION_EVALUATION
            st_rec = stage_records[current_stage]
            sm = StageStateMachine()
            sm.transition_to(StageStatus.RUNNING)
            st_rec.status = StageStatus.RUNNING
            st_rec.start_time_utc = datetime.now(UTC).isoformat()
            t0_stage = time.perf_counter()

            evaluator = ImputationEvaluator(contract=self.contract, config=config.evaluation)
            benchmark_report = evaluator.compare_methods(
                imputed_results=imputed_datasets,
                ground_truth_store=mask_res.ground_truth_store,
                experiment_id=config.experiment_id,
                dataset_version=config.dataset_version,
                mask_rate=config.masking.mask_rate,
                mask_seed=config.masking.random_seed,
            )

            artifact_refs.append(
                self._create_artifact_ref(
                    "imputation_benchmark_report",
                    ArtifactCategory.EVALUATION,
                    benchmark_report,
                    output_dir,
                )
            )

            if config.execution.enable_checkpointing:
                self.checkpoint_manager.save_checkpoint(
                    stage=PipelineStage.IMPUTATION_EVALUATION,
                    experiment_id=config.experiment_id,
                    run_id=actual_run_id,
                    dataset_fingerprint=dataset_fp,
                    config_fingerprint=config_fp,
                    payload={"method_rankings": benchmark_report.method_rankings},
                )

            st_rec.duration_seconds = round(time.perf_counter() - t0_stage, 4)
            st_rec.end_time_utc = datetime.now(UTC).isoformat()
            sm.transition_to(StageStatus.COMPLETED)
            st_rec.status = StageStatus.COMPLETED

            # =========================================================================
            # STAGE 8: BIAS_ANALYSIS
            # =========================================================================
            current_stage = PipelineStage.BIAS_ANALYSIS
            st_rec = stage_records[current_stage]
            sm = StageStateMachine()
            sm.transition_to(StageStatus.RUNNING)
            st_rec.status = StageStatus.RUNNING
            st_rec.start_time_utc = datetime.now(UTC).isoformat()
            t0_stage = time.perf_counter()

            bias_engine = BiasAnalysisEngine(contract=self.contract, config=config.group_definition)
            bias_report = bias_engine.run_bias_analysis(
                df=clean_df,
                imputed_results=imputed_datasets,
                ground_truth_store=mask_res.ground_truth_store,
                experiment_id=config.experiment_id,
                dataset_version=config.dataset_version,
            )

            artifact_refs.append(
                self._create_artifact_ref(
                    "bias_analysis_report", ArtifactCategory.BIAS, bias_report, output_dir
                )
            )

            if config.execution.enable_checkpointing:
                self.checkpoint_manager.save_checkpoint(
                    stage=PipelineStage.BIAS_ANALYSIS,
                    experiment_id=config.experiment_id,
                    run_id=actual_run_id,
                    dataset_fingerprint=dataset_fp,
                    config_fingerprint=config_fp,
                    payload={"best_method_per_group": bias_report.best_method_per_group},
                )

            st_rec.duration_seconds = round(time.perf_counter() - t0_stage, 4)
            st_rec.end_time_utc = datetime.now(UTC).isoformat()
            sm.transition_to(StageStatus.COMPLETED)
            st_rec.status = StageStatus.COMPLETED

            # =========================================================================
            # STAGE 9: DOWNSTREAM_EVALUATION
            # =========================================================================
            current_stage = PipelineStage.DOWNSTREAM_EVALUATION
            st_rec = stage_records[current_stage]
            sm = StageStateMachine()
            sm.transition_to(StageStatus.RUNNING)
            st_rec.status = StageStatus.RUNNING
            st_rec.start_time_utc = datetime.now(UTC).isoformat()
            t0_stage = time.perf_counter()

            downstream_engine = DownstreamEvaluationEngine(
                contract=self.contract, config=config.downstream
            )
            downstream_report = downstream_engine.run_benchmark_suite(
                clean_df,
                mask_config=config.masking,
            )

            artifact_refs.append(
                self._create_artifact_ref(
                    "downstream_benchmark_report",
                    ArtifactCategory.DOWNSTREAM,
                    downstream_report,
                    output_dir,
                )
            )

            if config.execution.enable_checkpointing:
                self.checkpoint_manager.save_checkpoint(
                    stage=PipelineStage.DOWNSTREAM_EVALUATION,
                    experiment_id=config.experiment_id,
                    run_id=actual_run_id,
                    dataset_fingerprint=dataset_fp,
                    config_fingerprint=config_fp,
                    payload={"comparison_table": downstream_report.comparison_table},
                )

            st_rec.duration_seconds = round(time.perf_counter() - t0_stage, 4)
            st_rec.end_time_utc = datetime.now(UTC).isoformat()
            sm.transition_to(StageStatus.COMPLETED)
            st_rec.status = StageStatus.COMPLETED

            # =========================================================================
            # STAGE 10: ARTIFACT_VALIDATION
            # =========================================================================
            current_stage = PipelineStage.ARTIFACT_VALIDATION
            st_rec = stage_records[current_stage]
            sm = StageStateMachine()
            sm.transition_to(StageStatus.RUNNING)
            st_rec.status = StageStatus.RUNNING
            st_rec.start_time_utc = datetime.now(UTC).isoformat()
            t0_stage = time.perf_counter()

            for art in artifact_refs:
                art_path = Path(art.path)
                if not art_path.exists() or art_path.stat().st_size == 0:
                    raise PlatformError(
                        f"Artifact validation failed for '{art.name}': file is empty or missing."
                    )

            st_rec.duration_seconds = round(time.perf_counter() - t0_stage, 4)
            st_rec.end_time_utc = datetime.now(UTC).isoformat()
            sm.transition_to(StageStatus.COMPLETED)
            st_rec.status = StageStatus.COMPLETED

            # =========================================================================
            # STAGE 11: EXPERIMENT_FINALIZATION
            # =========================================================================
            current_stage = PipelineStage.EXPERIMENT_FINALIZATION
            st_rec = stage_records[current_stage]
            sm = StageStateMachine()
            sm.transition_to(StageStatus.RUNNING)
            st_rec.status = StageStatus.RUNNING
            st_rec.start_time_utc = datetime.now(UTC).isoformat()
            t0_stage = time.perf_counter()

            total_dur = round(time.perf_counter() - t0_total, 4)
            st_rec.duration_seconds = round(time.perf_counter() - t0_stage, 4)
            st_rec.end_time_utc = datetime.now(UTC).isoformat()
            sm.transition_to(StageStatus.COMPLETED)
            st_rec.status = StageStatus.COMPLETED

            final_status = StageStatus.COMPLETED

        except Exception as e:
            logger.error(
                "Experiment pipeline failed during execution",
                failed_stage=current_stage.value,
                error=str(e),
            )
            st_rec = stage_records[current_stage]
            st_rec.status = StageStatus.FAILED
            st_rec.error_code = ErrorCode.METHOD_FAILURE
            st_rec.error_message = str(e)
            st_rec.end_time_utc = datetime.now(UTC).isoformat()
            final_status = StageStatus.FAILED
            total_dur = round(time.perf_counter() - t0_total, 4)

        # Assemble Final Experiment Manifest
        manifest = ExperimentManifest(
            experiment_id=config.experiment_id,
            run_id=actual_run_id,
            dataset_version=config.dataset_version,
            dataset_fingerprint=dataset_fp,
            config_fingerprint=config_fp,
            code_version=env_info.get("git_commit", "unknown"),
            clean_worktree=env_info.get("clean_worktree", False),
            final_status=final_status,
            stage_statuses={s.value: rec.status.value for s, rec in stage_records.items()},
            stage_durations={s.value: rec.duration_seconds for s, rec in stage_records.items()},
            method_statuses={
                m: {
                    "status": s.status.value,
                    "duration_seconds": s.duration_seconds,
                    "error_code": s.error_code.value if s.error_code else None,
                    "error_message": s.error_message,
                }
                for m, s in method_statuses.items()
            },
            artifact_references=artifact_refs,
            seeds={
                "global_seed": config.random_seed,
                "mask_seed": config.masking.random_seed,
                "mitigation_seed": config.mitigation.random_seed,
                "downstream_seed": config.downstream.random_seed,
            },
            warnings=warnings_list,
            partial_failure=partial_failure,
            start_time_utc=start_time_utc,
            end_time_utc=datetime.now(UTC).isoformat(),
            total_duration_seconds=total_dur,
            config_snapshot=config_snapshot if "config_snapshot" in locals() else {},
        )

        manifest_path = output_dir / "manifest.json"
        manifest.save(manifest_path)
        logger.info(
            "Experiment pipeline finalized",
            experiment_id=config.experiment_id,
            run_id=actual_run_id,
            final_status=final_status.value,
            total_duration_seconds=total_dur,
            manifest_path=str(manifest_path),
        )

        return manifest
