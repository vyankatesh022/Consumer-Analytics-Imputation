"""Centralized, validated configuration for end-to-end experiment orchestration."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from missing_data_platform.bias.config import GroupDefinitionConfig
from missing_data_platform.downstream.config import DownstreamConfig
from missing_data_platform.evaluation.config import EvaluationConfig
from missing_data_platform.exceptions import ConfigurationError
from missing_data_platform.masking.config import MaskingConfig
from missing_data_platform.mitigation.config import MitigationConfig


@dataclass
class ResourceLimitsConfig:
    """Resource protection constraints and execution quotas for experiment runs."""

    max_records: int = 1_000_000
    max_features: int = 100
    max_groups: int = 50
    max_methods: int = 10
    max_retries: int = 2
    stage_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        """Validate resource boundary invariants."""
        if self.max_records < 1:
            raise ConfigurationError("max_records must be >= 1.")
        if self.max_features < 1:
            raise ConfigurationError("max_features must be >= 1.")
        if self.max_groups < 1:
            raise ConfigurationError("max_groups must be >= 1.")
        if self.max_methods < 1:
            raise ConfigurationError("max_methods must be >= 1.")
        if self.max_retries < 0:
            raise ConfigurationError("max_retries must be >= 0.")
        if self.stage_timeout_seconds <= 0.0:
            raise ConfigurationError("stage_timeout_seconds must be > 0.")


@dataclass
class ExecutionConfig:
    """Operational settings controlling execution, checkpointing, and failure behavior."""

    fail_fast: bool = True
    allow_partial_imputation_failure: bool = True
    enable_checkpointing: bool = True
    checkpoint_dir: str = "./artifacts/checkpoints"
    output_dir: str = "./artifacts"
    resume_from_checkpoint: bool = False
    clean_worktree_required: bool = False

    def __post_init__(self) -> None:
        """Validate execution parameters."""
        if not self.checkpoint_dir.strip():
            raise ConfigurationError("checkpoint_dir cannot be empty.")
        if not self.output_dir.strip():
            raise ConfigurationError("output_dir cannot be empty.")


@dataclass
class ExperimentPipelineConfig:
    """Unified master configuration for reproducible, hardened ML pipeline execution."""

    experiment_id: str
    dataset_version: str = "1.0.0"
    random_seed: int = 42
    imputation_methods: list[str] = field(
        default_factory=lambda: [
            "baseline_median",
            "baseline_mean",
            "knn",
            "iterative",
            "random_forest",
        ]
    )
    masking: MaskingConfig = field(
        default_factory=lambda: MaskingConfig(experiment_id="orchestrator_mask_default")
    )
    mitigation: MitigationConfig = field(default_factory=MitigationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    group_definition: GroupDefinitionConfig = field(default_factory=GroupDefinitionConfig)
    downstream: DownstreamConfig = field(default_factory=DownstreamConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    resource_limits: ResourceLimitsConfig = field(default_factory=ResourceLimitsConfig)

    def __post_init__(self) -> None:
        """Validate configuration integrity across all sub-configurations."""
        if not self.experiment_id or not self.experiment_id.strip():
            raise ConfigurationError("experiment_id cannot be empty.")

        if not self.dataset_version or not self.dataset_version.strip():
            raise ConfigurationError("dataset_version cannot be empty.")

        if not self.imputation_methods:
            raise ConfigurationError("imputation_methods list cannot be empty.")

        if len(self.imputation_methods) > self.resource_limits.max_methods:
            raise ConfigurationError(
                f"Requested {len(self.imputation_methods)} methods, exceeding max_methods quota ({self.resource_limits.max_methods}).",
                context={"max_methods": self.resource_limits.max_methods},
            )

        # Propagate experiment_id and seeds to sub-configs if matching defaults
        if self.masking.experiment_id == "orchestrator_mask_default":
            self.masking.experiment_id = f"{self.experiment_id}_mask"
        self.masking.random_seed = self.random_seed
        self.mitigation.random_seed = self.random_seed
        self.downstream.random_seed = self.random_seed

    def to_sanitized_dict(self) -> dict[str, Any]:
        """Generate sanitized dictionary representation guaranteed free of credentials."""
        return {
            "experiment_id": self.experiment_id,
            "dataset_version": self.dataset_version,
            "random_seed": self.random_seed,
            "imputation_methods": list(self.imputation_methods),
            "masking": asdict(self.masking),
            "mitigation": asdict(self.mitigation),
            "evaluation": asdict(self.evaluation),
            "group_definition": asdict(self.group_definition),
            "downstream": asdict(self.downstream),
            "execution": asdict(self.execution),
            "resource_limits": asdict(self.resource_limits),
        }

    def get_config_fingerprint(self) -> str:
        """Compute deterministic SHA-256 hash of canonical JSON config representation."""
        sanitized = self.to_sanitized_dict()
        # Ephemeral runtime flags (resume, dirs) do not alter the mathematical experiment definition
        sanitized_for_fp = {k: v for k, v in sanitized.items() if k != "execution"}
        canonical_json = json.dumps(sanitized_for_fp, sort_keys=True, default=str)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
