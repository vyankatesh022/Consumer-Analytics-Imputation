"""Pipeline lifecycle stages, status enumerations, error classifications, and state machines."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from missing_data_platform.exceptions import PlatformError


class PipelineStage(StrEnum):
    """Canonical sequential stages of the production ML experiment pipeline."""

    ENVIRONMENT_VALIDATION = "environment_validation"
    DATASET_VALIDATION = "dataset_validation"
    EXPERIMENT_INITIALIZATION = "experiment_initialization"
    MASKING = "masking"
    IMPUTATION = "imputation"
    MITIGATION = "mitigation"
    IMPUTATION_EVALUATION = "imputation_evaluation"
    BIAS_ANALYSIS = "bias_analysis"
    DOWNSTREAM_EVALUATION = "downstream_evaluation"
    ARTIFACT_VALIDATION = "artifact_validation"
    EXPERIMENT_FINALIZATION = "experiment_finalization"


class StageStatus(StrEnum):
    """Operational status of a pipeline stage or algorithm execution."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ErrorCode(StrEnum):
    """Structured, sanitized failure category identifiers."""

    CONFIG_INVALID = "CONFIG_INVALID"
    ENVIRONMENT_INCOMPATIBLE = "ENVIRONMENT_INCOMPATIBLE"
    DATASET_INVALID = "DATASET_INVALID"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    CHECKPOINT_INVALID = "CHECKPOINT_INVALID"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    MASKING_FAILURE = "MASKING_FAILURE"
    METHOD_FAILURE = "METHOD_FAILURE"
    MITIGATION_FAILURE = "MITIGATION_FAILURE"
    EVALUATION_FAILURE = "EVALUATION_FAILURE"
    BIAS_ANALYSIS_FAILURE = "BIAS_ANALYSIS_FAILURE"
    DOWNSTREAM_FAILURE = "DOWNSTREAM_FAILURE"
    ARTIFACT_FAILURE = "ARTIFACT_FAILURE"


@dataclass
class MethodExecutionStatus:
    """Execution telemetry and outcome for a specific algorithm within a stage."""

    method: str
    status: StageStatus
    start_time_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    end_time_utc: str | None = None
    duration_seconds: float = 0.0
    error_code: ErrorCode | None = None
    error_message: str | None = None
    artifact_ref: str | None = None

    def mark_completed(self, duration: float, artifact_ref: str | None = None) -> None:
        """Mark method execution as successfully completed."""
        self.status = StageStatus.COMPLETED
        self.end_time_utc = datetime.now(UTC).isoformat()
        self.duration_seconds = round(duration, 4)
        self.artifact_ref = artifact_ref

    def mark_failed(self, duration: float, error_code: ErrorCode, error_message: str) -> None:
        """Mark method execution as failed with sanitized error details."""
        self.status = StageStatus.FAILED
        self.end_time_utc = datetime.now(UTC).isoformat()
        self.duration_seconds = round(duration, 4)
        self.error_code = error_code
        self.error_message = error_message


class StageStateMachine:
    """Enforces valid lifecycle state transitions and prevents illegal stage states."""

    VALID_TRANSITIONS: dict[StageStatus, set[StageStatus]] = {
        StageStatus.PENDING: {StageStatus.RUNNING, StageStatus.SKIPPED},
        StageStatus.RUNNING: {StageStatus.COMPLETED, StageStatus.FAILED},
        StageStatus.COMPLETED: set(),  # Terminal state for stage run
        StageStatus.FAILED: set(),  # Terminal state
        StageStatus.SKIPPED: set(),  # Terminal state
    }

    def __init__(self, initial_status: StageStatus = StageStatus.PENDING) -> None:
        self._current_status = initial_status

    @property
    def current_status(self) -> StageStatus:
        """Retrieve current lifecycle status."""
        return self._current_status

    def transition_to(self, new_status: StageStatus) -> None:
        """Transition to a new lifecycle state if legally permitted."""
        allowed = self.VALID_TRANSITIONS.get(self._current_status, set())
        if new_status not in allowed:
            raise PlatformError(
                f"Illegal stage state transition: {self._current_status.value} -> {new_status.value}",
                context={
                    "current_status": self._current_status.value,
                    "target_status": new_status.value,
                    "allowed_transitions": [s.value for s in allowed],
                },
            )
        self._current_status = new_status


@dataclass
class StageRecord:
    """Execution record for a single pipeline stage."""

    stage: PipelineStage
    status: StageStatus = StageStatus.PENDING
    start_time_utc: str | None = None
    end_time_utc: str | None = None
    duration_seconds: float = 0.0
    error_code: ErrorCode | None = None
    error_message: str | None = None
    checkpoint_ref: str | None = None
    method_statuses: dict[str, MethodExecutionStatus] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
