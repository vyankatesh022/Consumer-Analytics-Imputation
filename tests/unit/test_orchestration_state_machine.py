"""Unit tests for StageStateMachine transition guards and lifecycle validation."""

import pytest

from missing_data_platform.exceptions import PlatformError
from missing_data_platform.orchestration.stages import (
    ErrorCode,
    MethodExecutionStatus,
    StageStateMachine,
    StageStatus,
)


def test_state_machine_valid_forward_transitions() -> None:
    """Assert valid lifecycle state progression: PENDING -> RUNNING -> COMPLETED."""
    sm = StageStateMachine(StageStatus.PENDING)
    assert sm.current_status == StageStatus.PENDING

    sm.transition_to(StageStatus.RUNNING)
    assert sm.current_status == StageStatus.RUNNING

    sm.transition_to(StageStatus.COMPLETED)
    assert sm.current_status == StageStatus.COMPLETED


def test_state_machine_valid_failure_transition() -> None:
    """Assert valid failure transition: PENDING -> RUNNING -> FAILED."""
    sm = StageStateMachine(StageStatus.PENDING)
    sm.transition_to(StageStatus.RUNNING)
    sm.transition_to(StageStatus.FAILED)
    assert sm.current_status == StageStatus.FAILED


def test_state_machine_valid_skip_transition() -> None:
    """Assert valid skip transition: PENDING -> SKIPPED."""
    sm = StageStateMachine(StageStatus.PENDING)
    sm.transition_to(StageStatus.SKIPPED)
    assert sm.current_status == StageStatus.SKIPPED


def test_state_machine_invalid_backward_transitions() -> None:
    """Assert PlatformError on illegal transitions such as COMPLETED -> RUNNING."""
    sm = StageStateMachine(StageStatus.PENDING)
    sm.transition_to(StageStatus.RUNNING)
    sm.transition_to(StageStatus.COMPLETED)

    with pytest.raises(PlatformError, match="Illegal stage state transition"):
        sm.transition_to(StageStatus.RUNNING)

    with pytest.raises(PlatformError, match="Illegal stage state transition"):
        sm.transition_to(StageStatus.PENDING)


def test_method_execution_status_tracking() -> None:
    """Assert MethodExecutionStatus records completion and failure accurately."""
    status = MethodExecutionStatus(method="knn", status=StageStatus.RUNNING)
    status.mark_completed(duration=1.2345, artifact_ref="knn_model_ref")

    assert status.status == StageStatus.COMPLETED
    assert status.duration_seconds == 1.2345
    assert status.artifact_ref == "knn_model_ref"

    err_status = MethodExecutionStatus(method="iterative", status=StageStatus.RUNNING)
    err_status.mark_failed(
        duration=0.5,
        error_code=ErrorCode.METHOD_FAILURE,
        error_message="Convergence failed",
    )

    assert err_status.status == StageStatus.FAILED
    assert err_status.error_code == ErrorCode.METHOD_FAILURE
    assert "Convergence failed" in (err_status.error_message or "")
