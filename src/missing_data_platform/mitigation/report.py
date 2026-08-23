"""Reporting schemas and before/after audit models for bias mitigation experiments."""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from missing_data_platform.bias.report import GroupImputationPerformance
from missing_data_platform.mitigation.config import MitigationDecision, MitigationStrategy


@dataclass
class MitigationResult:
    """Comprehensive container for before vs after mitigation evaluation and acceptance decision."""

    experiment_id: str
    dataset_version: str
    method: str
    mitigation_strategy: MitigationStrategy
    mitigation_config: dict[str, Any]
    baseline_mae: float | None
    baseline_rmse: float | None
    baseline_max_disparity: float | None
    mitigated_mae: float | None
    mitigated_rmse: float | None
    mitigated_max_disparity: float | None
    accuracy_change_pct: float | None
    disparity_reduction_pct: float | None
    decision: MitigationDecision
    decision_reason: str
    group_results_before: list[GroupImputationPerformance]
    group_results_after: list[GroupImputationPerformance]
    warnings: list[str] = field(default_factory=list)
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_metadata_dict(self) -> dict[str, Any]:
        """Export sanitized mitigation metadata without customer records."""
        return {
            "experiment_id": self.experiment_id,
            "dataset_version": self.dataset_version,
            "method": self.method,
            "mitigation_strategy": self.mitigation_strategy.value,
            "mitigation_config": self.mitigation_config,
            "baseline_mae": self.baseline_mae,
            "baseline_rmse": self.baseline_rmse,
            "baseline_max_disparity": self.baseline_max_disparity,
            "mitigated_mae": self.mitigated_mae,
            "mitigated_rmse": self.mitigated_rmse,
            "mitigated_max_disparity": self.mitigated_max_disparity,
            "accuracy_change_pct": self.accuracy_change_pct,
            "disparity_reduction_pct": self.disparity_reduction_pct,
            "decision": self.decision.value,
            "decision_reason": self.decision_reason,
            "group_results_before": [asdict(g) for g in self.group_results_before],
            "group_results_after": [asdict(g) for g in self.group_results_after],
            "warnings": self.warnings,
            "created_at_utc": self.created_at_utc,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize mitigation report to JSON string."""
        return json.dumps(self.to_metadata_dict(), indent=indent)
