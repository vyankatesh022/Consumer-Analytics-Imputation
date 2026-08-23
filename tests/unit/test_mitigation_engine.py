"""Unit tests for FairnessMitigationEngine execution, decision logic, and constraints."""

import numpy as np
import pandas as pd
import pytest

from missing_data_platform.masking.config import MaskingConfig, MaskingStrategy
from missing_data_platform.mitigation.config import (
    MitigationConfig,
    MitigationDecision,
    MitigationStrategy,
)
from missing_data_platform.mitigation.engine import FairnessMitigationEngine


@pytest.fixture
def imbalanced_group_df() -> pd.DataFrame:
    """Fixture with imbalanced segments and correlated numeric features."""
    np.random.seed(42)
    # 70% Gold, 30% Bronze
    segments = ["Gold"] * 35 + ["Bronze"] * 15
    x = [float(i) for i in range(50)]
    # In Gold, y = 2 * x; In Bronze, y = 4 * x
    y = [float(2 * xi if seg == "Gold" else 4 * xi) for xi, seg in zip(x, segments, strict=True)]

    return pd.DataFrame(
        {
            "customer_id": [f"C{i:02d}" for i in range(50)],
            "customer_segment": segments,
            "feat_x": x,
            "feat_y": y,
            "purchase_next_month": [0] * 50,
        }
    )


def test_disabled_mitigation_returns_baseline(imbalanced_group_df: pd.DataFrame) -> None:
    """Verify disabled mitigation matches baseline exactly with decision=ACCEPTED."""
    mask_config = MaskingConfig(
        experiment_id="mit_disabled_test",
        mask_rate=0.20,
        random_seed=42,
        strategy=MaskingStrategy.UNIFORM_RANDOM,
        target_features=["feat_y"],
    )

    config = MitigationConfig(enabled=False)
    engine = FairnessMitigationEngine(config=config)

    result = engine.mitigate_and_evaluate(
        df=imbalanced_group_df,
        mask_config=mask_config,
        method="random_forest",
    )

    assert result.decision == MitigationDecision.ACCEPTED
    assert result.accuracy_change_pct == 0.0
    assert result.disparity_reduction_pct == 0.0
    assert result.mitigated_mae == result.baseline_mae


def test_sample_weighting_mitigation(imbalanced_group_df: pd.DataFrame) -> None:
    """Verify sample weighting runs and produces complete before/after audit."""
    mask_config = MaskingConfig(
        experiment_id="mit_weighting_test",
        mask_rate=0.20,
        random_seed=42,
        strategy=MaskingStrategy.UNIFORM_RANDOM,
        target_features=["feat_y"],
    )

    config = MitigationConfig(
        enabled=True,
        strategy=MitigationStrategy.SAMPLE_WEIGHTING,
        group_column="customer_segment",
        max_allowed_accuracy_degradation=0.50,
        target_disparity_reduction=0.05,
    )
    engine = FairnessMitigationEngine(config=config)

    result = engine.mitigate_and_evaluate(
        df=imbalanced_group_df,
        mask_config=mask_config,
        method="random_forest",
    )

    assert result.mitigated_mae is not None
    assert result.baseline_mae is not None
    assert result.decision in (
        MitigationDecision.ACCEPTED,
        MitigationDecision.REQUIRES_REVIEW,
        MitigationDecision.REJECTED,
    )
    assert len(result.decision_reason) > 0


def test_group_specific_models_mitigation(imbalanced_group_df: pd.DataFrame) -> None:
    """Verify group-specific models strategy executes and evaluates correctly."""
    mask_config = MaskingConfig(
        experiment_id="mit_group_spec_test",
        mask_rate=0.20,
        random_seed=42,
        strategy=MaskingStrategy.UNIFORM_RANDOM,
        target_features=["feat_y"],
    )

    config = MitigationConfig(
        enabled=True,
        strategy=MitigationStrategy.GROUP_SPECIFIC,
        group_column="customer_segment",
        minimum_group_size=5,
    )
    engine = FairnessMitigationEngine(config=config)

    result = engine.mitigate_and_evaluate(
        df=imbalanced_group_df,
        mask_config=mask_config,
        method="random_forest",
    )

    assert result.decision in (
        MitigationDecision.ACCEPTED,
        MitigationDecision.REQUIRES_REVIEW,
        MitigationDecision.REJECTED,
    )
    assert result.mitigated_mae is not None
