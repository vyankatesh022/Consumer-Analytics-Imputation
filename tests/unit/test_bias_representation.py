"""Unit tests for demographic/segment group representation and missingness analysis."""

import numpy as np
import pandas as pd
import pytest

from missing_data_platform.bias.config import GroupDefinitionConfig, MissingGroupPolicy
from missing_data_platform.bias.engine import BiasAnalysisEngine
from missing_data_platform.masking.ground_truth import GroundTruthStore


@pytest.fixture
def demographic_consumer_df() -> pd.DataFrame:
    """Fixture with distinct customer segments, natural missingness, and age cohorts."""
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:02d}" for i in range(20)],
            "customer_segment": ["Gold"] * 10 + ["Silver"] * 7 + ["Bronze"] * 3,
            "age": [25.0] * 5 + [45.0] * 10 + [np.nan] * 5,
            "income": [100000.0] * 5 + [np.nan] * 5 + [50000.0] * 7 + [25000.0] * 3,
            "total_spend": [1500.0] * 20,
            "purchase_next_month": [1] * 20,
        }
    )


def test_group_representation_calculation(demographic_consumer_df: pd.DataFrame) -> None:
    """Verify group representation population counts and percentages."""
    config = GroupDefinitionConfig(group_column="customer_segment", minimum_group_size=5)
    engine = BiasAnalysisEngine(config=config)

    rep_results = engine.analyze_representation(demographic_consumer_df)
    assert len(rep_results) == 3

    gold_rep = next(r for r in rep_results if r.group_value == "Gold")
    assert gold_rep.population_count == 10
    assert gold_rep.population_percentage == 50.0
    assert gold_rep.is_small_group is False

    bronze_rep = next(r for r in rep_results if r.group_value == "Bronze")
    assert bronze_rep.population_count == 3
    assert bronze_rep.population_percentage == 15.0
    assert bronze_rep.is_small_group is True  # 3 < 5


def test_group_representation_with_evaluation_cells(
    demographic_consumer_df: pd.DataFrame,
) -> None:
    """Verify evaluation cell count distribution across demographic groups."""
    # Mask 4 cells in income: rows 0, 1 (Gold), and rows 10, 11 (Silver)
    mask_df = pd.DataFrame(False, index=demographic_consumer_df.index, columns=["income"])
    mask_df.loc[[0, 1, 10, 11], "income"] = True

    gt_store = GroundTruthStore(
        experiment_id="eval_rep_exp",
        mask_matrix=mask_df,
        original_values={"income": demographic_consumer_df.loc[[0, 1, 10, 11], "income"]},
    )

    config = GroupDefinitionConfig(group_column="customer_segment", minimum_group_size=5)
    engine = BiasAnalysisEngine(config=config)

    rep_results = engine.analyze_representation(
        demographic_consumer_df, ground_truth_store=gt_store
    )
    gold_rep = next(r for r in rep_results if r.group_value == "Gold")
    silver_rep = next(r for r in rep_results if r.group_value == "Silver")

    assert gold_rep.eligible_evaluation_cells == 2
    assert gold_rep.evaluation_percentage == 50.0
    assert silver_rep.eligible_evaluation_cells == 2
    assert silver_rep.evaluation_percentage == 50.0


def test_group_missingness_disparity(demographic_consumer_df: pd.DataFrame) -> None:
    """Verify natural missingness rates differ across groups."""
    config = GroupDefinitionConfig(group_column="customer_segment")
    engine = BiasAnalysisEngine(config=config)

    miss_results = engine.analyze_group_missingness(
        demographic_consumer_df, target_features=["income"]
    )
    gold_miss = next(m for m in miss_results if m.group_value == "Gold")
    silver_miss = next(m for m in miss_results if m.group_value == "Silver")

    # In Gold: 5 missing out of 10 -> 50%
    assert gold_miss.missing_count == 5
    assert gold_miss.missing_rate == 50.0

    # In Silver: 0 missing out of 7 -> 0%
    assert silver_miss.missing_count == 0
    assert silver_miss.missing_rate == 0.0


def test_age_band_group_extraction(demographic_consumer_df: pd.DataFrame) -> None:
    """Verify continuous age attribute is discretized into standard cohorts."""
    config = GroupDefinitionConfig(
        group_column="age", missing_group_policy=MissingGroupPolicy.UNKNOWN
    )
    engine = BiasAnalysisEngine(config=config)

    series = engine.extract_group_series(demographic_consumer_df)
    assert "25-34" in series.values
    assert "45-54" in series.values
    assert "Unknown" in series.values
