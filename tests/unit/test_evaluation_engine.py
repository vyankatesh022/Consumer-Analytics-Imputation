"""Unit tests for ImputationEvaluator engine, multi-method comparison, and ranking logic."""

import numpy as np
import pandas as pd
import pytest

from missing_data_platform.evaluation.engine import ImputationEvaluator
from missing_data_platform.exceptions import DataQualityError, EvaluationError
from missing_data_platform.masking.ground_truth import GroundTruthStore


@pytest.fixture
def synthetic_eval_data() -> tuple[pd.DataFrame, GroundTruthStore, dict[str, pd.DataFrame]]:
    """Fixture providing masked ground truth store and distinct method predictions."""
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4"],
            "income": [50000.0, 60000.0, 70000.0, 80000.0],
            "total_spend": [500.0, 600.0, 700.0, 800.0],
            "purchase_next_month": [0, 1, 0, 1],
        }
    )

    # Artificially mask row 1 and row 2 of income and total_spend
    mask_df = pd.DataFrame(
        {
            "income": [False, True, True, False],
            "total_spend": [False, True, False, True],
        },
        index=df.index,
    )

    gt_store = GroundTruthStore(
        experiment_id="test_exp",
        mask_matrix=mask_df,
        original_values={
            "income": df.loc[mask_df["income"], "income"],
            "total_spend": df.loc[mask_df["total_spend"], "total_spend"],
        },
    )

    # Method 1: Perfect predictions
    m1 = df.copy(deep=True)

    # Method 2: Offset predictions
    m2 = df.copy(deep=True)
    m2.loc[1, "income"] = 65000.0  # +5000
    m2.loc[2, "income"] = 75000.0  # +5000
    m2.loc[1, "total_spend"] = 650.0  # +50
    m2.loc[3, "total_spend"] = 850.0  # +50

    # Method 3: Incomplete predictions with NaN
    m3 = df.copy(deep=True)
    m3.loc[1, "income"] = np.nan
    m3.loc[2, "income"] = 70000.0

    return df, gt_store, {"perfect": m1, "offset": m2, "incomplete": m3}


def test_evaluate_method_perfect_score(synthetic_eval_data) -> None:
    """Verify evaluator produces MAE=0.0 and RMSE=0.0 on perfect predictions."""
    _, gt_store, methods = synthetic_eval_data
    evaluator = ImputationEvaluator()
    res = evaluator.evaluate_method(
        imputed_df=methods["perfect"],
        ground_truth_store=gt_store,
        method_name="perfect",
    )

    assert res.total_evaluated_cells == 4
    assert res.missing_prediction_count == 0
    assert res.weighted_mae == 0.0
    assert res.weighted_rmse == 0.0
    assert len(res.feature_results) == 2


def test_evaluate_method_offset_metrics(synthetic_eval_data) -> None:
    """Verify evaluator calculates correct MAE and RMSE on offset predictions."""
    _, gt_store, methods = synthetic_eval_data
    evaluator = ImputationEvaluator()
    res = evaluator.evaluate_method(
        imputed_df=methods["offset"],
        ground_truth_store=gt_store,
        method_name="offset",
    )

    assert res.total_evaluated_cells == 4
    income_res = next(f for f in res.feature_results if f.feature_name == "income")
    assert income_res.mae == 5000.0
    assert income_res.rmse == 5000.0

    spend_res = next(f for f in res.feature_results if f.feature_name == "total_spend")
    assert spend_res.mae == 50.0
    assert spend_res.rmse == 50.0


def test_evaluate_method_tracks_missing_predictions(synthetic_eval_data) -> None:
    """Verify evaluator tracks missing predictions without silently discarding."""
    _, gt_store, methods = synthetic_eval_data
    evaluator = ImputationEvaluator()
    res = evaluator.evaluate_method(
        imputed_df=methods["incomplete"],
        ground_truth_store=gt_store,
        method_name="incomplete",
    )

    assert res.missing_prediction_count == 1
    income_res = next(f for f in res.feature_results if f.feature_name == "income")
    assert income_res.missing_prediction_count == 1
    assert income_res.mae == 0.0  # Only valid cell was exact


def test_compare_methods_ranks_correctly(synthetic_eval_data) -> None:
    """Verify compare_methods assigns rank 1 to the best performing method."""
    _, gt_store, methods = synthetic_eval_data
    evaluator = ImputationEvaluator()
    report = evaluator.compare_methods(
        imputed_results=methods,
        ground_truth_store=gt_store,
        experiment_id="multi_comp",
    )

    assert len(report.method_rankings) == 3
    # Perfect method should be ranked #1
    assert report.method_rankings[0]["method"] == "perfect"
    assert report.method_rankings[0]["rank_mae"] == 1

    # Check DataFrame export
    df_summary = report.to_summary_dataframe()
    assert len(df_summary) == 3
    assert "Method" in df_summary.columns
    assert "Weighted MAE" in df_summary.columns

    # JSON export
    json_meta = report.to_json()
    assert "multi_comp" in json_meta
    assert "method_rankings" in json_meta


def test_evaluator_empty_df_raises_error(synthetic_eval_data) -> None:
    """Verify empty DataFrame raises DataQualityError."""
    _, gt_store, _ = synthetic_eval_data
    evaluator = ImputationEvaluator()
    with pytest.raises(DataQualityError):
        evaluator.evaluate_method(pd.DataFrame(), gt_store, "test")


def test_evaluator_zero_masked_cells_raises_error() -> None:
    """Verify 0 masked cells in GroundTruthStore raises EvaluationError."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    mask = pd.DataFrame({"a": [False, False]})
    gt_store = GroundTruthStore("empty_mask", mask, {})
    evaluator = ImputationEvaluator()
    with pytest.raises(EvaluationError):
        evaluator.evaluate_method(df, gt_store, "test")
