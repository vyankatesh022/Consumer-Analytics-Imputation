"""Unit tests for group-level imputation performance, disparity metrics, and small-group suppression."""

import pandas as pd
import pytest

from missing_data_platform.bias.config import GroupDefinitionConfig
from missing_data_platform.bias.engine import BiasAnalysisEngine
from missing_data_platform.masking.ground_truth import GroundTruthStore


@pytest.fixture
def group_evaluation_setup() -> tuple[pd.DataFrame, GroundTruthStore, dict[str, pd.DataFrame]]:
    """Fixture with two large groups (GroupA, GroupB) and one small group (GroupC)."""
    # 10 rows GroupA, 10 rows GroupB, 2 rows GroupC
    segments = ["GroupA"] * 10 + ["GroupB"] * 10 + ["GroupC"] * 2
    true_income = [float(50000 + 1000 * i) for i in range(22)]

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:02d}" for i in range(22)],
            "customer_segment": segments,
            "income": true_income,
            "purchase_next_month": [1] * 22,
        }
    )

    # Mask 5 cells in GroupA, 5 cells in GroupB, 2 cells in GroupC
    masked_indices = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21]
    mask_df = pd.DataFrame(False, index=df.index, columns=["income"])
    mask_df.loc[masked_indices, "income"] = True

    gt_store = GroundTruthStore(
        experiment_id="perf_disp_exp",
        mask_matrix=mask_df,
        original_values={"income": df.loc[masked_indices, "income"]},
    )

    # Method 1 (KNN): Performs well on GroupA (err 100), worse on GroupB (err 2000)
    m_knn = df.copy(deep=True)
    for idx in [0, 1, 2, 3, 4]:
        m_knn.loc[idx, "income"] += 100.0
    for idx in [10, 11, 12, 13, 14]:
        m_knn.loc[idx, "income"] += 2000.0
    for idx in [20, 21]:
        m_knn.loc[idx, "income"] += 500.0

    # Method 2 (RF): Performs worse on GroupA (err 1500), better on GroupB (err 200)
    m_rf = df.copy(deep=True)
    for idx in [0, 1, 2, 3, 4]:
        m_rf.loc[idx, "income"] += 1500.0
    for idx in [10, 11, 12, 13, 14]:
        m_rf.loc[idx, "income"] += 200.0
    for idx in [20, 21]:
        m_rf.loc[idx, "income"] += 300.0

    return df, gt_store, {"knn": m_knn, "rf": m_rf}


def test_group_performance_and_suppression(group_evaluation_setup) -> None:
    """Verify group-level metrics are computed for large groups and suppressed for small groups."""
    df, gt_store, methods = group_evaluation_setup
    config = GroupDefinitionConfig(group_column="customer_segment", minimum_group_size=5)
    engine = BiasAnalysisEngine(config=config)

    group_series = engine.extract_group_series(df)
    perf_results = engine.analyze_imputation_by_group(
        imputed_results=methods,
        ground_truth_store=gt_store,
        group_series=group_series,
    )

    # Check GroupA performance under KNN (MAE = 100)
    knn_ga_mae = next(
        p
        for p in perf_results
        if p.group_value == "GroupA" and p.method == "knn" and p.metric_name == "MAE"
    )
    assert knn_ga_mae.metric_value == 100.0
    assert knn_ga_mae.is_suppressed is False
    assert knn_ga_mae.sample_count == 5

    # Check GroupC performance (2 samples < 5 threshold -> suppressed)
    knn_gc_mae = next(
        p
        for p in perf_results
        if p.group_value == "GroupC" and p.method == "knn" and p.metric_name == "MAE"
    )
    assert knn_gc_mae.is_suppressed is True
    assert knn_gc_mae.metric_value is None
    assert knn_gc_mae.warning == "INSUFFICIENT_SAMPLE_SIZE"


def test_disparity_calculation(group_evaluation_setup) -> None:
    """Verify pairwise disparity calculation between groups."""
    df, gt_store, methods = group_evaluation_setup
    config = GroupDefinitionConfig(group_column="customer_segment", minimum_group_size=5)
    engine = BiasAnalysisEngine(config=config)

    group_series = engine.extract_group_series(df)
    perf_results = engine.analyze_imputation_by_group(methods, gt_store, group_series)
    disparities = engine.calculate_disparities(perf_results)

    # In KNN: GroupA MAE = 100, GroupB MAE = 2000 -> absolute disparity = 1900
    knn_disp = next(d for d in disparities if d.method == "knn" and d.metric_name == "MAE")
    assert knn_disp.absolute_disparity == 1900.0
    assert knn_disp.sample_count_a == 5
    assert knn_disp.sample_count_b == 5


def test_best_method_per_group_differs(group_evaluation_setup) -> None:
    """Verify analysis identifies when different methods perform best on different groups."""
    df, gt_store, methods = group_evaluation_setup
    config = GroupDefinitionConfig(group_column="customer_segment", minimum_group_size=5)
    engine = BiasAnalysisEngine(config=config)

    result = engine.run_bias_analysis(
        df=df,
        imputed_results=methods,
        ground_truth_store=gt_store,
        experiment_id="bias_multi_method",
    )

    # For GroupA, KNN has lower error (100 vs 1500)
    assert result.best_method_per_group["GroupA"] == "knn"
    # For GroupB, RF has lower error (200 vs 2000)
    assert result.best_method_per_group["GroupB"] == "rf"
