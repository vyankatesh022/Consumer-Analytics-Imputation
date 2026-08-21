"""Unit tests for GroundTruthStore."""

import pandas as pd

from missing_data_platform.masking.ground_truth import GroundTruthStore


def test_ground_truth_store_operations() -> None:
    """Verify mask querying and ground truth retrieval from GroundTruthStore."""
    mask_df = pd.DataFrame(
        {
            "age": [False, True, False],
            "income": [True, False, False],
        }
    )
    orig_values = {
        "age": pd.Series([35.0], index=[1]),
        "income": pd.Series([75000.0], index=[0]),
    }

    store = GroundTruthStore(
        experiment_id="gt_test_01",
        mask_matrix=mask_df,
        original_values=orig_values,
    )

    assert store.total_masked_cells == 2
    assert store.feature_masked_counts == {"age": 1, "income": 1}
    assert store.is_artificially_masked("age", 1) is True
    assert store.is_artificially_masked("age", 0) is False
    assert store.is_artificially_masked("unknown_col", 0) is False

    # Retrieve ground truth series
    age_gt = store.get_ground_truth("age")
    assert len(age_gt) == 1
    assert age_gt.loc[1] == 35.0

    # Metadata dict
    meta = store.to_metadata_dict()
    assert meta["experiment_id"] == "gt_test_01"
    assert meta["total_masked_cells"] == 2
    assert "35.0" not in str(meta)  # Values are not leaked in metadata
