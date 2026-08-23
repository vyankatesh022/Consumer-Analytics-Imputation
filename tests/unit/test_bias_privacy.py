"""Privacy and Small-Group Protection tests for Bias & Representation Analysis."""

import pandas as pd

from missing_data_platform.bias.config import GroupDefinitionConfig
from missing_data_platform.bias.engine import BiasAnalysisEngine
from missing_data_platform.masking.ground_truth import GroundTruthStore


def test_bias_analysis_output_preserves_privacy() -> None:
    """Verify BiasAnalysisResult and JSON exports contain only aggregated metrics without raw customer data."""
    df = pd.DataFrame(
        {
            "customer_id": [f"PII_CUSTOMER_{i:03d}" for i in range(15)],
            "customer_segment": ["Gold"] * 10 + ["Bronze"] * 5,
            "income": [float(100000 + 5000 * i) for i in range(15)],
            "purchase_next_month": [1] * 15,
        }
    )

    mask_df = pd.DataFrame(False, index=df.index, columns=["income"])
    mask_df.loc[[0, 1, 2, 10, 11], "income"] = True
    gt_store = GroundTruthStore(
        "priv_exp", mask_df, {"income": df.loc[[0, 1, 2, 10, 11], "income"]}
    )

    config = GroupDefinitionConfig(group_column="customer_segment", minimum_group_size=5)
    engine = BiasAnalysisEngine(config=config)

    result = engine.run_bias_analysis(
        df=df,
        imputed_results={"baseline": df},
        ground_truth_store=gt_store,
        experiment_id="priv_test_exp",
    )

    json_str = result.to_json()

    # Ensure no customer IDs or row-level raw values exist in metadata
    assert "PII_CUSTOMER" not in json_str
    assert "customer_id" not in json_str

    # Operational aggregate information must be present
    assert "priv_test_exp" in json_str
    assert "representation_results" in json_str
    assert "disparity_results" in json_str
