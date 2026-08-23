"""Privacy tests for Bias Mitigation Layer."""

import pandas as pd

from missing_data_platform.masking.config import MaskingConfig, MaskingStrategy
from missing_data_platform.mitigation.config import (
    MitigationConfig,
    MitigationStrategy,
)
from missing_data_platform.mitigation.engine import FairnessMitigationEngine


def test_mitigation_results_preserve_privacy() -> None:
    """Verify MitigationResult and JSON exports contain only aggregated metrics without raw customer data."""
    df = pd.DataFrame(
        {
            "customer_id": [f"PRIV_CUSTOMER_{i:03d}" for i in range(20)],
            "customer_segment": ["Gold"] * 10 + ["Silver"] * 10,
            "income": [float(100000 + 5000 * i) for i in range(20)],
            "purchase_next_month": [1] * 20,
        }
    )

    mask_config = MaskingConfig(
        experiment_id="priv_mit_exp",
        mask_rate=0.20,
        random_seed=42,
        strategy=MaskingStrategy.UNIFORM_RANDOM,
        target_features=["income"],
    )

    config = MitigationConfig(
        enabled=True,
        strategy=MitigationStrategy.SAMPLE_WEIGHTING,
        group_column="customer_segment",
    )
    engine = FairnessMitigationEngine(config=config)

    result = engine.mitigate_and_evaluate(df, mask_config, method="random_forest")
    json_meta = result.to_json()

    # Ensure no customer IDs or row-level raw values exist in metadata
    assert "PRIV_CUSTOMER" not in json_meta
    assert "customer_id" not in json_meta

    # Operational aggregate information must be present
    assert "priv_mit_exp" in json_meta
    assert "decision" in json_meta
