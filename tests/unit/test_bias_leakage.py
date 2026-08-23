"""Mandatory Data Leakage Tests for Bias & Representation Analysis."""

import pandas as pd

from missing_data_platform.bias.config import GroupDefinitionConfig
from missing_data_platform.bias.engine import BiasAnalysisEngine
from missing_data_platform.imputation.engine import BaselineImputationEngine
from missing_data_platform.masking.config import MaskingConfig, MaskingStrategy
from missing_data_platform.masking.engine import MaskingEngine


def test_group_analysis_does_not_alter_imputation() -> None:
    """Verify that performing bias analysis cannot mutate imputer outputs or models."""
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:02d}" for i in range(20)],
            "customer_segment": ["Gold"] * 10 + ["Silver"] * 10,
            "income": [float(50000 + 1000 * i) for i in range(20)],
            "purchase_next_month": [1] * 20,
        }
    )

    mask_config = MaskingConfig(
        experiment_id="bias_leak_exp",
        mask_rate=0.20,
        random_seed=42,
        strategy=MaskingStrategy.UNIFORM_RANDOM,
        target_features=["income"],
    )

    masking_engine = MaskingEngine()
    mask_res = masking_engine.generate_benchmark_dataset(df, mask_config)

    # Impute
    imp_engine = BaselineImputationEngine()
    imp_res = imp_engine.impute_rf_dataset(mask_res.masked_dataset, random_seed=42)
    original_imputed_copy = imp_res.imputed_dataset.copy(deep=True)

    # Run Bias Analysis with different group definitions
    config_a = GroupDefinitionConfig(group_column="customer_segment")
    engine_a = BiasAnalysisEngine(config=config_a)
    _ = engine_a.run_bias_analysis(
        df=df,
        imputed_results={"rf": imp_res.imputed_dataset},
        ground_truth_store=mask_res.ground_truth_store,
    )

    # Verify imputed DataFrame is unchanged
    pd.testing.assert_frame_equal(imp_res.imputed_dataset, original_imputed_copy)
