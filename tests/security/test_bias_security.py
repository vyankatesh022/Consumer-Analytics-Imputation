"""Security tests for Bias & Representation Analysis: log sanitization and operational privacy."""

import io
import logging

import pandas as pd

from missing_data_platform.bias.config import GroupDefinitionConfig
from missing_data_platform.bias.engine import BiasAnalysisEngine
from missing_data_platform.config import LogLevel, Settings
from missing_data_platform.logging import configure_logging
from missing_data_platform.masking.ground_truth import GroundTruthStore


def test_bias_logging_does_not_leak_customer_records() -> None:
    """Verify bias analysis logs never emit individual customer records, sensitive values, or PII."""
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)

    settings = Settings(LOG_LEVEL=LogLevel.INFO, LOG_FORMAT="json")
    configure_logging(settings)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    df = pd.DataFrame(
        {
            "customer_id": ["CONFIDENTIAL_CUSTOMER_XYZ_999"],
            "customer_segment": ["Gold"],
            "income": [750000.0],
            "purchase_next_month": [1],
        }
    )

    mask_df = pd.DataFrame(True, index=df.index, columns=["income"])
    gt_store = GroundTruthStore("bias_sec_exp", mask_df, {"income": df["income"]})

    config = GroupDefinitionConfig(group_column="customer_segment")
    engine = BiasAnalysisEngine(config=config)

    result = engine.run_bias_analysis(
        df=df,
        imputed_results={"test": df},
        ground_truth_store=gt_store,
        experiment_id="exp_bias_sec_audit",
    )

    assert result.experiment_id == "exp_bias_sec_audit"

    handler.flush()
    logs = log_capture.getvalue()

    # Verify sensitive customer values and IDs are absent from logs
    assert "CONFIDENTIAL_CUSTOMER" not in logs
    assert "750000" not in logs

    # Ensure operational tracking is present
    assert "exp_bias_sec_audit" in logs
    assert "customer_segment" in logs

    root_logger.removeHandler(handler)
