"""Security tests for Bias Mitigation: log sanitization and operational privacy."""

import io
import logging

import pandas as pd

from missing_data_platform.config import LogLevel, Settings
from missing_data_platform.logging import configure_logging
from missing_data_platform.masking.config import MaskingConfig, MaskingStrategy
from missing_data_platform.mitigation.config import (
    MitigationConfig,
    MitigationStrategy,
)
from missing_data_platform.mitigation.engine import FairnessMitigationEngine


def test_mitigation_logging_does_not_leak_customer_records() -> None:
    """Verify mitigation logs never emit individual customer records, sensitive values, or PII."""
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)

    settings = Settings(LOG_LEVEL=LogLevel.INFO, LOG_FORMAT="json")
    configure_logging(settings)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    df = pd.DataFrame(
        {
            "customer_id": [f"TOP_SECRET_CUSTOMER_{i:03d}" for i in range(10)],
            "customer_segment": ["Gold"] * 10,
            "income": [600000.0] * 10,
            "purchase_next_month": [1] * 10,
        }
    )

    mask_config = MaskingConfig(
        experiment_id="mit_sec_exp",
        mask_rate=0.2,
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
    assert result.experiment_id == "mit_sec_exp"

    handler.flush()
    logs = log_capture.getvalue()

    # Verify sensitive customer values and IDs are absent from logs
    assert "TOP_SECRET_CUSTOMER" not in logs
    assert "600000" not in logs

    # Ensure operational tracking is present
    assert "mit_sec_exp" in logs
    assert "sample_weighting" in logs

    root_logger.removeHandler(handler)
