"""Security tests for Artificial Missingness Masking: ground truth isolation and log sanitization."""

import io
import logging

import pandas as pd

from missing_data_platform.config import LogLevel, Settings
from missing_data_platform.logging import configure_logging
from missing_data_platform.masking.config import MaskingConfig
from missing_data_platform.masking.engine import MaskingEngine


def test_masking_logging_does_not_leak_ground_truth_values() -> None:
    """Verify that masking engine structured logs never emit ground-truth cell values or customer PII."""
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)

    settings = Settings(LOG_LEVEL=LogLevel.INFO, LOG_FORMAT="json")
    configure_logging(settings)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    df = pd.DataFrame(
        {
            "customer_id": ["PRIVATE_CUSTOMER_7777"],
            "income": [999999.0],
            "purchase_next_month": [1],
        }
    )

    config = MaskingConfig(
        experiment_id="exp_security_isolation_test",
        mask_rate=1.0,
        random_seed=42,
        target_features=["income"],
    )

    engine = MaskingEngine()
    result = engine.generate_benchmark_dataset(df, config)

    assert result.total_artificially_masked_cells == 1

    handler.flush()
    logs = log_capture.getvalue()

    # Verify sensitive IDs and financial values are completely absent from logs
    assert "PRIVATE_CUSTOMER_7777" not in logs
    assert "999999.0" not in logs

    # Ensure operational metadata is present
    assert "exp_security_isolation_test" in logs
    assert "total_masked_cells" in logs

    root_logger.removeHandler(handler)
