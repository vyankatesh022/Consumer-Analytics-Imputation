"""Security tests for Random Forest Imputation: model parameter isolation and log sanitization."""

import io
import logging

import pandas as pd

from missing_data_platform.config import LogLevel, Settings
from missing_data_platform.imputation.engine import BaselineImputationEngine
from missing_data_platform.logging import configure_logging


def test_rf_logging_does_not_leak_customer_records() -> None:
    """Verify that Random Forest imputation logs never emit customer values, tree matrices, or PII."""
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)

    settings = Settings(LOG_LEVEL=LogLevel.INFO, LOG_FORMAT="json")
    configure_logging(settings)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    df = pd.DataFrame(
        {
            "customer_id": ["PRIVATE_CUSTOMER_VIP_999"],
            "age": [45.0],
            "income": [450000.0],
            "purchase_next_month": [1],
        }
    )

    engine = BaselineImputationEngine()
    result = engine.impute_rf_dataset(
        df,
        experiment_id="exp_rf_security_audit",
        n_estimators=10,
    )

    assert result.total_records == 1

    handler.flush()
    logs = log_capture.getvalue()

    # Verify sensitive customer values and IDs are absent from logs
    assert "PRIVATE_CUSTOMER_VIP_999" not in logs
    assert "450000.0" not in logs

    # Ensure operational tracking is present
    assert "exp_rf_security_audit" in logs
    assert "n_estimators" in logs

    root_logger.removeHandler(handler)
