"""Security tests for Baseline Imputation: log sanitization and metadata privacy."""

import io
import logging

import pandas as pd

from missing_data_platform.config import LogLevel, Settings
from missing_data_platform.imputation.engine import BaselineImputationEngine
from missing_data_platform.logging import configure_logging


def test_imputation_logging_does_not_leak_customer_records() -> None:
    """Verify that imputation engine structured logs never emit customer record values or PII."""
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)

    settings = Settings(LOG_LEVEL=LogLevel.INFO, LOG_FORMAT="json")
    configure_logging(settings)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    df = pd.DataFrame(
        {
            "customer_id": ["TOP_SECRET_USER_101"],
            "income": [150000.0],
            "purchase_next_month": [1],
        }
    )

    engine = BaselineImputationEngine()
    result = engine.impute_dataset(df, experiment_id="exp_security_impute_audit")

    assert result.total_records == 1

    handler.flush()
    logs = log_capture.getvalue()

    # Verify sensitive customer values and IDs are absent from logs
    assert "TOP_SECRET_USER_101" not in logs
    assert "150000.0" not in logs

    # Ensure operational tracking is present
    assert "exp_security_impute_audit" in logs
    assert "total_cells_imputed" in logs

    root_logger.removeHandler(handler)
