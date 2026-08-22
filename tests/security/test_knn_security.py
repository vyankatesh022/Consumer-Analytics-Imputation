"""Security tests for KNN Imputation: neighbor privacy and structured log sanitization."""

import io
import logging

import pandas as pd

from missing_data_platform.config import LogLevel, Settings
from missing_data_platform.imputation.engine import BaselineImputationEngine
from missing_data_platform.logging import configure_logging


def test_knn_logging_does_not_leak_customer_records_or_neighbor_identities() -> None:
    """Verify that KNN imputation logs never emit customer values, neighbor IDs, or confidential records."""
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)

    settings = Settings(LOG_LEVEL=LogLevel.INFO, LOG_FORMAT="json")
    configure_logging(settings)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    df = pd.DataFrame(
        {
            "customer_id": ["CONFIDENTIAL_CUSTOMER_9999"],
            "age": [45.0],
            "income": [250000.0],
            "total_spend": [1500.0],
            "purchase_next_month": [1],
        }
    )

    engine = BaselineImputationEngine()
    result = engine.impute_knn_dataset(df, experiment_id="exp_knn_security_audit")

    assert result.total_records == 1

    handler.flush()
    logs = log_capture.getvalue()

    # Verify sensitive customer values and IDs are absent from logs
    assert "CONFIDENTIAL_CUSTOMER_9999" not in logs
    assert "250000.0" not in logs

    # Ensure operational tracking is present
    assert "exp_knn_security_audit" in logs
    assert "n_neighbors" in logs

    root_logger.removeHandler(handler)
