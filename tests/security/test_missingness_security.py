"""Security tests for Missingness Analysis Layer: log privacy and statistical data isolation."""

import io
import logging

import numpy as np
import pandas as pd

from missing_data_platform.config import LogLevel, Settings
from missing_data_platform.logging import configure_logging
from missing_data_platform.missingness.engine import MissingnessAnalysisEngine


def test_missingness_analysis_logging_does_not_leak_customer_records() -> None:
    """Verify that missingness engine structured logs contain zero customer row values or PII."""
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)

    settings = Settings(LOG_LEVEL=LogLevel.INFO, LOG_FORMAT="json")
    configure_logging(settings)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    data = {
        "customer_id": ["PRIVATE_CUST_9999", "PRIVATE_CUST_8888"],
        "age": [55.0, np.nan],
        "gender": ["Female", "Male"],
        "income": [np.nan, 210000.0],
        "region": ["PrivateTerritory", "PrivateTerritory"],
        "purchase_frequency": [4.0, 5.0],
        "average_purchase_value": [300.0, 400.0],
        "total_spend": [1200.0, 2000.0],
        "discount_usage": [0.0, 0.1],
        "website_visits": [15.0, 20.0],
        "campaign_exposure": [2.0, 3.0],
        "product_category": ["Luxury", "Luxury"],
        "customer_segment": ["Ultra_HNW", "Ultra_HNW"],
        "purchase_next_month": [1, 0],
    }
    df = pd.DataFrame(data)

    engine = MissingnessAnalysisEngine()
    report = engine.analyze(df, dataset_id="missingness_security_audit")

    assert report.total_records == 2

    handler.flush()
    logs = log_capture.getvalue()

    # Assert confidential customer identifiers and values are absent from logs
    assert "PRIVATE_CUST_9999" not in logs
    assert "PrivateTerritory" not in logs
    assert "210000.0" not in logs

    # Ensure operational metrics are logged safely
    assert "missingness_security_audit" in logs
    assert "missing_features_count" in logs

    root_logger.removeHandler(handler)
