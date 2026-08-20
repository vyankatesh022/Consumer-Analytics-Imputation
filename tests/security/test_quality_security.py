"""Security tests for Data Quality Layer: log sanitization and privacy isolation."""

import io
import logging

import pandas as pd

from missing_data_platform.config import LogLevel, Settings
from missing_data_platform.logging import configure_logging
from missing_data_platform.quality.engine import DataQualityEngine


def test_quality_audit_logging_does_not_leak_customer_records() -> None:
    """Verify that quality engine structured logs contain zero customer row values or PII."""
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)

    settings = Settings(LOG_LEVEL=LogLevel.INFO, LOG_FORMAT="json")
    configure_logging(settings)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    data = {
        "customer_id": ["CONFIDENTIAL_ID_8888"],
        "age": [45.0],
        "gender": ["Female"],
        "income": [195000.0],
        "education": ["Executive MBA"],
        "occupation": ["Chief Risk Officer"],
        "city": ["ConfidentialCity"],
        "region": ["East"],
        "purchase_frequency": [4.0],
        "average_purchase_value": [300.0],
        "total_spend": [1200.0],
        "discount_usage": [0.0],
        "website_visits": [15.0],
        "campaign_exposure": [2.0],
        "product_category": ["Luxury"],
        "customer_segment": ["VIP Diamond"],
        "purchase_next_month": [1],
    }
    df = pd.DataFrame(data)

    engine = DataQualityEngine()
    report = engine.audit_dataset(df, dataset_id="audit_security_test")

    assert report.total_records == 1

    handler.flush()
    logs = log_capture.getvalue()

    # Assert confidential customer identifiers and field values are never in logs
    assert "CONFIDENTIAL_ID_8888" not in logs
    assert "ConfidentialCity" not in logs
    assert "Chief Risk Officer" not in logs
    assert "195000.0" not in logs

    # Ensure operational metrics are safely recorded
    assert "audit_security_test" in logs
    assert "overall_status" in logs

    root_logger.removeHandler(handler)
