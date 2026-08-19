"""Security tests for ingestion layer: log sanitization and input resilience."""

import io
import logging

from missing_data_platform.config import LogLevel, Settings
from missing_data_platform.ingestion.engine import IngestionEngine
from missing_data_platform.logging import configure_logging


def test_ingestion_logging_does_not_leak_customer_records() -> None:
    """Verify that structured ingestion logs contain operational metadata only and no customer rows."""
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)

    settings = Settings(LOG_LEVEL=LogLevel.INFO, LOG_FORMAT="json")
    configure_logging(settings)

    # Attach capture handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    raw_csv = (
        "customer_id,age,gender,income,education,occupation,city,region,purchase_frequency,"
        "average_purchase_value,total_spend,discount_usage,website_visits,campaign_exposure,"
        "product_category,customer_segment,purchase_next_month\n"
        "SECRET_CUST_9999,35,Female,150000.0,PhD,Executive,SecretCity,West,5.0,200.0,1000.0,0.1,10,2,Luxury,VIP,1\n"
    )

    engine = IngestionEngine()
    result = engine.ingest_content(raw_csv, dataset_id="security_audit_dataset")

    assert len(result.valid_data) == 1

    handler.flush()
    logs = log_capture.getvalue()

    # Assert customer confidential tokens/values never appear in logs
    assert "SECRET_CUST_9999" not in logs
    assert "SecretCity" not in logs
    assert "150000.0" not in logs

    # Operational metrics should be logged safely
    assert "security_audit_dataset" in logs
    assert "total_records" in logs

    root_logger.removeHandler(handler)
