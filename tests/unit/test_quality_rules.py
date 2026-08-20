"""Unit tests for Data Quality rules and configuration."""

from missing_data_platform.quality.rules import (
    DataQualityConfig,
    QualityRule,
    QualitySeverity,
    QualityStatus,
)


def test_quality_status_and_severity_enums() -> None:
    """Verify QualityStatus and QualitySeverity values."""
    assert QualityStatus.PASS == "PASS"
    assert QualityStatus.WARN == "WARN"
    assert QualityStatus.FAIL == "FAIL"

    assert QualitySeverity.INFO == "INFO"
    assert QualitySeverity.WARNING == "WARNING"
    assert QualitySeverity.ERROR == "ERROR"


def test_default_data_quality_config() -> None:
    """Verify default configuration thresholds."""
    config = DataQualityConfig()
    assert config.max_missing_percentage_warning == 30.0
    assert config.max_missing_percentage_error == 80.0
    assert config.allow_duplicate_records is False
    assert config.allow_duplicate_ids is False
    assert config.strict_schema_matching is True


def test_custom_quality_rule() -> None:
    """Verify custom quality rule creation."""
    rule = QualityRule(
        name="custom_null_cap",
        description="Cap missingness at 15%",
        severity_on_failure=QualitySeverity.WARNING,
        threshold=15.0,
    )
    assert rule.name == "custom_null_cap"
    assert rule.threshold == 15.0
    assert rule.severity_on_failure == QualitySeverity.WARNING
