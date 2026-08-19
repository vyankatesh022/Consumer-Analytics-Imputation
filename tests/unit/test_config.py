"""Unit tests for configuration management and secret masking."""

import os
from pathlib import Path

from pydantic import SecretStr

from missing_data_platform.config import AppEnvironment, LogLevel, Settings, get_settings


def test_default_settings() -> None:
    """Verify that default settings instantiate with production-ready defaults."""
    settings = get_settings()
    assert settings.APP_NAME == "missing-data-platform"
    assert settings.APP_ENV in [
        AppEnvironment.DEVELOPMENT,
        AppEnvironment.STAGING,
        AppEnvironment.PRODUCTION,
        AppEnvironment.TEST,
    ]
    assert settings.LOG_LEVEL in [
        LogLevel.DEBUG,
        LogLevel.INFO,
        LogLevel.WARNING,
        LogLevel.ERROR,
        LogLevel.CRITICAL,
    ]
    assert settings.RANDOM_SEED == 42
    assert isinstance(settings.LOCAL_DATA_DIR, Path)


def test_environment_variable_override() -> None:
    """Verify that environment variables cleanly override defaults."""
    os.environ["APP_ENV"] = "production"
    os.environ["LOG_LEVEL"] = "ERROR"
    os.environ["RANDOM_SEED"] = "999"
    os.environ["USE_LOCAL_STORAGE"] = "false"
    os.environ["S3_BUCKET_NAME"] = "custom-prod-bucket"

    settings = Settings()
    assert settings.APP_ENV == AppEnvironment.PRODUCTION
    assert settings.LOG_LEVEL == LogLevel.ERROR
    assert settings.RANDOM_SEED == 999
    assert settings.USE_LOCAL_STORAGE is False
    assert settings.S3_BUCKET_NAME == "custom-prod-bucket"


def test_secret_string_masking() -> None:
    """Verify that sensitive fields use SecretStr and are never exposed in plaintext __str__ or __repr__."""
    settings = Settings(
        AWS_ACCESS_KEY_ID=SecretStr("super-secret-key-12345"),
        API_KEY_SECRET=SecretStr("super-secret-api-token-999"),
    )

    # String representations must mask secrets
    assert "super-secret-key-12345" not in str(settings.AWS_ACCESS_KEY_ID)
    assert "super-secret-key-12345" not in repr(settings.AWS_ACCESS_KEY_ID)
    assert "super-secret-api-token-999" not in str(settings.API_KEY_SECRET)
    assert "super-secret-api-token-999" not in repr(settings.API_KEY_SECRET)

    # Secret retrieval works explicitly
    assert settings.AWS_ACCESS_KEY_ID is not None
    assert settings.API_KEY_SECRET is not None
    assert settings.AWS_ACCESS_KEY_ID.get_secret_value() == "super-secret-key-12345"
    assert settings.API_KEY_SECRET.get_secret_value() == "super-secret-api-token-999"
