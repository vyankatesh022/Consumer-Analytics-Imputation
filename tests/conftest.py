"""Global Pytest test configuration and fixtures."""

import os
from collections.abc import Generator

import pytest

from missing_data_platform.config import AppEnvironment, LogLevel, Settings


@pytest.fixture
def test_settings() -> Settings:
    """Fixture providing clean, isolated Settings for unit testing."""
    return Settings(
        APP_ENV=AppEnvironment.TEST,
        APP_NAME="test-missing-data-platform",
        APP_DEBUG=True,
        LOG_LEVEL=LogLevel.DEBUG,
        LOG_FORMAT="console",
        RANDOM_SEED=123,
        USE_LOCAL_STORAGE=True,
        S3_BUCKET_NAME="test-bucket",
        AWS_REGION="us-east-1",
    )


@pytest.fixture(autouse=True)
def clean_env() -> Generator[None, None, None]:
    """Ensure environment variables are clean before each test execution."""
    old_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(old_env)
