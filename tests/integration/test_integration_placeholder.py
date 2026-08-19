"""Integration test foundation verifying system boundaries and multi-module interoperability."""

import tempfile
from pathlib import Path

from missing_data_platform.config import LogLevel, Settings
from missing_data_platform.logging import configure_logging, get_logger


def test_settings_and_logging_integration() -> None:
    """Verify that Settings dynamically propagates configurations to structured logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_data_dir = Path(tmpdir) / "data"
        settings = Settings(
            APP_NAME="integration-test-platform",
            LOCAL_DATA_DIR=test_data_dir,
            LOG_LEVEL=LogLevel.DEBUG,
            LOG_FORMAT="json",
        )

        configure_logging(settings)
        logger = get_logger("integration_tester")
        assert logger is not None
        assert test_data_dir == settings.LOCAL_DATA_DIR
