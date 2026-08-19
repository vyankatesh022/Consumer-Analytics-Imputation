"""Unit tests for repository foundation, package exports, and exceptions."""

import missing_data_platform
from missing_data_platform.exceptions import (
    ConfigurationError,
    DataQualityError,
    EvaluationError,
    ImputationError,
    ModelTrainingError,
    PlatformError,
    StorageError,
)
from missing_data_platform.logging import configure_logging, get_logger


def test_package_import_and_version() -> None:
    """Verify that the core package can be imported and exports valid metadata."""
    assert hasattr(missing_data_platform, "__version__")
    assert isinstance(missing_data_platform.__version__, str)
    assert len(missing_data_platform.__version__.split(".")) >= 3
    assert hasattr(missing_data_platform, "__author__")


def test_exception_hierarchy() -> None:
    """Verify that custom exception hierarchy works as expected."""
    base_err = PlatformError("Base platform error", context={"key": "val"})
    assert "Base platform error" in str(base_err)
    assert "val" in str(base_err)

    # Subclasses inherit properly
    assert issubclass(ConfigurationError, PlatformError)
    assert issubclass(StorageError, PlatformError)
    assert issubclass(DataQualityError, PlatformError)
    assert issubclass(ImputationError, PlatformError)
    assert issubclass(EvaluationError, PlatformError)
    assert issubclass(ModelTrainingError, PlatformError)


def test_logging_initialization(test_settings) -> None:
    """Verify that structured logging initializes and binds loggers correctly."""
    configure_logging(test_settings)
    logger = get_logger("unit_test_logger")
    assert logger is not None
    # Exercise logging output without error
    logger.info("Test logging message", test_key="test_value")
