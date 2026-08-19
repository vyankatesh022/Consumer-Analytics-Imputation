"""End-to-end smoke test verifying the overall foundation readiness."""

import missing_data_platform
from missing_data_platform.config import get_settings
from missing_data_platform.logging import configure_logging, get_logger


def test_e2e_foundation_smoke() -> None:
    """Smoke test ensuring full startup sequence operates without unhandled exceptions."""
    settings = get_settings()
    configure_logging(settings)
    logger = get_logger("e2e_smoke")

    logger.info(
        "Executing foundation smoke test",
        version=missing_data_platform.__version__,
        env=settings.APP_ENV.value,
    )
    assert missing_data_platform.__version__ is not None
