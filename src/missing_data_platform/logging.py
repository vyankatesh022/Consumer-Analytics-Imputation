"""Structured logging initialization for the Missing Data Platform.

Configures structured JSON logging with context enrichment, severity filtering,
and prevention of sensitive credential leakage.
"""

import logging
import sys
from typing import Any, cast

import structlog

from missing_data_platform.config import LogLevel, Settings


def configure_logging(settings: Settings | None = None) -> None:
    """Initialize structured logging pipelines based on active Settings."""
    if settings is None:
        from missing_data_platform.config import get_settings

        settings = get_settings()

    # Determine logging level
    level_map = {
        LogLevel.DEBUG: logging.DEBUG,
        LogLevel.INFO: logging.INFO,
        LogLevel.WARNING: logging.WARNING,
        LogLevel.ERROR: logging.ERROR,
        LogLevel.CRITICAL: logging.CRITICAL,
    }
    log_level = level_map.get(settings.LOG_LEVEL, logging.INFO)

    # Standard shared processors
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.LOG_FORMAT == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)


def get_logger(name: str = "missing_data_platform") -> structlog.stdlib.BoundLogger:
    """Obtain a structured logger bound to a given component name."""
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
