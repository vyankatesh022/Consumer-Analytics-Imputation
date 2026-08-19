"""Domain-specific exception hierarchy for the Missing Data Platform.

All custom platform exceptions inherit from PlatformError to ensure clean
error handling, context enrichment, and prevention of sensitive detail leakage.
"""


class PlatformError(Exception):
    """Base exception for all errors raised by the missing data platform."""

    def __init__(self, message: str, context: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} | Context: {self.context}"
        return self.message


class ConfigurationError(PlatformError):
    """Raised when application configuration or environment variables are invalid."""

    pass


class StorageError(PlatformError):
    """Raised when data lakehouse or storage I/O operations fail."""

    pass


class DataQualityError(PlatformError):
    """Raised when data contracts or schema validation rules are violated."""

    pass


class ImputationError(PlatformError):
    """Raised when an imputation strategy algorithm fails during execution."""

    pass


class EvaluationError(PlatformError):
    """Raised when metric or representation bias computation fails."""

    pass


class ModelTrainingError(PlatformError):
    """Raised when downstream ML training or cross-validation fails."""

    pass
