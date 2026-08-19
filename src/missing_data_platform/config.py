"""Application configuration management using Pydantic Settings.

Provides type-safe environment variable parsing with validation, sensible defaults,
and secure handling of credentials.
"""

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnvironment(StrEnum):
    """Supported application runtime environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class LogLevel(StrEnum):
    """Supported logging severity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Central configuration settings for the Missing Data Platform."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Core Application Settings
    APP_ENV: AppEnvironment = Field(
        default=AppEnvironment.DEVELOPMENT,
        description="Current runtime deployment environment",
    )
    APP_NAME: str = Field(
        default="missing-data-platform",
        description="Standardized platform name identifier",
    )
    APP_DEBUG: bool = Field(
        default=False,
        description="Enable debug mode (disables production assertions)",
    )

    # Logging Configuration
    LOG_LEVEL: LogLevel = Field(
        default=LogLevel.INFO,
        description="Active structured logging severity threshold",
    )
    LOG_FORMAT: Literal["json", "console"] = Field(
        default="json",
        description="Structured log emission format (json or console)",
    )

    # Reproducibility
    RANDOM_SEED: int = Field(
        default=42,
        description="Global pseudo-random seed for deterministic experimentation",
    )

    # Storage & Lakehouse Configuration
    USE_LOCAL_STORAGE: bool = Field(
        default=True,
        description="Fallback to local disk filesystem instead of AWS S3",
    )
    LOCAL_DATA_DIR: Path = Field(
        default=Path("./data"),
        description="Local data storage root path",
    )
    S3_BUCKET_NAME: str = Field(
        default="missing-data-platform-lakehouse",
        description="AWS S3 target bucket name",
    )
    AWS_REGION: str = Field(
        default="us-east-1",
        description="Target AWS cloud region",
    )
    AWS_ACCESS_KEY_ID: SecretStr | None = Field(
        default=None,
        description="AWS access key identifier (optional if using IAM role)",
    )
    AWS_SECRET_ACCESS_KEY: SecretStr | None = Field(
        default=None,
        description="AWS secret access key (optional if using IAM role)",
    )
    AWS_S3_ENDPOINT_URL: str | None = Field(
        default=None,
        description="Custom S3 endpoint URL (used for LocalStack / MinIO testing)",
    )

    # MLflow Experiment Tracking
    MLFLOW_TRACKING_URI: str = Field(
        default="http://localhost:5000",
        description="URI for remote or local MLflow tracking server",
    )
    MLFLOW_EXPERIMENT_NAME: str = Field(
        default="missing_data_imputation_study",
        description="Active MLflow experiment workspace name",
    )

    # API Service Parameters
    API_HOST: str = Field(
        default="0.0.0.0",
        description="FastAPI service binding interface host",
    )
    API_PORT: int = Field(
        default=8000,
        description="FastAPI service binding port",
    )
    API_KEY_SECRET: SecretStr | None = Field(
        default=None,
        description="Secret key token for API authentication",
    )
    API_RATE_LIMIT_PER_MINUTE: int = Field(
        default=100,
        description="Maximum requests allowed per client IP per minute",
    )


def get_settings() -> Settings:
    """Retrieve an initialized instance of application Settings."""
    return Settings()
