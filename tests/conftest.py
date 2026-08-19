"""Global Pytest test configuration and fixtures."""

import os
from collections.abc import Generator
from pathlib import Path

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


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    """Dynamically generate a synthetic raw consumer CSV file for testing."""
    csv_content = (
        "customer_id,age,gender,income,education,occupation,city,region,purchase_frequency,"
        "average_purchase_value,total_spend,discount_usage,website_visits,campaign_exposure,"
        "product_category,customer_segment,purchase_next_month\n"
        "CUST_0001,34,Female,65000.0,Bachelor,Engineer,Seattle,West,4.5,82.50,371.25,0.15,12,3,Electronics,Gold,1\n"
        "CUST_0002,,Male,48000.0,High School,Sales,Austin,South,2.0,45.00,90.00,0.00,5,1,Apparel,Silver,0\n"
        "CUST_0003,45,Female,,Master,Manager,Chicago,Midwest,8.0,120.00,960.00,0.30,22,5,Home,Platinum,1\n"
        "CUST_0004,28,Non-Binary,52000.0,Bachelor,,Boston,Northeast,1.5,60.00,90.00,0.10,8,2,Beauty,Bronze,0\n"
        "CUST_0005,52,Male,95000.0,PhD,Executive,New York,Northeast,6.0,210.00,1260.00,,15,4,Electronics,Platinum,1\n"
        "CUST_0006,22,Female,28000.0,Some College,Student,Denver,West,3.0,35.00,105.00,0.50,18,6,Apparel,Bronze,0\n"
        "CUST_0007,39,Male,78000.0,Bachelor,Marketing,San Francisco,West,,95.00,475.00,0.20,10,3,Home,Gold,1\n"
        "CUST_0008,61,Female,82000.0,Master,Retired,Miami,South,5.0,,600.00,0.10,6,1,Health,Gold,0\n"
        "CUST_0009,31,Male,59000.0,Bachelor,Analyst,Dallas,South,4.0,75.00,,0.25,,2,Electronics,Silver,1\n"
        "CUST_0010,48,Female,110000.0,Master,Director,Los Angeles,West,7.0,180.00,1260.00,0.05,25,,Luxury,Platinum,1\n"
    )
    file_path = tmp_path / "synthetic_consumer_sample.csv"
    file_path.write_text(csv_content, encoding="utf-8")
    return file_path
