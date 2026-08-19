"""Unit tests for RawDataContract and ColumnDefinition."""

from missing_data_platform.ingestion.contract import (
    ColumnDefinition,
    DataType,
    RawDataContract,
)


def test_default_consumer_contract_structure() -> None:
    """Verify that default consumer contract defines all required and nullable columns."""
    contract = RawDataContract.default_consumer_contract()
    assert contract.id_column == "customer_id"
    assert contract.target_column == "purchase_next_month"
    assert "customer_id" in contract.required_columns
    assert "purchase_next_month" in contract.required_columns

    # Verify nullable fields
    assert "age" in contract.nullable_columns
    assert "income" in contract.nullable_columns
    assert "discount_usage" in contract.nullable_columns
    assert "website_visits" in contract.nullable_columns

    # Check bounds
    age_col = contract.get_column("age")
    assert age_col is not None
    assert age_col.min_value == 0.0
    assert age_col.max_value == 130.0
    assert age_col.data_type == DataType.FLOAT


def test_contract_serialization() -> None:
    """Verify that RawDataContract serializes to a structured dictionary."""
    contract = RawDataContract.default_consumer_contract()
    data_dict = contract.to_dict()
    assert data_dict["version"] == "1.0.0"
    assert data_dict["id_column"] == "customer_id"
    assert "columns" in data_dict
    assert len(data_dict["columns"]) == len(contract.columns)


def test_custom_column_definition() -> None:
    """Verify creation of custom ColumnDefinition instances."""
    col = ColumnDefinition(
        name="custom_metric",
        data_type=DataType.FLOAT,
        nullable=True,
        min_value=-10.0,
        max_value=10.0,
    )
    assert col.name == "custom_metric"
    assert col.min_value == -10.0
    assert col.max_value == 10.0
    assert col.nullable is True
