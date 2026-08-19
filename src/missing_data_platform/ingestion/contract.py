"""Raw Data Contract definitions for incoming consumer datasets.

Defines expected schema, column types, nullability rules, and validation bounds
while explicitly preserving missing values as part of the core platform domain.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DataType(StrEnum):
    """Canonical data types supported by the raw data ingestion contract."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"


@dataclass(frozen=True)
class ColumnDefinition:
    """Specification of an individual column in the raw data contract."""

    name: str
    data_type: DataType
    nullable: bool = True
    is_identifier: bool = False
    is_target: bool = False
    allowed_categories: list[str] | None = None
    min_value: float | None = None
    max_value: float | None = None
    description: str = ""


@dataclass
class RawDataContract:
    """Contract specifying schema and structural rules for raw consumer datasets."""

    version: str = "1.0.0"
    id_column: str = "customer_id"
    target_column: str = "purchase_next_month"
    columns: dict[str, ColumnDefinition] = field(default_factory=dict)

    @classmethod
    def default_consumer_contract(cls) -> "RawDataContract":
        """Generate the standard raw consumer dataset contract."""
        cols = {
            "customer_id": ColumnDefinition(
                name="customer_id",
                data_type=DataType.STRING,
                nullable=False,
                is_identifier=True,
                description="Unique customer identifier",
            ),
            "age": ColumnDefinition(
                name="age",
                data_type=DataType.FLOAT,
                nullable=True,
                min_value=0.0,
                max_value=130.0,
                description="Customer age in years",
            ),
            "gender": ColumnDefinition(
                name="gender",
                data_type=DataType.STRING,
                nullable=True,
                description="Self-reported gender",
            ),
            "income": ColumnDefinition(
                name="income",
                data_type=DataType.FLOAT,
                nullable=True,
                min_value=0.0,
                description="Annual household income",
            ),
            "education": ColumnDefinition(
                name="education",
                data_type=DataType.STRING,
                nullable=True,
                description="Highest completed education level",
            ),
            "occupation": ColumnDefinition(
                name="occupation",
                data_type=DataType.STRING,
                nullable=True,
                description="Primary employment occupation category",
            ),
            "city": ColumnDefinition(
                name="city",
                data_type=DataType.STRING,
                nullable=True,
                description="Customer residence city",
            ),
            "region": ColumnDefinition(
                name="region",
                data_type=DataType.STRING,
                nullable=True,
                description="Geographical region or territory",
            ),
            "purchase_frequency": ColumnDefinition(
                name="purchase_frequency",
                data_type=DataType.FLOAT,
                nullable=True,
                min_value=0.0,
                description="Average monthly purchase transaction count",
            ),
            "average_purchase_value": ColumnDefinition(
                name="average_purchase_value",
                data_type=DataType.FLOAT,
                nullable=True,
                min_value=0.0,
                description="Average monetary value per transaction",
            ),
            "total_spend": ColumnDefinition(
                name="total_spend",
                data_type=DataType.FLOAT,
                nullable=True,
                min_value=0.0,
                description="Cumulative historical expenditure",
            ),
            "discount_usage": ColumnDefinition(
                name="discount_usage",
                data_type=DataType.FLOAT,
                nullable=True,
                min_value=0.0,
                max_value=1.0,
                description="Proportion of purchases made with discount/coupon",
            ),
            "website_visits": ColumnDefinition(
                name="website_visits",
                data_type=DataType.FLOAT,
                nullable=True,
                min_value=0.0,
                description="Monthly website visit frequency",
            ),
            "campaign_exposure": ColumnDefinition(
                name="campaign_exposure",
                data_type=DataType.FLOAT,
                nullable=True,
                min_value=0.0,
                description="Number of marketing campaign impressions received",
            ),
            "product_category": ColumnDefinition(
                name="product_category",
                data_type=DataType.STRING,
                nullable=True,
                description="Primary product category of interest",
            ),
            "customer_segment": ColumnDefinition(
                name="customer_segment",
                data_type=DataType.STRING,
                nullable=True,
                description="Business tier or behavioral cluster",
            ),
            "purchase_next_month": ColumnDefinition(
                name="purchase_next_month",
                data_type=DataType.INTEGER,
                nullable=False,
                is_target=True,
                min_value=0.0,
                max_value=1.0,
                description="Target indicator: 1 = purchase next month, 0 = no purchase",
            ),
        }
        return cls(columns=cols)

    @property
    def required_columns(self) -> list[str]:
        """List of non-nullable required columns."""
        return [col for col, defn in self.columns.items() if not defn.nullable]

    @property
    def nullable_columns(self) -> list[str]:
        """List of columns that legitimately accept missing values."""
        return [col for col, defn in self.columns.items() if defn.nullable]

    @property
    def column_names(self) -> list[str]:
        """Complete list of expected column names."""
        return list(self.columns.keys())

    def get_column(self, name: str) -> ColumnDefinition | None:
        """Retrieve column definition metadata by name."""
        return self.columns.get(name)

    def to_dict(self) -> dict[str, Any]:
        """Export contract specification as dictionary metadata."""
        return {
            "version": self.version,
            "id_column": self.id_column,
            "target_column": self.target_column,
            "columns": {
                name: {
                    "data_type": defn.data_type.value,
                    "nullable": defn.nullable,
                    "is_identifier": defn.is_identifier,
                    "is_target": defn.is_target,
                    "min_value": defn.min_value,
                    "max_value": defn.max_value,
                }
                for name, defn in self.columns.items()
            },
        }
