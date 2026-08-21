"""Unit tests for BaselineImputer statistical operations (Mean, Median, Mode, Constant)."""

import numpy as np
import pandas as pd
import pytest

from missing_data_platform.exceptions import DataQualityError, ImputationError
from missing_data_platform.imputation.baseline import BaselineImputer
from missing_data_platform.imputation.config import (
    BaselineImputationConfig,
    BaselineStrategy,
)


@pytest.fixture
def baseline_sample_df() -> pd.DataFrame:
    """Fixture with numeric and categorical columns with known missingness."""
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4", "C5"],
            "income": [10000.0, 20000.0, np.nan, 40000.0, 80000.0],  # Mean=37500, Median=30000
            "city": ["NYC", "LA", np.nan, "NYC", "CHI"],  # Mode=NYC
            "purchase_next_month": [1, 0, np.nan, 1, 0],
        }
    )


def test_baseline_imputer_mean(baseline_sample_df: pd.DataFrame) -> None:
    """Verify mean imputation on numerical columns."""
    config = BaselineImputationConfig(
        numeric_strategy=BaselineStrategy.MEAN,
        categorical_strategy=BaselineStrategy.MODE,
        target_features=["income", "city"],
    )
    imputer = BaselineImputer(config=config)
    imputed_df, metrics = imputer.fit(baseline_sample_df).transform_with_metrics(baseline_sample_df)

    assert imputer.imputation_parameters["income"] == 37500.0
    assert imputed_df.loc[2, "income"] == 37500.0
    assert imputed_df["income"].isna().sum() == 0

    # Protected columns untouched
    assert pd.isna(imputed_df.loc[2, "purchase_next_month"])


def test_baseline_imputer_median(baseline_sample_df: pd.DataFrame) -> None:
    """Verify median imputation on numerical columns."""
    config = BaselineImputationConfig(
        numeric_strategy=BaselineStrategy.MEDIAN,
        categorical_strategy=BaselineStrategy.MODE,
        target_features=["income", "city"],
    )
    imputer = BaselineImputer(config=config)
    imputed_df = imputer.fit_transform(baseline_sample_df)

    assert imputer.imputation_parameters["income"] == 30000.0
    assert imputed_df.loc[2, "income"] == 30000.0


def test_baseline_imputer_mode_tie_breaking() -> None:
    """Verify deterministic alphabetical tie-breaking for categorical mode imputation."""
    df = pd.DataFrame(
        {
            "category": ["Beta", "Alpha", np.nan],  # 'Alpha' and 'Beta' each have frequency 1
        }
    )
    config = BaselineImputationConfig(categorical_strategy=BaselineStrategy.MODE)
    imputer = BaselineImputer(config=config)
    imputed_df = imputer.fit_transform(df)

    # Alphabetical order: 'Alpha' comes before 'Beta'
    assert imputer.imputation_parameters["category"] == "Alpha"
    assert imputed_df.loc[2, "category"] == "Alpha"


def test_all_missing_column_raises_error() -> None:
    """Verify that attempting to compute mean/median on 100% missing column raises ImputationError."""
    df = pd.DataFrame(
        {
            "all_null_num": [np.nan, np.nan, np.nan],
        }
    )
    imputer = BaselineImputer()
    with pytest.raises(ImputationError) as exc_info:
        imputer.fit(df)
    assert "all values are missing" in str(exc_info.value)


def test_empty_dataframe_raises_error() -> None:
    """Verify that fitting an empty DataFrame raises DataQualityError."""
    imputer = BaselineImputer()
    with pytest.raises(DataQualityError):
        imputer.fit(pd.DataFrame())


def test_baseline_imputer_constant_strategy() -> None:
    """Verify constant fill value imputation on numerical and categorical columns."""
    df = pd.DataFrame(
        {
            "num_col": [1.0, np.nan, 3.0],
            "cat_col": ["A", np.nan, "B"],
        }
    )
    config = BaselineImputationConfig(
        numeric_strategy=BaselineStrategy.CONSTANT,
        categorical_strategy=BaselineStrategy.CONSTANT,
        constant_fill_value=-999,
    )
    imputer = BaselineImputer(config=config)
    imputed_df = imputer.fit_transform(df)

    assert imputed_df.loc[1, "num_col"] == -999
    assert imputed_df.loc[1, "cat_col"] == "-999"


def test_transform_without_fit_raises_error() -> None:
    """Verify that calling transform on an unfitted imputer raises ImputationError."""
    df = pd.DataFrame({"income": [10.0, np.nan]})
    imputer = BaselineImputer()
    with pytest.raises(ImputationError) as exc_info:
        imputer.transform(df)
    assert "must be fitted" in str(exc_info.value)


def test_all_missing_categorical_raises_error() -> None:
    """Verify that attempting to compute mode on 100% missing categorical column raises ImputationError."""
    df = pd.DataFrame(
        {
            "all_null_cat": pd.Series([np.nan, np.nan], dtype="object"),
        }
    )
    config = BaselineImputationConfig(categorical_strategy=BaselineStrategy.MODE)
    imputer = BaselineImputer(config=config)
    with pytest.raises(ImputationError) as exc_info:
        imputer.fit(df)
    assert "Cannot compute mode" in str(exc_info.value)


def test_invalid_numeric_conversion_raises_error() -> None:
    """Verify that a column declared numeric but containing non-convertible strings raises ImputationError."""
    from missing_data_platform.ingestion.contract import ColumnDefinition, DataType, RawDataContract

    df = pd.DataFrame(
        {
            "bad_num": ["abc", "def", np.nan],
        }
    )
    contract = RawDataContract(
        id_column="customer_id",
        target_column="purchase_next_month",
        columns={
            "bad_num": ColumnDefinition(name="bad_num", data_type=DataType.FLOAT),
        },
    )
    config = BaselineImputationConfig(
        numeric_strategy=BaselineStrategy.MEAN, target_features=["bad_num"]
    )
    imputer = BaselineImputer(config=config, contract=contract)
    with pytest.raises(ImputationError) as exc_info:
        imputer.fit(df)
    assert "no valid numeric values" in str(exc_info.value)
