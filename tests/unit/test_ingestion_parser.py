"""Unit tests for CsvParser engine."""

import tempfile
from pathlib import Path

import pytest

from missing_data_platform.exceptions import DataQualityError
from missing_data_platform.ingestion.parser import CsvParser


def test_parse_valid_csv_file(sample_csv_path: Path) -> None:
    """Verify parsing a valid CSV file preserving null entries."""
    parser = CsvParser()
    df = parser.parse_file(sample_csv_path)
    assert not df.empty
    assert len(df) == 10
    assert "customer_id" in df.columns

    # Record 2 has empty age -> must be parsed as NaN / null
    assert pd_isna(df.loc[1, "age"])
    # Record 3 has empty income -> must be parsed as NaN / null
    assert pd_isna(df.loc[2, "income"])


def test_parse_missing_file_raises_error() -> None:
    """Verify that attempting to parse a non-existent file raises DataQualityError."""
    parser = CsvParser()
    with pytest.raises(DataQualityError) as exc_info:
        parser.parse_file("/tmp/definitely_non_existent_file_12345.csv")
    assert "not found" in str(exc_info.value)


def test_parse_empty_file_raises_error() -> None:
    """Verify that parsing an empty 0-byte file raises DataQualityError."""
    parser = CsvParser()
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with pytest.raises(DataQualityError) as exc_info:
            parser.parse_file(tmp_path)
        assert "empty" in str(exc_info.value)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_parse_in_memory_content() -> None:
    """Verify parsing from in-memory string content."""
    parser = CsvParser()
    content = "customer_id,age,income\nC01,25,50000\nC02,,60000\n"
    df = parser.parse_string(content)

    assert len(df) == 2
    assert df.loc[0, "customer_id"] == "C01"
    assert df.loc[0, "age"] == "25"
    assert pd_isna(df.loc[1, "age"])


def test_parse_in_memory_empty_content_raises_error() -> None:
    """Verify that parsing empty string raises DataQualityError."""
    parser = CsvParser()
    with pytest.raises(DataQualityError) as exc_info:
        parser.parse_string("   ")
    assert "empty" in str(exc_info.value)


def test_parse_csv_file_with_chunksize(sample_csv_path: Path) -> None:
    """Verify parsing large CSV with chunked processing."""
    parser = CsvParser()
    df = parser.parse_file(sample_csv_path, chunksize=3)
    assert len(df) == 10
    assert pd_isna(df.loc[1, "age"])


def pd_isna(val: object) -> bool:
    """Helper to check if a value is NA/None/NaN."""
    import pandas as pd

    return bool(pd.isna(val))
