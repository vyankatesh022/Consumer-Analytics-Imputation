"""CSV parsing engine for raw tabular consumer datasets.

Parses CSV files and streams with deterministic missing-value preservation,
encoding handling, and support for chunked processing.
"""

from io import StringIO
from pathlib import Path
from typing import TextIO

import pandas as pd

from missing_data_platform.exceptions import DataQualityError


class CsvParser:
    """Robust parser for CSV input sources preserving genuine missing values."""

    # Standard explicit null value representations to treat as missing
    DEFAULT_NA_VALUES: list[str] = [
        "",
        "#N/A",
        "#N/A N/A",
        "#NA",
        "-1.#IND",
        "-1.#QNAN",
        "-NaN",
        "-nan",
        "1.#IND",
        "1.#QNAN",
        "<NA>",
        "N/A",
        "NA",
        "NULL",
        "NaN",
        "None",
        "n/a",
        "nan",
        "null",
    ]

    def __init__(self, na_values: list[str] | None = None) -> None:
        self.na_values = na_values or self.DEFAULT_NA_VALUES

    def parse_file(
        self,
        file_path: Path | str,
        chunksize: int | None = None,
        encoding: str = "utf-8",
    ) -> pd.DataFrame:
        """Parse a local CSV file into a pandas DataFrame preserving nulls.

        Raises:
            DataQualityError: If file is missing, empty, or unparseable.
        """
        path = Path(file_path)
        if not path.exists():
            raise DataQualityError(
                f"Raw input file not found at: {path}",
                context={"file_path": str(path)},
            )

        if path.stat().st_size == 0:
            raise DataQualityError(
                f"Raw input file is empty: {path}",
                context={"file_path": str(path)},
            )

        try:
            df = pd.read_csv(
                path,
                na_values=self.na_values,
                keep_default_na=True,
                encoding=encoding,
                chunksize=chunksize,
                dtype=object,  # Read raw as object first to prevent silent coercive truncation
            )
            if isinstance(df, pd.io.parsers.TextFileReader):
                # If chunked, concatenate all chunks into unified canonical dataframe
                df = pd.concat(df, ignore_index=True)

            if df.empty:
                raise DataQualityError(
                    f"Parsed CSV contains headers but 0 data rows: {path}",
                    context={"file_path": str(path)},
                )
            return df
        except UnicodeDecodeError as err:
            raise DataQualityError(
                f"Encoding error parsing {path} with encoding {encoding}: {err}",
                context={"file_path": str(path), "encoding": encoding},
            ) from err
        except Exception as err:
            if isinstance(err, DataQualityError):
                raise
            raise DataQualityError(
                f"Failed to parse CSV file {path}: {err}",
                context={"file_path": str(path), "error": str(err)},
            ) from err

    def parse_string(self, content: str | TextIO) -> pd.DataFrame:
        """Parse in-memory CSV text content or stream into a DataFrame."""
        if isinstance(content, str):
            if not content.strip():
                raise DataQualityError("Raw CSV content string is empty")
            stream = StringIO(content)
        else:
            stream = content  # type: ignore[assignment]

        try:
            df = pd.read_csv(
                stream,
                na_values=self.na_values,
                keep_default_na=True,
                dtype=object,
            )
            if df.empty:
                raise DataQualityError("Parsed in-memory CSV contains 0 data rows")
            return df
        except Exception as err:
            if isinstance(err, DataQualityError):
                raise
            raise DataQualityError(
                f"Failed to parse in-memory CSV: {err}",
                context={"error": str(err)},
            ) from err
