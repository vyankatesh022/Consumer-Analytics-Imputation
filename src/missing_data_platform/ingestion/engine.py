"""Ingestion Orchestration Engine.

Coordinates reading, structural validation, quarantine routing, lineage tracking,
and canonical bronze data generation without modifying underlying missing values.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import pandas as pd

from missing_data_platform.ingestion.contract import RawDataContract
from missing_data_platform.ingestion.lineage import IngestionLineage
from missing_data_platform.ingestion.parser import CsvParser
from missing_data_platform.ingestion.validator import ExtraColumnsAction, SchemaValidator
from missing_data_platform.logging import get_logger

logger = get_logger("ingestion.engine")


@dataclass
class IngestionResult:
    """Canonical outcome of an ingestion operation."""

    valid_data: pd.DataFrame
    quarantined_data: pd.DataFrame
    lineage: IngestionLineage
    contract: RawDataContract

    @property
    def is_clean(self) -> bool:
        """True if zero records were quarantined."""
        return len(self.quarantined_data) == 0

    def save_bronze_parquet(
        self,
        valid_output_path: Path | str,
        quarantine_output_path: Path | str | None = None,
    ) -> None:
        """Persist canonical Bronze datasets to Parquet storage."""
        valid_path = Path(valid_output_path)
        valid_path.parent.mkdir(parents=True, exist_ok=True)
        self.valid_data.to_parquet(valid_path, index=False, engine="pyarrow")

        if quarantine_output_path and not self.quarantined_data.empty:
            q_path = Path(quarantine_output_path)
            q_path.parent.mkdir(parents=True, exist_ok=True)
            self.quarantined_data.to_parquet(q_path, index=False, engine="pyarrow")


class IngestionEngine:
    """Production ingestion engine enforcing raw data contracts."""

    def __init__(
        self,
        contract: RawDataContract | None = None,
        extra_columns_action: ExtraColumnsAction = ExtraColumnsAction.PRESERVE,
    ) -> None:
        self.contract = contract or RawDataContract.default_consumer_contract()
        self.parser = CsvParser()
        self.validator = SchemaValidator(
            contract=self.contract,
            extra_columns_action=extra_columns_action,
        )

    def ingest_file(
        self,
        file_path: Path | str,
        dataset_id: str | None = None,
    ) -> IngestionResult:
        """Ingest and validate a raw CSV file from local storage or lakehouse landing.

        Raises:
            DataQualityError: On fatal unrecoverable schema mismatch or corrupt file.
        """
        path = Path(file_path)
        ds_id = dataset_id or f"raw_{path.stem}_{int(time.time())}"
        start_time = time.perf_counter()

        logger.info(
            "Ingestion started",
            dataset_id=ds_id,
            source_file=path.name,
            schema_version=self.contract.version,
        )

        # 1. Compute source digest
        source_hash = IngestionLineage.compute_file_sha256(path)

        # 2. Parse raw file
        raw_df = self.parser.parse_file(path)

        # 3. Validate and separate quarantined records
        val_result = self.validator.validate(raw_df)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 4. Generate Lineage
        lineage = IngestionLineage(
            dataset_id=ds_id,
            source_identifier=str(path),
            source_hash_sha256=source_hash,
            schema_version=self.contract.version,
            total_records=val_result.total_records,
            valid_records=val_result.valid_records_count,
            quarantined_records=val_result.quarantined_records_count,
            missingness_distribution=val_result.missingness_summary,
            execution_time_ms=round(elapsed_ms, 2),
        )

        logger.info(
            "Ingestion completed",
            dataset_id=ds_id,
            total_records=val_result.total_records,
            valid_records=val_result.valid_records_count,
            quarantined_records=val_result.quarantined_records_count,
            execution_time_ms=round(elapsed_ms, 2),
        )

        return IngestionResult(
            valid_data=val_result.valid_df,
            quarantined_data=val_result.quarantined_df,
            lineage=lineage,
            contract=self.contract,
        )

    def ingest_content(
        self,
        content: str | TextIO,
        dataset_id: str = "in_memory_dataset",
        source_identifier: str = "in_memory_stream",
    ) -> IngestionResult:
        """Ingest raw CSV text content or stream in-memory."""
        start_time = time.perf_counter()

        logger.info(
            "In-memory ingestion started",
            dataset_id=dataset_id,
            source=source_identifier,
        )

        raw_df = self.parser.parse_string(content)
        val_result = self.validator.validate(raw_df)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        lineage = IngestionLineage(
            dataset_id=dataset_id,
            source_identifier=source_identifier,
            source_hash_sha256=None,
            schema_version=self.contract.version,
            total_records=val_result.total_records,
            valid_records=val_result.valid_records_count,
            quarantined_records=val_result.quarantined_records_count,
            missingness_distribution=val_result.missingness_summary,
            execution_time_ms=round(elapsed_ms, 2),
        )

        logger.info(
            "In-memory ingestion completed",
            dataset_id=dataset_id,
            total_records=val_result.total_records,
            valid_records=val_result.valid_records_count,
            quarantined_records=val_result.quarantined_records_count,
            execution_time_ms=round(elapsed_ms, 2),
        )

        return IngestionResult(
            valid_data=val_result.valid_df,
            quarantined_data=val_result.quarantined_df,
            lineage=lineage,
            contract=self.contract,
        )
