"""Deterministic dataset, configuration, and environment lineage fingerprinting."""

import hashlib
import json
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

from missing_data_platform.ingestion.contract import DataType, RawDataContract


def calculate_dataset_fingerprint(
    df: pd.DataFrame,
    contract: RawDataContract | None = None,
) -> str:
    """Calculate deterministic SHA-256 fingerprint of dataset schema, nullability, and summary statistics.

    Guarantees:
    - Sensitive values, individual records, and PII are never included in the fingerprint computation.
    - Captures schema structural changes, row count modifications, and distribution shifts.
    """
    if df.empty:
        return hashlib.sha256(b"empty_dataframe").hexdigest()

    meta_parts: dict[str, Any] = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": {},
    }

    contract_obj = contract or RawDataContract.default_consumer_contract()

    for col in sorted(df.columns):
        series = df[col]
        col_defn = contract_obj.get_column(col)
        is_numeric = pd.api.types.is_numeric_dtype(series) or (
            col_defn is not None and col_defn.data_type in (DataType.FLOAT, DataType.INTEGER)
        )

        null_count = int(series.isna().sum())

        if is_numeric:
            valid_vals = series.dropna().astype(float)
            if not valid_vals.empty:
                col_meta: dict[str, Any] = {
                    "type": "numeric",
                    "null_count": null_count,
                    "min": round(float(np.min(valid_vals)), 4),
                    "max": round(float(np.max(valid_vals)), 4),
                    "mean": round(float(np.mean(valid_vals)), 4),
                    "std": round(float(np.std(valid_vals)), 4),
                }
            else:
                col_meta = {
                    "type": "numeric",
                    "null_count": null_count,
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None,
                }
        else:
            col_meta = {
                "type": "categorical",
                "null_count": null_count,
                "unique_count": int(series.nunique(dropna=True)),
            }

        meta_parts["columns"][col] = col_meta

    canonical_json = json.dumps(meta_parts, sort_keys=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def calculate_data_hash(data: Any) -> str:
    """Calculate deterministic SHA-256 hash for arbitrary serializable payload or string."""
    if isinstance(data, (dict, list)):
        payload_str = json.dumps(data, sort_keys=True, default=str)
    elif isinstance(data, str):
        payload_str = data
    else:
        payload_str = str(data)

    return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()


def get_environment_info() -> dict[str, Any]:
    """Inspect and resolve execution environment, platform details, and git code version."""
    env_info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "git_commit": "unknown",
        "clean_worktree": True,
    }

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        env_info["git_commit"] = commit

        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        env_info["clean_worktree"] = len(status) == 0
    except Exception:
        env_info["git_commit"] = "unversioned"
        env_info["clean_worktree"] = False

    return env_info
