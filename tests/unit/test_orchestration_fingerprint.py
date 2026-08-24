"""Unit tests for deterministic dataset, configuration, and environment fingerprinting."""

import pandas as pd

from missing_data_platform.orchestration.fingerprint import (
    calculate_dataset_fingerprint,
    get_environment_info,
)


def test_dataset_fingerprint_determinism() -> None:
    """Assert that identical dataframes yield bit-exact SHA-256 fingerprints."""
    df1 = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "age": [25.0, 30.0, 35.0],
            "gender": ["Female", "Male", "Female"],
            "income": [50000.0, 60000.0, 70000.0],
            "purchase_next_month": [1, 0, 1],
        }
    )
    df2 = df1.copy()

    fp1 = calculate_dataset_fingerprint(df1)
    fp2 = calculate_dataset_fingerprint(df2)

    assert fp1 == fp2
    assert len(fp1) == 64


def test_dataset_fingerprint_sensitivity_to_modifications() -> None:
    """Assert fingerprint changes when rows, values, or nullability shift."""
    df_base = pd.DataFrame(
        {
            "age": [25.0, 30.0, 35.0],
            "income": [50000.0, 60000.0, 70000.0],
        }
    )
    fp_base = calculate_dataset_fingerprint(df_base)

    # Modify a value
    df_val_mod = df_base.copy()
    df_val_mod.loc[0, "age"] = 99.0
    assert calculate_dataset_fingerprint(df_val_mod) != fp_base

    # Add a null
    df_null_mod = df_base.copy()
    df_null_mod.loc[0, "age"] = None
    assert calculate_dataset_fingerprint(df_null_mod) != fp_base


def test_environment_info_resolution() -> None:
    """Assert environment metadata is resolved without errors."""
    env = get_environment_info()
    assert "python_version" in env
    assert "platform_system" in env
    assert "git_commit" in env
    assert "clean_worktree" in env
