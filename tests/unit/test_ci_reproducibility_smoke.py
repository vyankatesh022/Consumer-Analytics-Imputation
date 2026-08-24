"""Unit tests for CI reproducibility smoke test harness."""

from scripts.ci.reproducibility_smoke_test import (
    generate_synthetic_smoke_fixture,
    run_reproducibility_smoke_test,
)


def test_synthetic_smoke_fixture_validity() -> None:
    """Assert generated synthetic smoke fixture meets minimum schema expectations."""
    df = generate_synthetic_smoke_fixture(n_records=30)
    assert len(df) == 30
    assert "customer_id" in df.columns
    assert "purchase_next_month" in df.columns
    assert "income" in df.columns


def test_reproducibility_smoke_test_execution() -> None:
    """Assert that the reproducibility smoke test completes with exit code 0."""
    exit_code = run_reproducibility_smoke_test()
    assert exit_code == 0
