"""Unit tests for CI secret scanner functionality and exclusion rules."""

from pathlib import Path

from scripts.ci.secret_scan import is_allowlisted, scan_file


def test_secret_scanner_detects_synthetic_secret(tmp_path: Path) -> None:
    """Assert scanner detects exposed AWS keys and generic API key patterns."""
    secret_file = tmp_path / "leaked_keys.py"
    # Construct synthetic secret pattern
    secret_file.write_text(
        "AWS_KEY = 'AKIAIOSFODNN7FAKEKEY'\nAPI_TOKEN = 'secret_key_1234567890abcdef12345'\n",
        encoding="utf-8",
    )

    findings = scan_file(secret_file)
    assert len(findings) >= 1
    # Ensure raw secret content is not dumped in findings
    for f in findings:
        assert "redacted" in f
        assert "1234567890abcdef" not in f


def test_secret_scanner_passes_clean_file(tmp_path: Path) -> None:
    """Assert scanner returns zero findings for standard clean code."""
    clean_file = tmp_path / "clean_code.py"
    clean_file.write_text(
        "import os\n\ndef get_value(x: int) -> int:\n    return x * 2\n",
        encoding="utf-8",
    )

    findings = scan_file(clean_file)
    assert len(findings) == 0


def test_secret_scanner_allowlist_suppression() -> None:
    """Assert lines marked with allowlist tags or dummy patterns are ignored."""
    assert is_allowlisted("token = 'dummy_token'  # ci-secret-allow") is True
    assert is_allowlisted("cust_id = 'SECRET_ID_0001'") is True
    assert is_allowlisted("SECRET_PATTERNS = []") is True
    assert is_allowlisted("clean_variable = 123") is False
