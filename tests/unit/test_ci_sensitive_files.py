"""Unit tests for sensitive file detection quality gate."""

from pathlib import Path

from scripts.ci.check_sensitive_files import check_sensitive_files


def test_sensitive_files_clean_repository(tmp_path: Path) -> None:
    """Assert clean repository with standard code files passes."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')", encoding="utf-8")
    (tmp_path / ".env.example").write_text("KEY=example", encoding="utf-8")

    status = check_sensitive_files(tmp_path)
    assert status == 0


def test_sensitive_files_detects_prohibited_extensions(tmp_path: Path) -> None:
    """Assert scanner rejects .env and private key files."""
    (tmp_path / ".env").write_text("SECRET=123", encoding="utf-8")
    (tmp_path / "server.key").write_text("dummy key", encoding="utf-8")

    status = check_sensitive_files(tmp_path)
    assert status == 1
