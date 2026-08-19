"""Security tests verifying secret exclusion, .gitignore rules, and sanitized templates."""

import re
from pathlib import Path


def test_env_example_contains_no_real_secrets() -> None:
    """Verify that .env.example contains only placeholders and zero real secret patterns."""
    repo_root = Path(__file__).parent.parent.parent
    env_example_path = repo_root / ".env.example"

    assert env_example_path.exists(), ".env.example must exist in repository root"
    content = env_example_path.read_text(encoding="utf-8")

    # Reject standard realistic secret patterns
    real_aws_key_pattern = r"(AKIA[0-9A-Z]{16})"
    real_jwt_pattern = r"(eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,})"
    private_key_header = r"-----BEGIN (RSA|EC|OPENSSH|PRIVATE) KEY-----"

    assert not re.search(real_aws_key_pattern, content), (
        "Found realistic AWS Key ID in .env.example!"
    )
    assert not re.search(real_jwt_pattern, content), "Found realistic JWT token in .env.example!"
    assert not re.search(private_key_header, content), "Found private key header in .env.example!"


def test_gitignore_covers_critical_sensitive_patterns() -> None:
    """Verify that .gitignore contains rules protecting environment files, keys, and data."""
    repo_root = Path(__file__).parent.parent.parent
    gitignore_path = repo_root / ".gitignore"

    assert gitignore_path.exists(), ".gitignore must exist in repository root"
    content = gitignore_path.read_text(encoding="utf-8")
    lines = [
        line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")
    ]

    critical_patterns = [
        ".gitignore",
        ".env",
        ".env.*",
        ".env.example",
        "docs/",
        "CONTRIBUTING.md",
        "ENGINEERING_RULES.md",
        "*.log",
        "*.db",
        "*.sqlite",
        "data/",
        "models/*.pkl",
        "artifacts/",
        "credentials/",
        "secrets/",
        "*.pem",
        "*.key",
    ]

    for pattern in critical_patterns:
        assert pattern in lines or any(p in lines for p in [pattern, pattern + "/"]), (
            f"Missing critical gitignore pattern: {pattern}"
        )
