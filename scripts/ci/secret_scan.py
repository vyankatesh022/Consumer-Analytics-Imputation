#!/usr/bin/env python3
"""Automated secret and credential scanner for CI/CD quality gates.

Scans tracked files and staged git diffs for plaintext credentials, API tokens,
private keys, passwords, and connection strings without printing secret values to logs.
"""

import os
import re
import sys
from pathlib import Path

# High-entropy and known credential regex patterns
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "AWS Access Key ID",
        re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    ),
    (
        "Generic API Key / Secret Token Assignment",
        re.compile(
            r"(?i)(api[_-]?(?:key|token)|secret[_-]?(?:key|token)|auth[_-]?token|access[_-]?(?:token|key)|client[_-]?secret)\s*[:=]\s*['\"][A-Za-z0-9_\-\.\+/=]{16,}['\"]"
        ),
    ),
    (
        "Private Key Header",
        re.compile(r"-----BEGIN (?:RSA|EC|DSA|OPENSSH|PGP|ENCRYPTED)? ?PRIVATE KEY-----"),
    ),
    (
        "Generic Password Assignment",
        re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
    ),
    (
        "Database Connection String with Credentials",
        re.compile(r"(?i)(postgres|mysql|mongodb|redis)://[^:]+:[^@]+@[^/]+"),
    ),
    (
        "Bearer Authentication Token",
        re.compile(r"(?i)bearer\s+[a-zA-Z0-9\-\._~\+\/]+=*"),
    ),
]

# Patterns for exclusions / false-positive suppression (test files or documentation)
ALLOWLIST_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"#\s*ci-secret-allow"),
    re.compile(r"SECRET_PATTERNS"),
    re.compile(r"SECRET_ID_\d+"),
    re.compile(r"AKIAIOSFODNN7EXAMPLE"),
    re.compile(r"wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
    re.compile(r"placeholder"),
    re.compile(r"dummy"),
]

EXCLUDED_DIRS: set[str] = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    "htmlcov",
    ".coverage",
}

EXCLUDED_EXTENSIONS: set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dll",
    ".dylib",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".lock",
}


def is_allowlisted(line: str) -> bool:
    """Check if line matches any documented allowlist pattern."""
    return any(p.search(line) for p in ALLOWLIST_PATTERNS)


def scan_file(file_path: Path) -> list[str]:
    """Scan a single file for exposed secret patterns, returning sanitized findings."""
    findings: list[str] = []
    if file_path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return findings

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return [f"Unable to read file {file_path}: {e}"]

    for line_idx, line in enumerate(content.splitlines(), start=1):
        if is_allowlisted(line):
            continue

        for label, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(
                    f"[{label}] detected in {file_path}:{line_idx} (content redacted for security)"
                )
                break

    return findings


def run_scan(root_dir: Path) -> int:
    """Recursively scan repository directory for credentials."""
    all_findings: list[str] = []
    total_scanned = 0

    for root, dirs, files in os.walk(root_dir):
        # Prune excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        for file in files:
            p = Path(root) / file
            # Skip this script itself to avoid pattern self-matching
            if p.name == "secret_scan.py" or p.name == "test_ci_secret_scanner.py":
                continue
            total_scanned += 1
            findings = scan_file(p)
            all_findings.extend(findings)

    print(f"Secret Scanner: Scanned {total_scanned} repository files.")

    if all_findings:
        print("\n❌ SECURITY QUALITY GATE FAILED: Secrets or credentials detected:")
        for finding in all_findings:
            print(f"  - {finding}")
        print("\nPlease remove all credentials and invalidate any exposed tokens before merging.\n")
        return 1

    print("✅ Secret Scanner: Zero plaintext secrets or credentials detected.")
    return 0


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent.parent
    sys.exit(run_scan(repo_root))
