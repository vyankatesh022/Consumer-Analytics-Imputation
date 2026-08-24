#!/usr/bin/env python3
"""Automated sensitive file gate for CI/CD.

Verifies that untracked or accidentally staged sensitive files (such as .env,
private keys, local databases, or raw data dumps) are not present in the repository.
"""

import fnmatch
import os
import subprocess
import sys
from pathlib import Path

PROHIBITED_PATTERNS: list[str] = [
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.pfx",
    "*.p12",
    "id_rsa",
    "id_rsa.pub",
    "*.sqlite",
    "*.sqlite3",
    "*.db",
    "data/raw/*",
    "data/local/*",
    "data/bronze/*",
    "data/quarantine/*",
]

EXCLUDED_ALLOWED_FILES: set[str] = {
    ".env.example",
}


def get_tracked_and_staged_files(repo_root: Path) -> list[str]:
    """Retrieve all git tracked and staged files."""
    try:
        tracked = subprocess.check_output(
            ["git", "ls-files"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).splitlines()
        return [f.strip() for f in tracked if f.strip()]
    except Exception:
        # Fallback to os.walk if not a git repository in test environments
        tracked_files: list[str] = []
        for root, _, files in os.walk(repo_root):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), repo_root)
                if not rel_path.startswith(".git"):
                    tracked_files.append(rel_path)
        return tracked_files


def check_sensitive_files(repo_root: Path) -> int:
    """Scan tracked repository files against prohibited sensitive file patterns."""
    files = get_tracked_and_staged_files(repo_root)
    violations: list[str] = []

    for file_path_str in files:
        file_name = os.path.basename(file_path_str)
        if file_name in EXCLUDED_ALLOWED_FILES or file_path_str in EXCLUDED_ALLOWED_FILES:
            continue

        for pattern in PROHIBITED_PATTERNS:
            if fnmatch.fnmatch(file_name, pattern) or fnmatch.fnmatch(file_path_str, pattern):
                violations.append(f"Prohibited sensitive file pattern '{pattern}': {file_path_str}")
                break

    if violations:
        print("\n❌ SENSITIVE FILE QUALITY GATE FAILED:")
        for violation in violations:
            print(f"  - {violation}")
        print("\nPlease remove sensitive files and update .gitignore.\n")
        return 1

    print(f"✅ Sensitive File Gate: All {len(files)} tracked files verified clean.")
    return 0


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent.parent
    sys.exit(check_sensitive_files(repo_root))
