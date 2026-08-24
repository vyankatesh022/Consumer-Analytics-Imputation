"""Unit tests verifying GitHub Actions workflow syntax, permissions, and quality gate structure."""

from pathlib import Path

import yaml


def test_ci_workflow_yaml_structure() -> None:
    """Assert CI workflow YAML is valid, secure, and contains all required quality gate jobs."""
    workflow_path = (
        Path(__file__).resolve().parent.parent.parent / ".github" / "workflows" / "ci.yml"
    )
    assert workflow_path.exists(), "CI workflow file .github/workflows/ci.yml must exist."

    content = workflow_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(content)

    # 1. Assert minimal read-only permissions
    assert parsed.get("permissions") == "read-all"

    # 2. Assert triggers
    triggers = parsed.get("on") or parsed.get(True, {})
    assert "pull_request" in triggers
    assert "push" in triggers

    # 3. Assert all required quality gate jobs exist
    jobs = parsed.get("jobs", {})
    required_jobs = {
        "code-quality",
        "security-and-secrets",
        "test-suite",
        "reproducibility-smoke-test",
        "quality-gate",
    }
    assert required_jobs.issubset(set(jobs.keys()))

    # 4. Assert timeouts configured for jobs
    for job_name in required_jobs:
        job = jobs[job_name]
        if job_name != "quality-gate":
            assert "timeout-minutes" in job
            assert job["timeout-minutes"] <= 30

    # 5. Assert quality-gate requires all other validation jobs
    qg_needs = set(jobs["quality-gate"].get("needs", []))
    assert qg_needs == {
        "code-quality",
        "security-and-secrets",
        "test-suite",
        "reproducibility-smoke-test",
    }
