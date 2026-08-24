.PHONY: help install test test-cov lint format format-check type-check check security-check smoke-test ci clean

help:
	@echo "AI/ML Missing Data Imputation & Bias Reduction Platform"
	@echo "========================================================="
	@echo "make install        - Install Python package and development dependencies"
	@echo "make test           - Run test suite with pytest"
	@echo "make test-cov       - Run test suite with coverage report"
	@echo "make lint           - Run ruff linter"
	@echo "make format         - Format codebase with ruff"
	@echo "make format-check   - Check formatting without modifying files"
	@echo "make type-check     - Run static type checking with mypy"
	@echo "make security-check - Run secret scanner and sensitive file checks"
	@echo "make smoke-test     - Run deterministic reproducibility smoke test"
	@echo "make check          - Run lint, format check, type check, and tests"
	@echo "make ci             - Run complete local CI quality gate pipeline"
	@echo "make clean          - Remove cached files and build artifacts"

install:
	python3 -m pip install --upgrade pip
	python3 -m pip install -e ".[dev]"

test:
	PYTHONPATH=src pytest tests/

test-cov:
	PYTHONPATH=src pytest --cov=src/missing_data_platform --cov-report=term-missing --cov-fail-under=80 tests/

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

format-check:
	ruff format --check src/ tests/ scripts/

type-check:
	PYTHONPATH=src mypy src/ tests/ scripts/

security-check:
	python3 scripts/ci/secret_scan.py
	python3 scripts/ci/check_sensitive_files.py

smoke-test:
	PYTHONPATH=src python3 scripts/ci/reproducibility_smoke_test.py

check: lint format-check type-check test

ci: format-check lint type-check security-check test-cov smoke-test
	@echo "=========================================="
	@echo "✅ All Local CI Quality Gates Passed!"
	@echo "=========================================="

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build dist *.egg-info htmlcov .coverage

