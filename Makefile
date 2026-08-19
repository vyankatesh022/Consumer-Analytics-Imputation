.PHONY: help install test lint format type-check check clean

help:
	@echo "AI/ML Missing Data Imputation & Bias Reduction Platform"
	@echo "========================================================="
	@echo "make install     - Install Python package and development dependencies"
	@echo "make test        - Run test suite with pytest"
	@echo "make test-cov    - Run test suite with coverage report"
	@echo "make lint        - Run ruff linter"
	@echo "make format      - Format codebase with ruff"
	@echo "make type-check  - Run static type checking with mypy"
	@echo "make check       - Run lint, format check, type check, and tests"
	@echo "make clean       - Remove cached files and build artifacts"

install:
	python3 -m pip install --upgrade pip
	python3 -m pip install -e ".[dev]"

test:
	pytest tests/

test-cov:
	pytest --cov=src/missing_data_platform --cov-report=term-missing tests/

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

format-check:
	ruff format --check src/ tests/

type-check:
	mypy src/ tests/

check: lint format-check type-check test

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build dist *.egg-info htmlcov .coverage
