# AI/ML-Based Missing Data Imputation & Bias Reduction Platform

An enterprise-grade, scalable platform engineered to handle missing data in multi-million record consumer datasets, audit representation stability across customer segments, and benchmark downstream predictive machine learning performance.

---

## Current Status
> **SESSION 2 — REPOSITORY FOUNDATION & DEVELOPMENT ENVIRONMENT**
>
> *Note: This repository is currently in Session 2. Core repository layout, typed configuration management, structured logging foundation, code-quality tooling (Ruff, Mypy), security checks, and testing harnesses are established. Data ingestion, PySpark pipelines, imputation engines, ML models, and API endpoints will be implemented in subsequent sessions.*

---

## Architectural Overview

```
Raw Consumer Data (S3 Raw)
          ↓
Ingestion & Schema Quarantine (S3 Bronze)
          ↓
Data Quality & Missingness Diagnostics (S3 Silver)
          ↓
Modular Imputation Strategies (Median / KNN / MICE / MissForest)
          ↓
Dual Evaluation Framework:
  ├── 1. Numerical & Categorical Reconstruction Fidelity (RMSE, MAE, R², Macro-F1)
  └── 2. Demographic Representation Stability (PSI, TVD, Subgroup Disparity)
          ↓
Downstream ML Predictor (Target: purchase_next_month)
          ↓
Multi-Criteria Champion Model Selection & MLflow Registry
          ↓
FastAPI Low-Latency Inference & Observability
```

---

## Repository Structure

```text
.
├── .env.example              # Sanitized environment variable template
├── .gitignore                # Comprehensive production-grade exclusion rules
├── Makefile                  # Developer CLI commands (test, lint, format, type-check)
├── pyproject.toml            # Build system, dependencies, and tooling configuration
├── README.md                 # Project overview and quickstart guide
├── ENGINEERING_RULES.md      # Standards, clean architecture, typing, error handling rules
├── CONTRIBUTING.md           # Git workflow, branch naming, Conventional Commits guide
├── config/                   # Base, development, and production YAML configurations
├── docs/                     # Architecture, deployment, and developer documentation
├── infrastructure/           # Dockerfiles, docker-compose, and Terraform IaC
├── notebooks/                # Exploratory prototyping notebooks
├── pipelines/                # Batch automation pipelines (ingestion, imputation, ML)
├── scripts/                  # Developer utilities and dataset generators
├── src/
│   └── missing_data_platform/
│       ├── __init__.py       # Package exports
│       ├── __version__.py    # Version metadata
│       ├── config.py         # Type-safe Pydantic Settings
│       ├── exceptions.py     # Custom exception hierarchy
│       └── logging.py        # Structured JSON logging with structlog
└── tests/
    ├── conftest.py           # Pytest fixtures and test environment cleanup
    ├── unit/                 # Unit tests for core algorithms and utilities
    ├── integration/          # Multi-component integration tests
    ├── security/             # Secret exclusion and .gitignore audit tests
    └── e2e/                  # End-to-end smoke and workflow tests
```

---

## Development Setup

### 1. Prerequisites
* Python 3.10+ (Python 3.11+ recommended)
* `git`
* `make` (optional, for CLI shortcuts)

### 2. Environment Initialization
```bash
# Clone the repository
git clone <repo-url>
cd media

# Create and activate a virtual environment (optional if using global environment)
python3 -m venv .venv
source .venv/bin/activate

# Install the package in editable mode with development dependencies
make install
# Alternatively: pip install -e ".[dev]"
```

### 3. Environment Variable Configuration
```bash
# Copy template to local .env
cp .env.example .env

# Edit .env to configure local paths or settings
```

### 4. Running Quality Checks & Tests
```bash
# Run linting
make lint

# Run format check
make format-check

# Run static type checking
make type-check

# Run full test suite
make test

# Run all checks together
make check
```
