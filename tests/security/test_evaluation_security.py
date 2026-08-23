"""Security tests for the Evaluation framework: log sanitization and metadata privacy."""

import io
import logging

import pandas as pd

from missing_data_platform.config import LogLevel, Settings
from missing_data_platform.evaluation.engine import ImputationEvaluator
from missing_data_platform.logging import configure_logging
from missing_data_platform.masking.config import MaskingConfig, MaskingStrategy


def test_evaluation_logging_does_not_leak_customer_records() -> None:
    """Verify evaluation logs never emit sensitive customer records, ground truth values, or PII."""
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)

    settings = Settings(LOG_LEVEL=LogLevel.INFO, LOG_FORMAT="json")
    configure_logging(settings)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    df = pd.DataFrame(
        {
            "customer_id": [f"SECRET_CUSTOMER_{i:03d}" for i in range(15)],
            "age": [float(30 + i) for i in range(15)],
            "income": [float(500000 + 10000 * i) for i in range(15)],
            "purchase_next_month": [1] * 15,
        }
    )

    mask_config = MaskingConfig(
        experiment_id="exp_eval_security_audit",
        mask_rate=0.2,
        random_seed=42,
        strategy=MaskingStrategy.UNIFORM_RANDOM,
        target_features=["income"],
    )

    evaluator = ImputationEvaluator()
    report = evaluator.run_benchmark_suite(
        df=df,
        mask_config=mask_config,
        methods=["baseline_median", "knn"],
    )

    assert len(report.method_results) == 2

    handler.flush()
    logs = log_capture.getvalue()

    # Assert no sensitive IDs or raw ground-truth numbers in logs
    assert "SECRET_CUSTOMER" not in logs
    assert "500000" not in logs

    # Ensure operational tracking is present
    assert "exp_eval_security_audit" in logs

    root_logger.removeHandler(handler)
