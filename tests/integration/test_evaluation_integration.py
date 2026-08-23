"""End-to-End Integration tests connecting Ingestion -> Quality -> Missingness -> Masking -> Imputers -> Evaluation."""

from pathlib import Path

from missing_data_platform.evaluation.engine import ImputationEvaluator
from missing_data_platform.ingestion.engine import IngestionEngine
from missing_data_platform.masking.config import MaskingConfig, MaskingStrategy
from missing_data_platform.missingness.engine import MissingnessAnalysisEngine
from missing_data_platform.quality.engine import DataQualityEngine


def test_full_pipeline_to_benchmark_evaluation(sample_csv_path: Path) -> None:
    """Verify end-to-end flow from raw CSV ingestion through to comparative evaluation."""
    # 1. Ingestion
    ingestion = IngestionEngine()
    ingest_result = ingestion.ingest_file(sample_csv_path, dataset_id="eval_pipeline_ingest")
    assert ingest_result.is_clean is True

    # 2. Quality Audit
    quality = DataQualityEngine()
    q_report = quality.audit_dataset(ingest_result.valid_data, dataset_id="eval_pipeline_quality")
    assert q_report.failed_checks == 0

    # 3. Missingness Diagnostics
    missingness = MissingnessAnalysisEngine()
    m_report = missingness.analyze(ingest_result.valid_data, dataset_id="eval_pipeline_missingness")
    assert m_report.total_records == 10

    # 4. Artificial Masking
    mask_config = MaskingConfig(
        experiment_id="eval_bench_e2e",
        mask_rate=0.20,
        random_seed=42,
        strategy=MaskingStrategy.UNIFORM_RANDOM,
        target_features=["average_purchase_value", "total_spend"],
    )

    # 5. Automated Multi-Algorithm Benchmarking and Evaluation
    evaluator = ImputationEvaluator()
    comparison_report = evaluator.run_benchmark_suite(
        df=ingest_result.valid_data,
        mask_config=mask_config,
        methods=["baseline_median", "baseline_mean", "knn", "iterative", "random_forest"],
    )

    assert len(comparison_report.method_results) == 5
    assert len(comparison_report.method_rankings) == 5

    # Verify summary table contains valid metric entries
    summary_df = comparison_report.to_summary_dataframe()
    assert len(summary_df) == 5
    assert summary_df["Weighted MAE"].isna().sum() == 0
    assert summary_df["Weighted RMSE"].isna().sum() == 0

    # Best method has rank 1
    assert summary_df.loc[0, "Rank (MAE)"] == 1
