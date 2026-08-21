"""Integration tests connecting Ingestion -> Quality -> Missingness -> Masking -> Baseline Imputation."""

from pathlib import Path

from missing_data_platform.imputation.config import BaselineStrategy
from missing_data_platform.imputation.engine import BaselineImputationEngine
from missing_data_platform.ingestion.engine import IngestionEngine
from missing_data_platform.masking.config import MaskingConfig, MaskingStrategy
from missing_data_platform.masking.engine import MaskingEngine
from missing_data_platform.missingness.engine import MissingnessAnalysisEngine
from missing_data_platform.quality.engine import DataQualityEngine


def test_full_pipeline_to_baseline_imputation(sample_csv_path: Path) -> None:
    """Verify end-to-end flow from raw CSV ingestion through to baseline imputation."""
    # 1. Ingestion
    ingestion = IngestionEngine()
    ingest_result = ingestion.ingest_file(sample_csv_path, dataset_id="full_ingest")
    assert ingest_result.is_clean is True

    # 2. Quality Audit
    quality = DataQualityEngine()
    q_report = quality.audit_dataset(ingest_result.valid_data, dataset_id="full_quality")
    assert q_report.failed_checks == 0

    # 3. Missingness Diagnostics
    missingness = MissingnessAnalysisEngine()
    m_report = missingness.analyze(ingest_result.valid_data, dataset_id="full_missingness")
    assert m_report.total_records == 10

    # 4. Benchmark Masking
    masking = MaskingEngine()
    mask_config = MaskingConfig(
        experiment_id="exp_full_benchmark",
        mask_rate=0.20,
        random_seed=42,
        strategy=MaskingStrategy.UNIFORM_RANDOM,
        target_features=["average_purchase_value", "total_spend"],
    )
    bench_result = masking.generate_benchmark_dataset(ingest_result.valid_data, mask_config)

    # 5. Baseline Imputation
    imputation_engine = BaselineImputationEngine()
    impute_result = imputation_engine.impute_dataset(
        bench_result.masked_dataset,
        experiment_id="baseline_imputation_run_01",
        numeric_strategy=BaselineStrategy.MEDIAN,
        categorical_strategy=BaselineStrategy.MODE,
    )

    assert impute_result.total_records == 10
    assert impute_result.total_cells_imputed > 0

    # Assert eligible features now have 0 missing values
    assert impute_result.imputed_dataset["age"].isna().sum() == 0
    assert impute_result.imputed_dataset["income"].isna().sum() == 0
    assert impute_result.imputed_dataset["average_purchase_value"].isna().sum() == 0
    assert impute_result.imputed_dataset["total_spend"].isna().sum() == 0

    # Assert protected identifier and target remain protected
    assert impute_result.imputed_dataset["customer_id"].isna().sum() == 0
    assert impute_result.imputed_dataset["purchase_next_month"].isna().sum() == 0
