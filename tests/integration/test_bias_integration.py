"""End-to-End Integration tests connecting Ingestion -> Quality -> Missingness -> Masking -> Imputers -> Evaluation -> Bias Analysis."""

from pathlib import Path

from missing_data_platform.bias.config import GroupDefinitionConfig
from missing_data_platform.bias.engine import BiasAnalysisEngine
from missing_data_platform.imputation.engine import BaselineImputationEngine
from missing_data_platform.ingestion.engine import IngestionEngine
from missing_data_platform.masking.config import MaskingConfig, MaskingStrategy
from missing_data_platform.masking.engine import MaskingEngine
from missing_data_platform.missingness.engine import MissingnessAnalysisEngine
from missing_data_platform.quality.engine import DataQualityEngine


def test_full_pipeline_to_bias_analysis(sample_csv_path: Path) -> None:
    """Verify end-to-end execution through to bias and representation disparity analysis."""
    # 1. Ingestion
    ingestion = IngestionEngine()
    ingest_result = ingestion.ingest_file(sample_csv_path, dataset_id="bias_pipeline_ingest")
    assert ingest_result.is_clean is True

    # 2. Quality Audit
    quality = DataQualityEngine()
    q_report = quality.audit_dataset(ingest_result.valid_data, dataset_id="bias_pipeline_quality")
    assert q_report.failed_checks == 0

    # 3. Missingness Diagnostics
    missingness = MissingnessAnalysisEngine()
    m_report = missingness.analyze(ingest_result.valid_data, dataset_id="bias_pipeline_missingness")
    assert m_report.total_records == 10

    # 4. Artificial Masking
    mask_config = MaskingConfig(
        experiment_id="bias_pipeline_mask",
        mask_rate=0.20,
        random_seed=42,
        strategy=MaskingStrategy.UNIFORM_RANDOM,
        target_features=["average_purchase_value", "total_spend"],
    )
    masking = MaskingEngine()
    mask_res = masking.generate_benchmark_dataset(ingest_result.valid_data, mask_config)

    # 5. Imputation
    imp_engine = BaselineImputationEngine()
    imp_knn = imp_engine.impute_knn_dataset(mask_res.masked_dataset)
    imp_rf = imp_engine.impute_rf_dataset(mask_res.masked_dataset)

    # 6. Bias & Representation Analysis
    bias_config = GroupDefinitionConfig(group_column="customer_segment", minimum_group_size=2)
    bias_engine = BiasAnalysisEngine(config=bias_config)

    bias_report = bias_engine.run_bias_analysis(
        df=ingest_result.valid_data,
        imputed_results={"knn": imp_knn.imputed_dataset, "rf": imp_rf.imputed_dataset},
        ground_truth_store=mask_res.ground_truth_store,
        experiment_id="bias_e2e_run",
    )

    assert bias_report.experiment_id == "bias_e2e_run"
    assert len(bias_report.representation_results) > 0
    assert len(bias_report.missingness_results) > 0
    assert len(bias_report.performance_results) > 0
    assert isinstance(bias_report.to_representation_dataframe(), type(ingest_result.valid_data))
