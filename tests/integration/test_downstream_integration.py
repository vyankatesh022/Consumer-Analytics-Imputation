"""End-to-end integration test validating the entire platform pipeline."""

from pathlib import Path

from missing_data_platform.bias.engine import BiasAnalysisEngine
from missing_data_platform.downstream.config import DownstreamBenchmarkConfig, DownstreamConfig
from missing_data_platform.downstream.engine import DownstreamEvaluationEngine
from missing_data_platform.evaluation.engine import ImputationEvaluator
from missing_data_platform.imputation.engine import BaselineImputationEngine
from missing_data_platform.ingestion.parser import CsvParser
from missing_data_platform.ingestion.validator import SchemaValidator
from missing_data_platform.masking.config import MaskingConfig
from missing_data_platform.masking.engine import MaskingEngine
from missing_data_platform.mitigation.config import MitigationConfig
from missing_data_platform.mitigation.engine import FairnessMitigationEngine
from missing_data_platform.quality.engine import DataQualityEngine


def test_full_platform_pipeline_integration(tmp_path: Path) -> None:
    """Execute complete 10-stage end-to-end validation pipeline."""
    # 1. Generate & Ingest dataset
    records = []
    for i in range(150):
        records.append(
            f"CUST_{i:04d},{20.0 + (i % 45)},{'Female' if i % 2 == 0 else 'Male'},{35000.0 + (i * 600.0)},"
            f"{'Bachelor' if i % 3 == 0 else 'Master'},Engineer,Seattle,West,{2.0 + (i % 6)},{60.0 + i},"
            f"{300.0 + (i * 10)},0.15,{8 + (i % 10)},2,Electronics,{'Gold' if i % 2 == 0 else 'Silver'},"
            f"{1 if i % 3 == 0 else 0}"
        )
    csv_text = (
        "customer_id,age,gender,income,education,occupation,city,region,purchase_frequency,"
        "average_purchase_value,total_spend,discount_usage,website_visits,campaign_exposure,"
        "product_category,customer_segment,purchase_next_month\n" + "\n".join(records)
    )
    csv_file = tmp_path / "integration_consumer_data.csv"
    csv_file.write_text(csv_text, encoding="utf-8")

    # Stage 1: Ingestion
    parser = CsvParser()
    raw_df = parser.parse_file(str(csv_file))
    validator = SchemaValidator()
    val_res = validator.validate(raw_df)
    assert val_res.is_valid

    # Stage 2: Quality Validation
    quality_engine = DataQualityEngine()
    qual_report = quality_engine.audit_dataset(val_res.valid_df)
    assert not qual_report.has_failures

    df = val_res.valid_df

    # Stage 3: Masking
    mask_cfg = MaskingConfig(experiment_id="integ_mask", mask_rate=0.15, random_seed=42)
    masking_engine = MaskingEngine()
    mask_res = masking_engine.generate_benchmark_dataset(df, mask_cfg)
    assert mask_res.ground_truth_store.total_masked_cells > 0

    # Stage 4: Imputation
    imputation_engine = BaselineImputationEngine()
    imp_res = imputation_engine.impute_rf_dataset(mask_res.masked_dataset, random_seed=42)
    assert imp_res.total_cells_imputed > 0

    # Stage 5: Mitigation
    mit_cfg = MitigationConfig(enabled=True, group_column="customer_segment")
    mit_engine = FairnessMitigationEngine(config=mit_cfg)
    mit_df = mit_engine.impute_with_mitigation(mask_res.masked_dataset)
    assert not mit_df.empty

    # Stage 6: Imputation Evaluation
    evaluator = ImputationEvaluator()
    eval_res = evaluator.evaluate_method(
        imputed_df=imp_res.imputed_dataset,
        ground_truth_store=mask_res.ground_truth_store,
        method_name="random_forest",
    )
    assert eval_res.weighted_mae is not None

    # Stage 7: Bias Analysis
    bias_engine = BiasAnalysisEngine()
    bias_res = bias_engine.analyze_representation(df, mask_res.ground_truth_store)
    assert len(bias_res) > 0

    # Stage 8: Downstream ML Evaluation
    downstream_cfg = DownstreamConfig(random_seed=42)
    downstream_engine = DownstreamEvaluationEngine(config=downstream_cfg)
    bench_cfg = DownstreamBenchmarkConfig(
        experiment_id="integ_downstream_bench",
        methods=["baseline_median", "knn", "random_forest"],
        include_mitigation=True,
    )
    downstream_report = downstream_engine.run_benchmark_suite(
        df,
        mask_config=mask_cfg,
        benchmark_config=bench_cfg,
    )

    assert downstream_report.complete_baseline is not None
    assert "random_forest" in downstream_report.method_results
    assert "fairness_weighted_rf" in downstream_report.mitigated_results
    assert len(downstream_report.comparison_table) >= 4
