"""Missingness Analysis & Mechanism Diagnostics Orchestrator.

Coordinates feature, row, pattern, group disparity, and statistical mechanism
diagnostics into a unified, reproducible MissingnessAnalysisReport.
"""

import pandas as pd

from missing_data_platform.exceptions import DataQualityError
from missing_data_platform.ingestion.contract import DataType, RawDataContract
from missing_data_platform.logging import get_logger
from missing_data_platform.missingness.diagnostics import (
    generate_mnar_limitation_statement,
    run_mar_association_tests,
    run_mcar_diagnostics,
)
from missing_data_platform.missingness.group_analysis import (
    analyze_feature_missingness_by_group,
)
from missing_data_platform.missingness.profiler import (
    profile_feature_missingness,
    profile_missingness_patterns,
    profile_row_missingness,
)
from missing_data_platform.missingness.report import MissingnessAnalysisReport

logger = get_logger("missingness.engine")


class MissingnessAnalysisEngine:
    """Orchestrates comprehensive, evidence-based missingness analysis and mechanism diagnostics."""

    def __init__(
        self,
        contract: RawDataContract | None = None,
        alpha: float = 0.05,
    ) -> None:
        self.contract = contract or RawDataContract.default_consumer_contract()
        self.alpha = alpha

    def analyze(
        self,
        df: pd.DataFrame,
        dataset_id: str = "missingness_study",
        group_columns: list[str] | None = None,
    ) -> MissingnessAnalysisReport:
        """Execute complete missingness profiling, group analysis, and statistical mechanism diagnostics.

        Raises:
            DataQualityError: If input DataFrame is empty.
        """
        if df.empty:
            raise DataQualityError(
                "Cannot perform missingness analysis on an empty DataFrame.",
                context={"dataset_id": dataset_id},
            )

        logger.info(
            "Starting missingness analysis and diagnostics",
            dataset_id=dataset_id,
            total_records=len(df),
            total_columns=len(df.columns),
        )

        # 1. Feature-level Profiling
        feature_profiles = profile_feature_missingness(df)
        missing_features = [p.column_name for p in feature_profiles if p.missing_count > 0]

        # 2. Row-level Profiling
        row_profile = profile_row_missingness(df)

        # 3. Combinatorial Missingness Patterns
        top_patterns = profile_missingness_patterns(df, max_patterns=15)

        # 4. Group-level Disparity Analysis
        default_groups = ["age", "gender", "region", "customer_segment"]
        active_groups = group_columns or [g for g in default_groups if g in df.columns]

        group_disparities = []
        for target_feat in missing_features:
            for grp_col in active_groups:
                if grp_col != target_feat:
                    disp = analyze_feature_missingness_by_group(
                        df=df,
                        target_feature=target_feat,
                        grouping_column=grp_col,
                    )
                    if disp.groups:
                        group_disparities.append(disp)

        # 5. Statistical Diagnostics
        # Continuous auxiliary features for MCAR t-tests
        continuous_covariates = [
            col
            for col, defn in self.contract.columns.items()
            if col in df.columns
            and defn.data_type in (DataType.FLOAT, DataType.INTEGER)
            and not defn.is_identifier
            and not defn.is_target
        ]

        # Categorical covariates for MAR Chi-Square tests
        categorical_covariates = [
            col
            for col, defn in self.contract.columns.items()
            if col in df.columns and defn.data_type == DataType.STRING and not defn.is_identifier
        ]

        mcar_report = run_mcar_diagnostics(
            df=df,
            target_missing_features=missing_features,
            auxiliary_continuous_features=continuous_covariates,
            alpha=self.alpha,
        )

        mar_report = run_mar_association_tests(
            df=df,
            target_missing_features=missing_features,
            categorical_covariates=categorical_covariates,
            alpha=self.alpha,
        )

        # 6. Formal MNAR Limitation Statement
        mnar_statement = generate_mnar_limitation_statement()

        # 7. Executive Synthesis
        if len(missing_features) == 0:
            exec_interp = "The dataset contains 0 missing values across all observed variables."
        elif (
            mcar_report.significant_tests_count > 0 or mar_report.significant_associations_count > 0
        ):
            exec_interp = (
                f"Missingness is present in {len(missing_features)} features. Bivariate diagnostics indicate "
                f"significant associations between missing indicators and observed demographic/behavioral variables "
                f"(inconsistent with pure MCAR). Modeling under MAR assumptions (e.g., MICE, MissForest) is recommended."
            )
        else:
            exec_interp = (
                f"Missingness observed in {len(missing_features)} features exhibits no significant statistical "
                f"dependence with the observed covariates tested, compatible with MCAR assumptions."
            )

        report = MissingnessAnalysisReport(
            dataset_id=dataset_id,
            total_records=len(df),
            total_features=len(df.columns),
            features_with_missingness_count=len(missing_features),
            feature_profiles=feature_profiles,
            row_profile=row_profile,
            top_patterns=top_patterns,
            group_disparities=group_disparities,
            mcar_diagnostics=mcar_report,
            mar_diagnostics=mar_report,
            mnar_limitation_statement=mnar_statement,
            executive_statistical_interpretation=exec_interp,
        )

        logger.info(
            "Missingness analysis completed",
            dataset_id=dataset_id,
            missing_features_count=len(missing_features),
            mcar_sig_tests=mcar_report.significant_tests_count,
            mar_sig_tests=mar_report.significant_associations_count,
        )

        return report
