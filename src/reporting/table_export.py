"""Exports manuscript (curated, publication-facing) and supplementary
(complete, raw) table workbooks via openpyxl (through pandas.ExcelWriter).
"""
from pathlib import Path

import pandas as pd


def export_manuscript_tables(summary, out_path: Path) -> None:
    """Small, curated tables suitable for direct inclusion in a manuscript."""
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary.cohort_metrics[["Cohort", "Accuracy", "ROC_AUC", "Sensitivity", "Specificity"]].to_excel(
            writer, sheet_name="Table1_ExternalValidation", index=False)
        summary.shap_global_importance.head(10)[["rank", "gene_name", "mean_abs_shap"]].to_excel(
            writer, sheet_name="Table2_TopBiomarkers", index=False)
        summary.hazard_ratios.head(10)[["gene_name", "hazard_ratio", "p_value"]].to_excel(
            writer, sheet_name="Table3_HazardRatios", index=False)
        summary.therapeutic_priority.head(10)[["gene_name", "target_priority_score"]].to_excel(
            writer, sheet_name="Table4_DrugTargets", index=False)


def export_supplementary_tables(summary, out_path: Path) -> None:
    """Complete, raw per-module tables for reviewer/reproducibility purposes."""
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary.feature_importance.to_excel(writer, sheet_name="RF_Feature_Importance", index=False)
        summary.cohort_metrics.to_excel(writer, sheet_name="Cohort_Metrics", index=False)
        summary.shap_global_importance.to_excel(writer, sheet_name="SHAP_Global_Importance", index=False)
        summary.biomarker_annotation.to_excel(writer, sheet_name="Biomarker_Annotation", index=False)
        summary.hazard_ratios.to_excel(writer, sheet_name="Survival_Hazard_Ratios", index=False)
        summary.km_statistics.to_excel(writer, sheet_name="Survival_KM_Statistics", index=False)
        summary.therapeutic_priority.to_excel(writer, sheet_name="Drug_Target_Priority", index=False)
        summary.drug_repurposing.to_excel(writer, sheet_name="Drug_Repurposing", index=False)
