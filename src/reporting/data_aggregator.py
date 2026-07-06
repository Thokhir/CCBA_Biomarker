"""Pulls real headline statistics and tables from every completed module
(8-12) into one PlatformSummary, the single source of truth every report
renderer (PDF/DOCX/HTML/reviewer package) builds from - so results are
computed once and never re-derived or transcribed differently per format.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"


@dataclass
class PlatformSummary:
    model_metadata: dict
    feature_importance: pd.DataFrame
    overall_metrics: pd.Series
    cohort_metrics: pd.DataFrame
    shap_global_importance: pd.DataFrame
    explainability_summary: dict
    biomarker_annotation: pd.DataFrame
    biological_validation_summary: dict
    survival_summary: dict
    hazard_ratios: pd.DataFrame
    km_statistics: pd.DataFrame
    therapeutic_priority: pd.DataFrame
    drug_repurposing: pd.DataFrame
    drug_target_summary: dict


def _read_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / relative_path)


def _read_json(relative_path: str) -> dict:
    with open(RESULTS_DIR / relative_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_platform_summary() -> PlatformSummary:
    return PlatformSummary(
        model_metadata=_read_json("trained_model/model_metadata.json"),
        feature_importance=_read_csv("trained_model/feature_importance.csv"),
        overall_metrics=_read_csv("external_validation/overall_metrics.csv").iloc[0],
        cohort_metrics=_read_csv("external_validation/cohort_metrics.csv"),
        shap_global_importance=_read_csv("explainability/global/shap_global_importance.csv"),
        explainability_summary=_read_json("explainability/reports/explainability_summary.json"),
        biomarker_annotation=_read_csv("pathway_analysis/Biomarker_Annotation.csv"),
        biological_validation_summary=_read_json("pathway_analysis/reports/biological_validation_summary.json"),
        survival_summary=_read_json("survival/reports/survival_summary.json"),
        hazard_ratios=_read_csv("survival/Hazard_Ratios.csv"),
        km_statistics=_read_csv("survival/KM_statistics.csv"),
        therapeutic_priority=_read_csv("drug_targets/TherapeuticPriority.csv"),
        drug_repurposing=_read_csv("drug_targets/DrugRepurposing.csv"),
        drug_target_summary=_read_json("drug_targets/reports/drug_target_summary.json"),
    )


if __name__ == "__main__":
    summary = build_platform_summary()
    print("Model:", summary.model_metadata["training_samples"], "training samples")
    print("Overall ROC-AUC:", summary.overall_metrics["ROC_AUC"])
    print("Top SHAP gene:", summary.shap_global_importance.iloc[0]["gene_name"])
    print("Survival cohort:", summary.survival_summary["cohort_size"])
    print("Top drug candidate:", summary.drug_repurposing.iloc[0]["drug_name"])
