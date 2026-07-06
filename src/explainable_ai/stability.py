"""Biomarker stability integration: merges every importance/stability signal
computed across the project into one table per clinically-deployed gene, to
identify biomarkers that are simultaneously predictive, stable, and
clinically measurable.

The 20 clinical-panel genes and the 53 "nested CV feature stability" genes
have zero overlap (confirmed directly) - this is the expected, documented
behavior described in the project's own design rationale (feature stability
optimizes internal TCGA model performance; the clinical panel optimizes
reproducibility across independent GEO cohorts), not a bug. This table
surfaces that fact explicitly rather than letting a selection_count of 0
look like an error.
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
CLINICAL_PANEL_FILE = BASE_DIR / "results" / "clinical_biomarker_panel.csv"
NESTED_CV_STABILITY_FILE = BASE_DIR / "results" / "nested_cv_feature_stability.csv"
RF_FEATURE_IMPORTANCE_FILE = BASE_DIR / "results" / "trained_model" / "feature_importance.csv"


def build_stability_summary(shap_importance_df: pd.DataFrame, permutation_importance_df: pd.DataFrame,
                             feature_order: list) -> pd.DataFrame:
    clinical_panel = pd.read_csv(CLINICAL_PANEL_FILE)[
        ["gene_name", "dea_rank", "rf_rank", "xgb_rank", "consensus_score"]
    ].rename(columns={"rf_rank": "discovery_rf_rank", "xgb_rank": "discovery_xgb_rank"})

    nested_cv_stability = pd.read_csv(NESTED_CV_STABILITY_FILE)[["gene_name", "selection_count"]]

    rf_importance = pd.read_csv(RF_FEATURE_IMPORTANCE_FILE)[["gene_name", "rank"]].rename(
        columns={"rank": "rf_builtin_rank"}
    )
    shap_rank = shap_importance_df[["gene_name", "rank"]].rename(columns={"rank": "shap_rank"})
    permutation_rank = permutation_importance_df[["gene_name", "rank_auc"]].rename(
        columns={"rank_auc": "permutation_rank"}
    )

    summary = pd.DataFrame({"gene_name": feature_order})
    summary = summary.merge(clinical_panel, on="gene_name", how="left")
    summary = summary.merge(nested_cv_stability, on="gene_name", how="left")
    summary["selection_count"] = summary["selection_count"].fillna(0).astype(int)
    summary = summary.merge(rf_importance, on="gene_name", how="left")
    summary = summary.merge(shap_rank, on="gene_name", how="left")
    summary = summary.merge(permutation_rank, on="gene_name", how="left")

    summary["in_nested_cv_stability_pool"] = summary["selection_count"] > 0
    return summary.sort_values("shap_rank").reset_index(drop=True)
