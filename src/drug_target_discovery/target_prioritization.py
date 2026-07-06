"""Composite target prioritization score, combining four independent
per-gene signals already computed by earlier modules:

- predictive_importance: SHAP global importance rank (Module 9, TCGA training)
- external_relevance: mean |SHAP| across all 444 external cohort samples
  (Module 9) - a genuine per-gene "does this signal hold up outside TCGA"
  measure, distinct from training-only importance
- biological_relevance: GO/pathway term counts + STRING degree (Module 10)
- survival_association: 1 - univariate Cox p-value (Module 11)

Each is min-max normalized to [0, 1] and averaged (equal weights - no
principled basis exists yet for weighting one signal over another, so an
unweighted mean is the honest default rather than an invented weighting
scheme). Druggability (this module) is reported alongside the score as
supporting evidence, not folded into the ranking itself, since a gene can be
a high-priority *biomarker* without yet having any known chemical matter -
conflating "important" with "druggable" would bias the ranking away from
genuinely novel targets that deserve investment precisely because no drug
exists yet.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
SHAP_GLOBAL_FILE = BASE_DIR / "results" / "explainability" / "global" / "shap_global_importance.csv"
SHAP_EXTERNAL_FILE = BASE_DIR / "results" / "explainability" / "local" / "shap_values_external_all.csv"
PANEL_ANNOTATION_FILE = BASE_DIR / "results" / "pathway_analysis" / "Biomarker_Annotation.csv"
HAZARD_RATIOS_FILE = BASE_DIR / "results" / "survival" / "Hazard_Ratios.csv"


def _minmax(series: pd.Series) -> pd.Series:
    value_range = series.max() - series.min()
    if value_range == 0:
        return pd.Series(0.5, index=series.index)
    return (series - series.min()) / value_range


def load_predictive_importance() -> pd.DataFrame:
    df = pd.read_csv(SHAP_GLOBAL_FILE)
    return df[["gene_name", "mean_abs_shap"]]


def load_external_relevance(genes: list) -> pd.DataFrame:
    external = pd.read_csv(SHAP_EXTERNAL_FILE)
    records = [{"gene_name": gene, "mean_abs_shap_external": external[f"shap_{gene}"].abs().mean()}
               for gene in genes]
    return pd.DataFrame(records)


def load_biological_relevance() -> pd.DataFrame:
    df = pd.read_csv(PANEL_ANNOTATION_FILE)
    term_cols = [c for c in df.columns if c.startswith("n_GO_") or c.startswith("n_") and c.endswith("_terms")]
    df["total_bio_terms"] = df[term_cols].sum(axis=1)
    return df[["gene_name", "total_bio_terms", "string_degree"]]


def load_survival_association() -> pd.DataFrame:
    df = pd.read_csv(HAZARD_RATIOS_FILE)
    return df[["gene_name", "p_value"]].rename(columns={"p_value": "cox_p_value"})


def build_target_priority_table(genes: list) -> pd.DataFrame:
    table = pd.DataFrame({"gene_name": genes})

    table = table.merge(load_predictive_importance(), on="gene_name", how="left")
    table = table.merge(load_external_relevance(genes), on="gene_name", how="left")
    table = table.merge(load_biological_relevance(), on="gene_name", how="left")
    table = table.merge(load_survival_association(), on="gene_name", how="left")
    assert len(table) == len(genes)

    table["score_predictive"] = _minmax(table["mean_abs_shap"])
    table["score_external"] = _minmax(table["mean_abs_shap_external"])
    table["score_biological"] = _minmax(table["total_bio_terms"] + table["string_degree"])
    table["score_survival"] = _minmax(1 - table["cox_p_value"])

    table["target_priority_score"] = table[
        ["score_predictive", "score_external", "score_biological", "score_survival"]
    ].mean(axis=1)

    return table.sort_values("target_priority_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    panel_genes = pd.read_csv(BASE_DIR / "results" / "trained_model" / "feature_order.csv")["gene_name"].tolist()
    priority = build_target_priority_table(panel_genes)
    print(priority[["gene_name", "target_priority_score", "score_predictive",
                     "score_external", "score_biological", "score_survival"]])
