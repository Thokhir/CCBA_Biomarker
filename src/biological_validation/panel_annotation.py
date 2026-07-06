"""Consolidated per-gene biological validation summary: merges GO/pathway
term counts, STRING degree/hub status, top disease association, and
literature co-occurrence into one row per clinical-panel gene.

Named panel_annotation.py (not biomarker_annotation.py) to avoid confusion
with the pre-existing results/biomarker_annotation_table.csv, which maps a
different, larger gene list (from consensus_biomarker_signature.csv, the
nested-CV stability pool) - not the 20-gene clinical panel this module
concerns itself with.
"""
import pandas as pd


def _term_counts(genes: list, result_df: pd.DataFrame) -> pd.Series:
    exploded = result_df["Genes"].str.split(";").explode()
    counts = exploded.value_counts()
    return pd.Series(genes).map(counts).fillna(0).astype(int).values


def build_panel_annotation(genes: list, go_results: dict, pathway_results: dict,
                            string_degree_df: pd.DataFrame, disease_df: pd.DataFrame,
                            literature_df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({"gene_name": genes})

    for namespace, df in go_results.items():
        summary[f"n_GO_{namespace}_terms"] = _term_counts(genes, df)
    for name, df in pathway_results.items():
        summary[f"n_{name}_terms"] = _term_counts(genes, df)
    assert len(summary) == len(genes)

    summary = summary.merge(string_degree_df[["gene_name", "string_degree", "is_hub"]], on="gene_name", how="left")
    assert len(summary) == len(genes)

    valid_disease = disease_df.dropna(subset=["association_score"])
    top_disease = (
        valid_disease.sort_values("association_score", ascending=False)
        .groupby("gene_name").first()
        .rename(columns={"disease_name": "top_disease_name", "association_score": "top_disease_score"})
    )
    summary = summary.merge(top_disease[["top_disease_name", "top_disease_score"]],
                             on="gene_name", how="left")
    assert len(summary) == len(genes)

    summary = summary.merge(
        literature_df[["gene_name", "cca_hit_count", "novel_in_cca_literature"]], on="gene_name", how="left"
    )
    assert len(summary) == len(genes)

    return summary
