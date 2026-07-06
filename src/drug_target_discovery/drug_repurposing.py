"""Drug repurposing candidates: known drugs (from DGIdb and/or Open Targets)
targeting clinical-panel genes, cross-referenced with each gene's target
priority rank so the most actionable candidates (already-drugged AND
highly-prioritized) stand out from drugs targeting low-priority genes.

DrugBank is not integrated here - its API now requires a paid academic
license (confirmed during Module 10 planning); DGIdb and Open Targets are
both free and already provide real interaction/known-drug evidence for
this panel.
"""
import pandas as pd


def build_repurposing_candidates(dgidb_df: pd.DataFrame, open_targets_drugs_df: pd.DataFrame,
                                  priority_df: pd.DataFrame) -> pd.DataFrame:
    dgidb_candidates = dgidb_df.dropna(subset=["drug_name"])[["gene_name", "drug_name"]].copy()
    dgidb_candidates["source"] = "DGIdb"

    ot_candidates = open_targets_drugs_df.dropna(subset=["drug_name"])[["gene_name", "drug_name"]].copy()
    ot_candidates["source"] = "OpenTargets"

    combined = pd.concat([dgidb_candidates, ot_candidates], ignore_index=True)
    combined = combined.drop_duplicates(subset=["gene_name", "drug_name"])

    combined = combined.merge(
        priority_df[["gene_name", "target_priority_score"]], on="gene_name", how="left"
    )
    return combined.sort_values("target_priority_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    try:
        from . import dgidb_search as ds
        from . import open_targets_drugs as otd
        from . import target_prioritization as tp
    except ImportError:
        import dgidb_search as ds
        import open_targets_drugs as otd
        import target_prioritization as tp

    panel_genes = pd.read_csv(otd.CLINICAL_PANEL_FILE)["gene_name"].tolist()
    dgidb_results = ds.run_dgidb_search(panel_genes)
    ot_drug_results = otd.run_known_drugs(panel_genes)
    priority_table = tp.build_target_priority_table(panel_genes)

    candidates = build_repurposing_candidates(dgidb_results, ot_drug_results, priority_table)
    print(f"Repurposing candidates: {len(candidates)}")
    print(candidates.head(15))
