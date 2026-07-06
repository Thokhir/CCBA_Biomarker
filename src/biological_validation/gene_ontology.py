"""Gene Ontology enrichment (Biological Process / Cellular Component /
Molecular Function) for the 20-gene clinical biomarker panel via Enrichr.

Hosts the shared run_enrichr() helper used by both this file and
pathway_enrichment.py (KEGG/Reactome/WikiPathways), since all four
libraries are queried through the exact same gseapy.enrichr() call with
only the gene-set-library string differing.
"""
from pathlib import Path

import gseapy as gp
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURE_ORDER_FILE = BASE_DIR / "results" / "trained_model" / "feature_order.csv"

GO_LIBRARIES = {
    "BP": "GO_Biological_Process_2025",
    "CC": "GO_Cellular_Component_2025",
    "MF": "GO_Molecular_Function_2025",
}

RESULT_COLUMNS = ["gene_set_library", "Term", "Overlap", "P-value",
                  "Adjusted P-value", "Odds Ratio", "Combined Score", "Genes"]


def load_panel_genes() -> list:
    return pd.read_csv(FEATURE_ORDER_FILE)["gene_name"].tolist()


def run_enrichr(genes: list, gene_set_library: str) -> pd.DataFrame:
    result = gp.enrichr(gene_list=genes, gene_sets=[gene_set_library], organism="human", outdir=None)
    df = result.results.copy()
    df = df.sort_values("Adjusted P-value").reset_index(drop=True)
    df.insert(0, "gene_set_library", gene_set_library)
    return df[RESULT_COLUMNS]


def run_go_analysis(genes: list) -> dict:
    return {namespace: run_enrichr(genes, library) for namespace, library in GO_LIBRARIES.items()}


if __name__ == "__main__":
    panel_genes = load_panel_genes()
    print(f"Panel genes: {len(panel_genes)}")
    go_results = run_go_analysis(panel_genes)
    for namespace, df in go_results.items():
        print(f"\nGO_{namespace}: {len(df)} terms, min adj p-value = {df['Adjusted P-value'].min()}")
        print(df.head(3))
