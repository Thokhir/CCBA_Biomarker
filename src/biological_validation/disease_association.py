"""Disease association for the clinical biomarker panel via Open Targets
(substituting for DisGeNET, whose classic REST API no longer returns data
without registration - confirmed directly; Open Targets aggregates
DisGeNET-sourced evidence anyway, and needs no API key).
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import open_targets_client as otc

BASE_DIR = Path(__file__).resolve().parents[2]
CLINICAL_PANEL_FILE = BASE_DIR / "results" / "clinical_biomarker_panel.csv"
SLEEP_INTERVAL = 0.2


def load_gene_to_ensembl() -> dict:
    panel = pd.read_csv(CLINICAL_PANEL_FILE)
    return dict(zip(panel["gene_name"], panel["gene_id"].str.split(".").str[0]))


def run_disease_association(genes: list, top_n: int = 5) -> pd.DataFrame:
    gene_to_ensembl = load_gene_to_ensembl()
    records = []

    for gene in genes:
        ensembl_id = gene_to_ensembl.get(gene)
        if ensembl_id is None:
            hit = otc.search_target_by_symbol(gene)
            ensembl_id = hit["id"] if hit else None

        rows = otc.fetch_disease_associations(ensembl_id, size=top_n) if ensembl_id else []

        if not rows:
            records.append({"gene_name": gene, "ensembl_id": ensembl_id,
                             "disease_name": None, "disease_id": None, "association_score": None})
        else:
            for r in rows:
                records.append({
                    "gene_name": gene, "ensembl_id": ensembl_id,
                    "disease_name": r["disease"]["name"], "disease_id": r["disease"]["id"],
                    "association_score": r["score"],
                })
        time.sleep(SLEEP_INTERVAL)

    return pd.DataFrame(records)


if __name__ == "__main__":
    try:
        from . import gene_ontology as go
    except ImportError:
        import gene_ontology as go

    panel_genes = go.load_panel_genes()
    result = run_disease_association(panel_genes)
    print(f"Total rows: {len(result)}, genes with >=1 association: {result.groupby('gene_name')['disease_name'].apply(lambda s: s.notna().any()).sum()} / {len(panel_genes)}")
    print(result[result["gene_name"] == "OTC"])
