"""Open Targets known-drug and tractability evidence for the clinical
biomarker panel. Reuses src/utils/open_targets_client.py (already used by
Module 10 for disease association) rather than duplicating the GraphQL
client.
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


def run_known_drugs(genes: list) -> pd.DataFrame:
    gene_to_ensembl = load_gene_to_ensembl()
    records = []

    for gene in genes:
        ensembl_id = gene_to_ensembl.get(gene)
        rows = otc.fetch_known_drugs(ensembl_id) if ensembl_id else []

        if not rows:
            records.append({"gene_name": gene, "ensembl_id": ensembl_id, "drug_name": None,
                             "drug_type": None, "max_clinical_stage": None, "diseases": None})
        else:
            for r in rows:
                diseases = "; ".join(d["disease"]["name"] for d in r.get("diseases", []) if d.get("disease"))
                records.append({
                    "gene_name": gene, "ensembl_id": ensembl_id,
                    "drug_name": r["drug"]["name"], "drug_type": r["drug"]["drugType"],
                    "max_clinical_stage": r["maxClinicalStage"], "diseases": diseases,
                })
        time.sleep(SLEEP_INTERVAL)

    return pd.DataFrame(records)


def run_tractability(genes: list) -> pd.DataFrame:
    """One row per gene, summarizing druggability across modalities: counts
    of true tractability flags for small molecule (SM), antibody (AB),
    PROTAC (PR), and other modalities, plus whether an approved drug
    already exists for that gene (any modality)."""
    gene_to_ensembl = load_gene_to_ensembl()
    records = []

    for gene in genes:
        ensembl_id = gene_to_ensembl.get(gene)
        flags = otc.fetch_tractability(ensembl_id) if ensembl_id else []
        true_flags = [f for f in flags if f["value"]]

        records.append({
            "gene_name": gene,
            "ensembl_id": ensembl_id,
            "n_druggability_flags": len(true_flags),
            "has_approved_drug": any(f["label"] == "Approved Drug" and f["value"] for f in flags),
            "is_druggable_family_sm": any(
                f["label"] == "Druggable Family" and f["modality"] == "SM" and f["value"] for f in flags
            ),
            "druggability_labels": "; ".join(f"{f['modality']}:{f['label']}" for f in true_flags),
        })
        time.sleep(SLEEP_INTERVAL)

    return pd.DataFrame(records)


if __name__ == "__main__":
    panel = pd.read_csv(CLINICAL_PANEL_FILE)
    panel_genes = panel["gene_name"].tolist()

    drugs_df = run_known_drugs(panel_genes)
    print(f"Known drug rows: {len(drugs_df)}, genes with >=1 known drug: "
          f"{drugs_df.groupby('gene_name')['drug_name'].apply(lambda s: s.notna().any()).sum()} / {len(panel_genes)}")

    tract_df = run_tractability(panel_genes)
    print(tract_df[["gene_name", "n_druggability_flags", "has_approved_drug", "is_druggable_family_sm"]])
