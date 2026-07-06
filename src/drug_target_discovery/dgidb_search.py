"""DGIdb (Drug-Gene Interaction Database) evidence for the clinical
biomarker panel, via DGIdb's public GraphQL API (no key required).
"""
import time

import pandas as pd
import requests

DGIDB_API = "https://dgidb.org/api/graphql"
SLEEP_INTERVAL = 0.2

QUERY = """
query GeneDrugs($name: String!) {
  genes(names: [$name]) {
    nodes {
      name
      interactions {
        drug { name }
        interactionScore
        interactionTypes { type }
      }
    }
  }
}
"""


def fetch_gene_interactions(gene: str) -> list:
    response = requests.post(DGIDB_API, json={"query": QUERY, "variables": {"name": gene}}, timeout=20)
    response.raise_for_status()
    nodes = response.json()["data"]["genes"]["nodes"]
    return nodes[0]["interactions"] if nodes else []


def run_dgidb_search(genes: list) -> pd.DataFrame:
    records = []
    for gene in genes:
        interactions = fetch_gene_interactions(gene)

        if not interactions:
            records.append({"gene_name": gene, "drug_name": None,
                             "interaction_score": None, "interaction_types": None})
        else:
            for interaction in interactions:
                types = "; ".join(t["type"] for t in interaction.get("interactionTypes", []))
                records.append({
                    "gene_name": gene, "drug_name": interaction["drug"]["name"],
                    "interaction_score": interaction["interactionScore"], "interaction_types": types,
                })
        time.sleep(SLEEP_INTERVAL)

    return pd.DataFrame(records)


if __name__ == "__main__":
    try:
        from . import open_targets_drugs as otd
    except ImportError:
        import open_targets_drugs as otd

    panel = pd.read_csv(otd.CLINICAL_PANEL_FILE)
    panel_genes = panel["gene_name"].tolist()

    result = run_dgidb_search(panel_genes)
    n_with_drugs = result.groupby("gene_name")["drug_name"].apply(lambda s: s.notna().any()).sum()
    print(f"DGIdb rows: {len(result)}, genes with >=1 interaction: {n_with_drugs} / {len(panel_genes)}")
    print(result[result["drug_name"].notna()])
