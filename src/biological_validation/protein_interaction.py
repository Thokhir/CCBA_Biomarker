"""STRING protein-protein interaction network and degree-based hub analysis
for the clinical biomarker panel.

Manual degree counting is used instead of networkx: with 20 nodes and a
confirmed 7-edge network, a graph library would be unjustified complexity
for what a plain dictionary count already answers.
"""
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

try:
    from . import gene_ontology as go
except ImportError:
    import gene_ontology as go

STRING_API_NETWORK = "https://string-db.org/api/tsv/network"
SPECIES_HUMAN = 9606
REQUIRED_SCORE = 400  # medium confidence
HUB_DEGREE_THRESHOLD = 2  # appropriate for this graph's small size, not a principled cutoff


def fetch_string_network(genes: list) -> pd.DataFrame:
    params = {
        "identifiers": "\r".join(genes),
        "species": SPECIES_HUMAN,
        "required_score": REQUIRED_SCORE,
    }
    response = requests.get(STRING_API_NETWORK, params=params, timeout=30)
    response.raise_for_status()
    if not response.text.strip():
        return pd.DataFrame(columns=[
            "stringId_A", "stringId_B", "preferredName_A", "preferredName_B",
            "ncbiTaxonId", "score", "nscore", "fscore", "pscore", "ascore", "escore", "dscore", "tscore",
        ])
    return pd.read_csv(StringIO(response.text), sep="\t")


def compute_degree_centrality(network_df: pd.DataFrame, all_genes: list) -> pd.DataFrame:
    degree = {gene: 0 for gene in all_genes}
    for _, row in network_df.iterrows():
        degree[row["preferredName_A"]] = degree.get(row["preferredName_A"], 0) + 1
        degree[row["preferredName_B"]] = degree.get(row["preferredName_B"], 0) + 1

    df = pd.DataFrame({"gene_name": list(degree.keys()), "string_degree": list(degree.values())})
    df["is_hub"] = df["string_degree"] >= HUB_DEGREE_THRESHOLD
    return df.sort_values("string_degree", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    panel_genes = go.load_panel_genes()
    network = fetch_string_network(panel_genes)
    print(f"STRING network edges: {len(network)}")
    print(network[["preferredName_A", "preferredName_B", "score"]] if len(network) else "No edges.")

    degree_df = compute_degree_centrality(network, panel_genes)
    print(f"\nDegree centrality ({degree_df['is_hub'].sum()} hub genes at threshold >= {HUB_DEGREE_THRESHOLD}):")
    print(degree_df)
