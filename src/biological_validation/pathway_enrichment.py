"""KEGG / Reactome / WikiPathways enrichment for the clinical biomarker panel.

Shares gene_ontology.run_enrichr() - these three libraries are queried
through the identical gseapy.enrichr() call GO uses, just a different
library string, so they don't get separate near-duplicate files.
"""
try:
    from . import gene_ontology as go
except ImportError:
    import gene_ontology as go

PATHWAY_LIBRARIES = {
    "KEGG": "KEGG_2021_Human",
    "Reactome": "Reactome_Pathways_2024",
    "WikiPathways": "WikiPathways_2024_Human",
}


def run_pathway_analysis(genes: list) -> dict:
    return {name: go.run_enrichr(genes, library) for name, library in PATHWAY_LIBRARIES.items()}


if __name__ == "__main__":
    panel_genes = go.load_panel_genes()
    results = run_pathway_analysis(panel_genes)
    for name, df in results.items():
        print(f"\n{name}: {len(df)} terms, min adj p-value = {df['Adjusted P-value'].min()}")
        print(df.head(3)[["Term", "Genes", "Adjusted P-value"]])
