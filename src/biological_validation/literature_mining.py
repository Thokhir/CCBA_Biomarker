"""PubMed literature co-occurrence mining for the clinical biomarker panel
via NCBI E-utilities. Structured hit-count mining, not LLM summarization -
deliberately kept simple, no new dependency beyond requests already used
throughout this codebase.

A gene with zero cholangiocarcinoma-specific hits is a real, expected result
(some panel genes may be genuinely novel/understudied in this disease
context) - flagged explicitly via novel_in_cca_literature rather than left
as an ambiguous blank.
"""
import time

import pandas as pd
import requests

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
SLEEP_INTERVAL = 0.35  # stays under NCBI's 3 req/sec unauthenticated limit

CCA_QUERY = '"{gene}"[Title/Abstract] AND ("cholangiocarcinoma"[Title/Abstract] OR "bile duct cancer"[Title/Abstract])'
CANCER_QUERY = '"{gene}"[Title/Abstract] AND "cancer"[Title/Abstract]'


def esearch_count_and_pmids(query: str, retmax: int = 5) -> tuple:
    params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": retmax}
    response = requests.get(ESEARCH, params=params, timeout=20)
    response.raise_for_status()
    result = response.json()["esearchresult"]
    return int(result["count"]), result.get("idlist", [])


def esummary_titles(pmids: list) -> dict:
    if not pmids:
        return {}
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"}
    response = requests.get(ESUMMARY, params=params, timeout=20)
    response.raise_for_status()
    result = response.json()["result"]
    return {pmid: result[pmid]["title"] for pmid in pmids if pmid in result}


def run_literature_mining(genes: list, top_n_pmids: int = 3) -> pd.DataFrame:
    records = []
    for gene in genes:
        cca_count, cca_pmids = esearch_count_and_pmids(CCA_QUERY.format(gene=gene), retmax=top_n_pmids)
        time.sleep(SLEEP_INTERVAL)
        cancer_count, _ = esearch_count_and_pmids(CANCER_QUERY.format(gene=gene), retmax=1)
        time.sleep(SLEEP_INTERVAL)
        titles = esummary_titles(cca_pmids)
        time.sleep(SLEEP_INTERVAL)

        records.append({
            "gene_name": gene,
            "cca_hit_count": cca_count,
            "cancer_broad_hit_count": cancer_count,
            "top_pmids": ";".join(cca_pmids),
            "top_titles": " | ".join(titles.get(p, "") for p in cca_pmids),
            "novel_in_cca_literature": cca_count == 0,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    try:
        from . import gene_ontology as go
    except ImportError:
        import gene_ontology as go

    panel_genes = go.load_panel_genes()
    result = run_literature_mining(panel_genes)
    print(result[["gene_name", "cca_hit_count", "cancer_broad_hit_count", "novel_in_cca_literature"]])
    print(f"\nNovel/understudied in CCA literature: {result['novel_in_cca_literature'].sum()} / {len(result)}")
