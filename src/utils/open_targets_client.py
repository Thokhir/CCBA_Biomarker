"""Shared Open Targets GraphQL client: resolves genes to Open Targets target
records and fetches disease/drug association evidence.

Used by Module 10 (biological_validation/disease_association.py) for disease
association, and intended for reuse by the future Module 12 (drug target
discovery) for drug/therapeutic evidence - written once here rather than
duplicated, to avoid the same overlap problem that led to removing drug
association from Module 10's own scope.
"""
import requests

OPEN_TARGETS_API = "https://api.platform.opentargets.org/api/v4/graphql"
REQUEST_TIMEOUT = 30


def resolve_ensembl_id(ensembl_id_versioned: str) -> str:
    return ensembl_id_versioned.split(".")[0]


def search_target_by_symbol(gene_symbol: str) -> dict:
    query = """
    query SearchTarget($q: String!) {
      search(queryString: $q, entityNames: ["target"]) {
        hits { id name entity }
      }
    }
    """
    response = requests.post(
        OPEN_TARGETS_API, json={"query": query, "variables": {"q": gene_symbol}},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    hits = response.json()["data"]["search"]["hits"]
    return hits[0] if hits else None


def fetch_disease_associations(ensembl_id: str, size: int = 10) -> list:
    query = """
    query TargetDiseases($ensemblId: String!, $size: Int!) {
      target(ensemblId: $ensemblId) {
        approvedSymbol
        associatedDiseases(page: {index: 0, size: $size}) {
          count
          rows {
            score
            disease { id name }
          }
        }
      }
    }
    """
    response = requests.post(
        OPEN_TARGETS_API,
        json={"query": query, "variables": {"ensemblId": ensembl_id, "size": size}},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()["data"]["target"]
    return data["associatedDiseases"]["rows"] if data else []


if __name__ == "__main__":
    hit = search_target_by_symbol("OTC")
    print("Symbol search for OTC:", hit)
    associations = fetch_disease_associations(resolve_ensembl_id("ENSG00000036473.8"), size=5)
    for a in associations:
        print(f"{a['disease']['name']}: score={a['score']:.4f}")
