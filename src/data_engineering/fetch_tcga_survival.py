"""Fetches TCGA-CHOL clinical/survival data from the GDC API for the same
36 patients already used in data/metadata/tcga_chol_metadata.csv.

This is a data-acquisition prerequisite for the planned Survival Analysis
module (Module 11) - tcga_chol_metadata.csv only carries file_id/case_id/
sample_type, with no survival endpoints. Output follows standard survival
analysis convention: OS_time (days) and OS_status (1=death observed,
0=censored at last follow-up).
"""
import json
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[2]
METADATA_DIR = BASE_DIR / "data" / "metadata"
CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"

FIELDS = [
    "submitter_id",
    "demographic.vital_status",
    "demographic.age_at_index",
    "demographic.days_to_death",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.primary_diagnosis",
]


def load_case_ids() -> list:
    metadata = pd.read_csv(METADATA_DIR / "tcga_chol_metadata.csv")
    return sorted(metadata["case_id"].unique().tolist())


def fetch_clinical_data(case_ids: list) -> list:
    filters = {"op": "in", "content": {"field": "submitter_id", "value": case_ids}}
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(FIELDS),
        "format": "JSON",
        "size": str(len(case_ids)),
    }
    response = requests.get(CASES_ENDPOINT, params=params, timeout=30)
    response.raise_for_status()
    return response.json()["data"]["hits"]


def select_primary_diagnosis(diagnoses: list) -> dict:
    if not diagnoses:
        return {}
    for d in diagnoses:
        if "cholangiocarcinoma" in d.get("primary_diagnosis", "").lower():
            return d
    return diagnoses[0]


def parse_hits(hits: list) -> pd.DataFrame:
    records = []
    for hit in hits:
        demographic = hit.get("demographic", {}) or {}
        diagnosis = select_primary_diagnosis(hit.get("diagnoses", []) or [])

        vital_status = demographic.get("vital_status")
        days_to_death = demographic.get("days_to_death")
        days_to_last_follow_up = diagnosis.get("days_to_last_follow_up")

        if vital_status == "Dead":
            os_time = days_to_death
            os_status = 1
        else:
            os_time = days_to_last_follow_up
            os_status = 0

        records.append({
            "case_id": hit["submitter_id"],
            "vital_status": vital_status,
            "age_at_index": demographic.get("age_at_index"),
            "ajcc_pathologic_stage": diagnosis.get("ajcc_pathologic_stage"),
            "primary_diagnosis": diagnosis.get("primary_diagnosis"),
            "days_to_death": days_to_death,
            "days_to_last_follow_up": days_to_last_follow_up,
            "OS_time": os_time,
            "OS_status": os_status,
        })

    return pd.DataFrame(records)


def run() -> None:
    case_ids = load_case_ids()
    print(f"Fetching clinical data for {len(case_ids)} TCGA-CHOL cases...")

    hits = fetch_clinical_data(case_ids)
    print(f"Received {len(hits)} case records from GDC.")

    survival_df = parse_hits(hits)

    missing = set(case_ids) - set(survival_df["case_id"])
    if missing:
        print(f"WARNING: no clinical record returned for {len(missing)} cases: {sorted(missing)}")

    missing_os_time = survival_df["OS_time"].isna().sum()
    print(f"Cases with usable OS_time: {len(survival_df) - missing_os_time} / {len(survival_df)}")
    print(survival_df["vital_status"].value_counts())
    print(survival_df["ajcc_pathologic_stage"].value_counts(dropna=False))

    out_file = METADATA_DIR / "tcga_chol_survival.csv"
    survival_df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    run()
