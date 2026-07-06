"""Builds the analysis-ready survival cohort: tumor-sample expression for the
20 clinical biomarker panel genes, merged with TCGA-CHOL survival endpoints.

Restricted to Primary Tumor samples only (survival-by-biomarker-expression is
not meaningful for normal tissue) with usable OS_time. Of 44 TCGA-CHOL
samples, 35 are Primary Tumor and 34 of those have usable follow-up data -
a small cohort (18 deaths observed), which caps how many covariates any
downstream model can reliably support.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
EXPRESSION_FILE = BASE_DIR / "data" / "processed" / "expression_logCPM.csv"
METADATA_FILE = BASE_DIR / "data" / "metadata" / "tcga_chol_metadata.csv"
SURVIVAL_FILE = BASE_DIR / "data" / "metadata" / "tcga_chol_survival.csv"
FEATURE_ORDER_FILE = BASE_DIR / "results" / "trained_model" / "feature_order.csv"


def load_panel_genes() -> list:
    return pd.read_csv(FEATURE_ORDER_FILE)["gene_name"].tolist()


def load_survival_cohort() -> pd.DataFrame:
    """Returns one row per tumor sample with usable survival data, columns:
    file_id, case_id, OS_time, OS_status, age_at_index, ajcc_pathologic_stage,
    plus one column per clinical-panel gene (log-CPM expression).
    """
    genes = load_panel_genes()

    expr = pd.read_csv(EXPRESSION_FILE)
    expr["gene_name"] = expr["gene_name"].astype(str).str.upper()
    expr = expr.drop(columns=["gene_id"])
    if expr["gene_name"].duplicated().any():
        sample_columns = expr.columns[1:]
        expr = expr.groupby("gene_name", as_index=False)[sample_columns].mean()
    expr = expr.set_index("gene_name")
    gene_matrix = expr.loc[genes].T
    gene_matrix.index.name = "file_id"
    gene_matrix = gene_matrix.reset_index()

    metadata = pd.read_csv(METADATA_FILE)
    survival = pd.read_csv(SURVIVAL_FILE)

    tumor_samples = metadata[metadata["sample_type"] == "Primary Tumor"]
    cohort = tumor_samples.merge(survival, on="case_id", how="left")
    cohort = cohort.merge(gene_matrix, on="file_id", how="left")
    cohort = cohort.dropna(subset=["OS_time"]).reset_index(drop=True)
    cohort["OS_status"] = cohort["OS_status"].astype(int)

    return cohort


def add_expression_groups(cohort: pd.DataFrame, genes: list) -> pd.DataFrame:
    """Adds a High/Low group column per gene via median split."""
    cohort = cohort.copy()
    for gene in genes:
        median = cohort[gene].median()
        cohort[f"{gene}_group"] = np.where(cohort[gene] > median, "High", "Low")
    return cohort


if __name__ == "__main__":
    panel_genes = load_panel_genes()
    survival_cohort = load_survival_cohort()
    print(f"Survival cohort: {len(survival_cohort)} tumor samples, "
          f"{survival_cohort['OS_status'].sum()} deaths, "
          f"{len(survival_cohort) - survival_cohort['OS_status'].sum()} censored")
    print(survival_cohort[["case_id", "OS_time", "OS_status"] + panel_genes[:3]].head())
