"""
=========================================================
DAY 7
Differential Expression Analysis
Industrial Version
=========================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

expression_file = (
    BASE_DIR /
    "data" /
    "processed" /
    "expression_logCPM.csv"
)

metadata_file = (
    BASE_DIR /
    "data" /
    "metadata" /
    "tcga_chol_metadata.csv"
)

RESULT_DIR = (
    BASE_DIR /
    "results"
)

RESULT_DIR.mkdir(
    exist_ok=True
)

# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading Expression Matrix...")

expression_df = pd.read_csv(
    expression_file
)

print(
    "Expression Shape:",
    expression_df.shape
)

print("\nLoading Metadata...")

metadata_df = pd.read_csv(
    metadata_file
)

print(
    "Metadata Shape:",
    metadata_df.shape
)

# =====================================================
# CREATE LABELS
# =====================================================

if "label" not in metadata_df.columns:

    metadata_df["label"] = metadata_df[
        "sample_type"
    ].apply(
        lambda x:
        1 if x == "Primary Tumor"
        else 0
    )

# =====================================================
# SAMPLE GROUPS
# =====================================================

tumor_samples = metadata_df[
    metadata_df["label"] == 1
]["file_id"].tolist()

normal_samples = metadata_df[
    metadata_df["label"] == 0
]["file_id"].tolist()

available_samples = expression_df.columns[2:]

tumor_samples = [
    s for s in tumor_samples
    if s in available_samples
]

normal_samples = [
    s for s in normal_samples
    if s in available_samples
]

print(
    "\nTumor Samples:",
    len(tumor_samples)
)

print(
    "Normal Samples:",
    len(normal_samples)
)

if len(tumor_samples) == 0:
    raise ValueError(
        "No tumor samples matched."
    )

if len(normal_samples) == 0:
    raise ValueError(
        "No normal samples matched."
    )

# =====================================================
# CHECK DATA TYPES
# =====================================================

print("\nColumn Data Types")

print(
    expression_df.dtypes.head(10)
)

# =====================================================
# DEA
# =====================================================

results = []

print(
    "\nRunning DEA..."
)

for idx, row in expression_df.iterrows():

    gene_id = row["gene_id"]
    gene_name = row["gene_name"]

    tumor_values = pd.to_numeric(
        row[tumor_samples],
        errors="coerce"
    ).values.astype(float)

    normal_values = pd.to_numeric(
        row[normal_samples],
        errors="coerce"
    ).values.astype(float)

    tumor_values = tumor_values[
        ~np.isnan(tumor_values)
    ]

    normal_values = normal_values[
        ~np.isnan(normal_values)
    ]

    if (
        len(tumor_values) < 3
        or
        len(normal_values) < 3
    ):
        continue

    if idx == 0:

        print("\nDEBUG FIRST GENE")
        print("Gene:", gene_name)

        print(
            "Tumor dtype:",
            tumor_values.dtype
        )

        print(
            "Normal dtype:",
            normal_values.dtype
        )

        print(
            "Tumor values:",
            tumor_values[:5]
        )

        print(
            "Normal values:",
            normal_values[:5]
        )

    tumor_mean = np.mean(
        tumor_values
    )

    normal_mean = np.mean(
        normal_values
    )

    log2fc = np.log2(
        (tumor_mean + 0.01)
        /
        (normal_mean + 0.01)
    )

    try:

        stat, pvalue = mannwhitneyu(
            tumor_values,
            normal_values,
            alternative="two-sided"
        )

    except Exception as e:

        print(
            f"ERROR: {gene_name}"
        )

        print(e)

        pvalue = np.nan

    results.append({

        "gene_id":
            gene_id,

        "gene_name":
            gene_name,

        "log2FC":
            log2fc,

        "pvalue":
            pvalue
    })

# =====================================================
# RESULTS DATAFRAME
# =====================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.dropna(
    subset=["pvalue"]
)

print(
    "\nTotal Tested Genes:",
    len(results_df)
)

# =====================================================
# MULTIPLE TESTING
# =====================================================

results_df["padj"] = multipletests(
    results_df["pvalue"],
    method="fdr_bh"
)[1]

# =====================================================
# DIAGNOSTICS
# =====================================================

print("\nP-value Summary")

print(
    results_df["pvalue"].describe()
)

print("\nAdjusted P-value Summary")

print(
    results_df["padj"].describe()
)

print(
    "\nMinimum P-value:",
    results_df["pvalue"].min()
)

print(
    "Minimum Adjusted P-value:",
    results_df["padj"].min()
)

# =====================================================
# BIOMARKER TIERS
# =====================================================

strict_genes = results_df[

    (abs(results_df["log2FC"]) >= 1)

    &

    (results_df["padj"] < 0.05)

]

moderate_genes = results_df[

    (abs(results_df["log2FC"]) >= 1)

    &

    (results_df["pvalue"] < 0.01)

]

exploratory_genes = results_df[

    (abs(results_df["log2FC"]) >= 0.5)

    &

    (results_df["pvalue"] < 0.05)

]

strict_genes = strict_genes.sort_values(
    "padj"
)

moderate_genes = moderate_genes.sort_values(
    "pvalue"
)

exploratory_genes = exploratory_genes.sort_values(
    "pvalue"
)

# =====================================================
# SAVE RESULTS
# =====================================================

results_df.to_csv(
    RESULT_DIR /
    "all_genes_dea.csv",
    index=False
)

strict_genes.to_csv(
    RESULT_DIR /
    "candidate_biomarkers.csv",
    index=False
)

moderate_genes.to_csv(
    RESULT_DIR /
    "candidate_biomarkers_moderate.csv",
    index=False
)

exploratory_genes.to_csv(
    RESULT_DIR /
    "candidate_biomarkers_exploratory.csv",
    index=False
)

# =====================================================
# VOLCANO PLOT
# =====================================================

results_df["minus_log10_padj"] = (
    -np.log10(
        results_df["padj"]
        +
        1e-300
    )
)

plt.figure(
    figsize=(10, 8)
)

sns.scatterplot(
    data=results_df,
    x="log2FC",
    y="minus_log10_padj",
    alpha=0.5
)

plt.axvline(
    1,
    linestyle="--"
)

plt.axvline(
    -1,
    linestyle="--"
)

plt.axhline(
    -np.log10(0.05),
    linestyle="--"
)

plt.title(
    "Volcano Plot"
)

plt.tight_layout()

plt.savefig(
    RESULT_DIR /
    "volcano_plot.png",
    dpi=300
)

plt.close()

# =====================================================
# SUMMARY
# =====================================================

print("\n")
print("=" * 60)

print(
    "STRICT BIOMARKERS:",
    len(strict_genes)
)

print(
    "MODERATE BIOMARKERS:",
    len(moderate_genes)
)

print(
    "EXPLORATORY BIOMARKERS:",
    len(exploratory_genes)
)

print("=" * 60)

print("\nTop Strict Biomarkers")

print(
    strict_genes.head(20)
)

print("\nTop Moderate Biomarkers")

print(
    moderate_genes.head(20)
)

print("\nTop Exploratory Biomarkers")

print(
    exploratory_genes.head(20)
)