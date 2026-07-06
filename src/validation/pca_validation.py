import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

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

panel_file = (
    BASE_DIR /
    "results" /
    "clinical_biomarker_panel.csv"
)

# =====================================================
# LOAD FILES
# =====================================================

expression_df = pd.read_csv(
    expression_file
)

metadata_df = pd.read_csv(
    metadata_file
)

panel_df = pd.read_csv(
    panel_file
)

print("Expression Shape:", expression_df.shape)
print("Panel Shape:", panel_df.shape)

# =====================================================
# FILTER TO PANEL GENES
# =====================================================

panel_genes = set(
    panel_df["gene_name"]
)

expression_df = expression_df[
    expression_df["gene_name"].isin(
        panel_genes
    )
]

print(
    "Panel Genes Found:",
    len(expression_df)
)

# =====================================================
# BUILD FEATURE MATRIX
# =====================================================

X = expression_df.iloc[:,2:].T

print(
    "Feature Matrix Shape:",
    X.shape
)

# =====================================================
# SAMPLE LABELS
# =====================================================

metadata_df["group"] = metadata_df[
    "sample_type"
].apply(
    lambda x:
    "Tumor"
    if x == "Primary Tumor"
    else "Normal"
)

sample_groups = {}

for _, row in metadata_df.iterrows():

    sample_groups[
        row["file_id"]
    ] = row["group"]

groups = []

for sample in X.index:

    groups.append(
        sample_groups.get(
            sample,
            "Unknown"
        )
    )

# =====================================================
# STANDARDIZE DATA
# =====================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)

# =====================================================
# PCA
# =====================================================

pca = PCA(
    n_components=2,
    random_state=42
)

principal_components = pca.fit_transform(
    X_scaled
)

pca_df = pd.DataFrame({
    "PC1": principal_components[:,0],
    "PC2": principal_components[:,1],
    "Group": groups
})

# =====================================================
# VARIANCE EXPLAINED
# =====================================================

explained_variance = (
    pca.explained_variance_ratio_
)

print(
    "\nExplained Variance:"
)

print(
    explained_variance
)

print(
    "\nTotal Variance:",
    explained_variance.sum()
)

# =====================================================
# PLOT
# =====================================================

plt.figure(
    figsize=(10,8)
)

tumor_df = pca_df[
    pca_df["Group"] == "Tumor"
]

normal_df = pca_df[
    pca_df["Group"] == "Normal"
]

plt.scatter(
    normal_df["PC1"],
    normal_df["PC2"],
    label="Normal",
    s=120
)

plt.scatter(
    tumor_df["PC1"],
    tumor_df["PC2"],
    label="Tumor",
    s=120
)

plt.xlabel(
    f"PC1 ({explained_variance[0]*100:.2f}%)"
)

plt.ylabel(
    f"PC2 ({explained_variance[1]*100:.2f}%)"
)

plt.title(
    "PCA of Clinical Biomarker Panel"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

output_file = (
    BASE_DIR /
    "results" /
    "pca_biomarker_panel.png"
)

plt.savefig(
    output_file,
    dpi=300
)

plt.show()

print(
    f"\nSaved PCA plot:\n{output_file}"
)