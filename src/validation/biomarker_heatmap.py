import pandas as pd
import numpy as np

from pathlib import Path

import seaborn as sns
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
# LOAD DATA
# =====================================================

print("Loading files...")

expression_df = pd.read_csv(
    expression_file
)

metadata_df = pd.read_csv(
    metadata_file
)

panel_df = pd.read_csv(
    panel_file
)

print(
    "Expression Shape:",
    expression_df.shape
)

print(
    "Panel Shape:",
    panel_df.shape
)

# =====================================================
# PANEL GENES
# =====================================================

panel_genes = set(
    panel_df["gene_name"]
)

heatmap_df = expression_df[
    expression_df["gene_name"].isin(
        panel_genes
    )
].copy()

print(
    "Panel genes found:",
    len(heatmap_df)
)

# =====================================================
# BUILD MATRIX
# =====================================================

heatmap_matrix = heatmap_df.set_index(
    "gene_name"
)

heatmap_matrix = heatmap_matrix.iloc[:,1:]

# =====================================================
# SAMPLE ANNOTATION
# =====================================================

metadata_df["label"] = metadata_df[
    "sample_type"
].apply(
    lambda x:
    "Tumor"
    if x == "Primary Tumor"
    else "Normal"
)

sample_group = {}

for _, row in metadata_df.iterrows():

    sample_group[
        row["file_id"]
    ] = row["label"]

column_labels = []

for sample in heatmap_matrix.columns:

    if sample in sample_group:

        column_labels.append(
            sample_group[sample]
        )

    else:

        column_labels.append(
            "Unknown"
        )

# =====================================================
# Z-SCORE NORMALIZATION
# =====================================================

heatmap_matrix = heatmap_matrix.apply(
    lambda row:
    (
        row - row.mean()
    ) /
    row.std(),
    axis=1
)

heatmap_matrix = heatmap_matrix.replace(
    [np.inf, -np.inf],
    np.nan
)

heatmap_matrix = heatmap_matrix.fillna(0)

# =====================================================
# COLOR ANNOTATIONS
# =====================================================

col_colors = pd.Series(
    column_labels,
    index=heatmap_matrix.columns
)

col_colors = col_colors.map({
    "Tumor": "red",
    "Normal": "blue",
    "Unknown": "gray"
})

# =====================================================
# CLUSTERED HEATMAP
# =====================================================

sns.clustermap(
    heatmap_matrix,
    cmap="vlag",
    figsize=(16,10),
    col_colors=col_colors,
    xticklabels=False,
    yticklabels=True
)

plt.suptitle(
    "Clinical Biomarker Heatmap",
    y=1.02
)

output_file = (
    BASE_DIR /
    "results" /
    "biomarker_heatmap.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Heatmap saved to:\n{output_file}"
)