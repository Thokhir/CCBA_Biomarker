import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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

expression_df = pd.read_csv(
    expression_file
)

metadata_df = pd.read_csv(
    metadata_file
)

panel_df = pd.read_csv(
    panel_file
)

# =====================================================
# LABELS
# =====================================================

if "label" not in metadata_df.columns:

    metadata_df["label"] = (
        metadata_df["sample_type"]
        .apply(
            lambda x:
            1 if x == "Primary Tumor"
            else 0
        )
    )

sample_labels = {}

for _, row in metadata_df.iterrows():

    sample_labels[
        row["file_id"]
    ] = row["label"]

# =====================================================
# FILTER TO PANEL GENES
# =====================================================

panel_genes = set(
    panel_df["gene_name"]
)

panel_expression = expression_df[
    expression_df["gene_name"]
    .isin(panel_genes)
]

print(
    "Panel Genes:",
    len(panel_expression)
)

# =====================================================
# BUILD FEATURE MATRIX
# =====================================================

X = panel_expression.iloc[:,2:].T

y = np.array([
    sample_labels[s]
    for s in X.index
])

# =====================================================
# STANDARDIZE
# =====================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)

# =====================================================
# TRAIN MODEL
# =====================================================

rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)

rf.fit(
    X_scaled,
    y
)

# =====================================================
# DIAGNOSTIC SCORE
# =====================================================

probabilities = rf.predict_proba(
    X_scaled
)[:,1]

scores = probabilities * 100

results_df = pd.DataFrame({

    "sample_id":
        X.index,

    "diagnostic_score":
        scores,

    "prediction_probability":
        probabilities
})

# =====================================================
# RISK CATEGORY
# =====================================================

def assign_risk(score):

    if score < 33:

        return "Low Risk"

    elif score < 66:

        return "Intermediate Risk"

    else:

        return "High Risk"

results_df["risk_category"] = (
    results_df["diagnostic_score"]
    .apply(assign_risk)
)

# =====================================================
# TRUE LABEL
# =====================================================

results_df["true_label"] = y

results_df["true_class"] = (
    results_df["true_label"]
    .apply(
        lambda x:
        "Tumor"
        if x == 1
        else "Normal"
    )
)

# =====================================================
# SAVE
# =====================================================

output_file = (
    BASE_DIR /
    "results" /
    "diagnostic_scores.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print(
    "\nTop Diagnostic Scores"
)

print(
    results_df
    .sort_values(
        "diagnostic_score",
        ascending=False
    )
    .head(20)
)

print(
    f"\nSaved:\n{output_file}"
)