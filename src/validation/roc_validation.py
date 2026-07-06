import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import (
    train_test_split
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    confusion_matrix
)

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]

# =====================================================
# LOAD DATA
# =====================================================

expression_df = pd.read_csv(
    BASE_DIR /
    "data" /
    "processed" /
    "expression_logCPM.csv"
)

metadata_df = pd.read_csv(
    BASE_DIR /
    "data" /
    "metadata" /
    "tcga_chol_metadata.csv"
)

panel_df = pd.read_csv(
    BASE_DIR /
    "results" /
    "clinical_biomarker_panel.csv"
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

expression_df = expression_df[
    expression_df["gene_name"]
    .isin(panel_genes)
]

print(
    "Panel Genes Found:",
    len(expression_df)
)

# =====================================================
# BUILD ML MATRIX
# =====================================================

X = expression_df.iloc[:,2:].T

y = np.array([
    sample_labels[s]
    for s in X.index
])

# =====================================================
# TRAIN TEST
# =====================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
)

# =====================================================
# MODEL
# =====================================================

rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

pred = rf.predict(
    X_test
)

probs = rf.predict_proba(
    X_test
)[:,1]

# =====================================================
# METRICS
# =====================================================

acc = accuracy_score(
    y_test,
    pred
)

fpr, tpr, _ = roc_curve(
    y_test,
    probs
)

roc_auc = auc(
    fpr,
    tpr
)

print(
    "Accuracy:",
    acc
)

print(
    "AUC:",
    roc_auc
)

print(
    confusion_matrix(
        y_test,
        pred
    )
)

# =====================================================
# PLOT
# =====================================================

plt.figure(
    figsize=(8,6)
)

plt.plot(
    fpr,
    tpr,
    label=f"AUC={roc_auc:.3f}"
)

plt.plot(
    [0,1],
    [0,1]
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    BASE_DIR /
    "results" /
    "roc_curve.png",
    dpi=300
)

plt.show()