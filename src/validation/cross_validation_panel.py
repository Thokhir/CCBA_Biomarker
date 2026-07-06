import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    StratifiedKFold
)

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score
)

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
# PANEL GENES
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
# FEATURE MATRIX
# =====================================================

X = panel_expression.iloc[:,2:].T

y = np.array([
    sample_labels[s]
    for s in X.index
])

print(
    "Samples:",
    len(X)
)

print(
    "Features:",
    X.shape[1]
)

# =====================================================
# CROSS VALIDATION
# =====================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

accuracy_scores = []
auc_scores = []

fold = 1

for train_idx, test_idx in cv.split(X, y):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

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

    acc = accuracy_score(
        y_test,
        pred
    )

    auc = roc_auc_score(
        y_test,
        probs
    )

    accuracy_scores.append(
        acc
    )

    auc_scores.append(
        auc
    )

    print(
        f"Fold {fold}"
    )

    print(
        f"Accuracy = {acc:.4f}"
    )

    print(
        f"AUC = {auc:.4f}"
    )

    print("-" * 40)

    fold += 1

# =====================================================
# SUMMARY
# =====================================================

print("\n")
print("=" * 60)

print(
    "Mean Accuracy:",
    np.mean(
        accuracy_scores
    )
)

print(
    "Std Accuracy:",
    np.std(
        accuracy_scores
    )
)

print(
    "Mean AUC:",
    np.mean(
        auc_scores
    )
)

print(
    "Std AUC:",
    np.std(
        auc_scores
    )
)

print("=" * 60)