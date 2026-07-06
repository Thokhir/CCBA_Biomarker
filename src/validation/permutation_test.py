import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    StratifiedKFold
)

from sklearn.metrics import (
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

# =====================================================
# FEATURE MATRIX
# =====================================================

X = panel_expression.iloc[:,2:].T

y = np.array([
    sample_labels[s]
    for s in X.index
])

print("Samples:", len(X))
print("Features:", X.shape[1])

# =====================================================
# PERMUTE LABELS
# =====================================================

rng = np.random.default_rng(
    seed=42
)

y_permuted = rng.permutation(y)

print("\nOriginal Labels")

print(
    np.bincount(y)
)

print("\nPermuted Labels")

print(
    np.bincount(y_permuted)
)

# =====================================================
# CROSS VALIDATION
# =====================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

auc_scores = []

fold = 1

for train_idx, test_idx in cv.split(X, y_permuted):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y_permuted[train_idx]
    y_test = y_permuted[test_idx]

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    probs = model.predict_proba(
        X_test
    )[:,1]

    auc = roc_auc_score(
        y_test,
        probs
    )

    auc_scores.append(
        auc
    )

    print(
        f"Fold {fold} AUC = {auc:.4f}"
    )

    fold += 1

# =====================================================
# SUMMARY
# =====================================================

print("\n" + "="*60)

print(
    "Permutation Mean AUC:",
    np.mean(auc_scores)
)

print(
    "Permutation Std AUC:",
    np.std(auc_scores)
)

print("="*60)