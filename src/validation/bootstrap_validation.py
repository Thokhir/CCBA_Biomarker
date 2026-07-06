import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

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

signature_file = (
    BASE_DIR /
    "results" /
    "consensus_biomarker_signature.csv"
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

signature_df = pd.read_csv(
    signature_file
)

# =====================================================
# LABELS
# =====================================================

if "label" not in metadata_df.columns:

    metadata_df["label"] = metadata_df[
        "sample_type"
    ].apply(
        lambda x:
        1 if x == "Primary Tumor"
        else 0
    )

sample_labels = {}

for _, row in metadata_df.iterrows():

    sample_labels[
        row["file_id"]
    ] = row["label"]

# =====================================================
# SIGNATURE GENES
# =====================================================

signature_genes = set(
    signature_df["gene_name"]
)

panel_expression = expression_df[
    expression_df["gene_name"].isin(
        signature_genes
    )
]

print(
    "Signature Genes:",
    len(panel_expression)
)

# =====================================================
# FEATURE MATRIX
# =====================================================

X = panel_expression.iloc[:, 2:].T

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
# BOOTSTRAP SETTINGS
# =====================================================

N_BOOTSTRAPS = 1000

accuracy_scores = []
auc_scores = []

rng = np.random.default_rng(
    seed=42
)

# =====================================================
# BOOTSTRAP LOOP
# =====================================================

for iteration in range(
    N_BOOTSTRAPS
):

    train_idx = rng.choice(
        len(X),
        size=len(X),
        replace=True
    )

    test_idx = np.setdiff1d(
        np.arange(len(X)),
        np.unique(train_idx)
    )

    if len(test_idx) < 2:
        continue

    X_train = X.iloc[train_idx]
    y_train = y[train_idx]

    X_test = X.iloc[test_idx]
    y_test = y[test_idx]

    if len(np.unique(y_test)) < 2:
        continue

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    accuracy_scores.append(
        accuracy
    )

    auc_scores.append(
        auc
    )

# =====================================================
# SUMMARY
# =====================================================

accuracy_scores = np.array(
    accuracy_scores
)

auc_scores = np.array(
    auc_scores
)

print("\n")
print("="*70)
print("BOOTSTRAP VALIDATION SUMMARY")
print("="*70)

print(
    "Valid Iterations:",
    len(auc_scores)
)

print()

print(
    f"Mean Accuracy: {np.mean(accuracy_scores):.4f}"
)

print(
    f"Std Accuracy : {np.std(accuracy_scores):.4f}"
)

print()

print(
    f"Mean AUC: {np.mean(auc_scores):.4f}"
)

print(
    f"Std AUC : {np.std(auc_scores):.4f}"
)

# =====================================================
# 95% CONFIDENCE INTERVAL
# =====================================================

auc_ci_lower = np.percentile(
    auc_scores,
    2.5
)

auc_ci_upper = np.percentile(
    auc_scores,
    97.5
)

acc_ci_lower = np.percentile(
    accuracy_scores,
    2.5
)

acc_ci_upper = np.percentile(
    accuracy_scores,
    97.5
)

print("\n")
print("="*70)
print("95% CONFIDENCE INTERVALS")
print("="*70)

print(
    f"Accuracy CI: [{acc_ci_lower:.4f}, {acc_ci_upper:.4f}]"
)

print(
    f"AUC CI      : [{auc_ci_lower:.4f}, {auc_ci_upper:.4f}]"
)

# =====================================================
# SAVE RESULTS
# =====================================================

RESULT_DIR = (
    BASE_DIR /
    "results"
)

RESULT_DIR.mkdir(
    exist_ok=True
)

pd.DataFrame({
    "accuracy":
        accuracy_scores,
    "auc":
        auc_scores
}).to_csv(
    RESULT_DIR /
    "bootstrap_validation_results.csv",
    index=False
)

print("\nSaved:")

print(
    RESULT_DIR /
    "bootstrap_validation_results.csv"
)