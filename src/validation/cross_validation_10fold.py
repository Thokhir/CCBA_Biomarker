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

expression_df = pd.read_csv(expression_file)

metadata_df = pd.read_csv(metadata_file)

panel_df = pd.read_csv(panel_file)

# =====================================================
# LABELS
# =====================================================

if "label" not in metadata_df.columns:

    metadata_df["label"] = metadata_df[
        "sample_type"
    ].apply(
        lambda x: 1 if x == "Primary Tumor" else 0
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
    expression_df["gene_name"].isin(
        panel_genes
    )
]

print(
    "Panel Genes:",
    len(panel_expression)
)

# =====================================================
# FEATURE MATRIX
# =====================================================

X = panel_expression.iloc[:, 2:].T

y = np.array([
    sample_labels[sample]
    for sample in X.index
])

print(
    "Samples:",
    len(X)
)

print(
    "Features:",
    X.shape[1]
)

print(
    "Tumor Samples:",
    np.sum(y == 1)
)

print(
    "Normal Samples:",
    np.sum(y == 0)
)

# =====================================================
# 10-FOLD CV
# =====================================================

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

accuracy_scores = []
auc_scores = []

fold_results = []

fold = 1

for train_idx, test_idx in cv.split(X, y):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

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

    fold_results.append({
        "Fold": fold,
        "Accuracy": accuracy,
        "AUC": auc
    })

    print("\n" + "="*40)
    print(f"Fold {fold}")
    print("="*40)
    print(f"Accuracy = {accuracy:.4f}")
    print(f"AUC      = {auc:.4f}")

    fold += 1

# =====================================================
# SUMMARY
# =====================================================

print("\n")
print("="*60)
print("10-FOLD CROSS VALIDATION SUMMARY")
print("="*60)

print(
    f"Mean Accuracy : {np.mean(accuracy_scores):.4f}"
)

print(
    f"Std Accuracy  : {np.std(accuracy_scores):.4f}"
)

print(
    f"Min Accuracy  : {np.min(accuracy_scores):.4f}"
)

print(
    f"Max Accuracy  : {np.max(accuracy_scores):.4f}"
)

print()

print(
    f"Mean AUC      : {np.mean(auc_scores):.4f}"
)

print(
    f"Std AUC       : {np.std(auc_scores):.4f}"
)

print(
    f"Min AUC       : {np.min(auc_scores):.4f}"
)

print(
    f"Max AUC       : {np.max(auc_scores):.4f}"
)

# =====================================================
# SAVE RESULTS
# =====================================================

RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(
    exist_ok=True
)

pd.DataFrame(
    fold_results
).to_csv(
    RESULT_DIR /
    "10fold_cross_validation_results.csv",
    index=False
)

print(
    "\nResults saved:"
)

print(
    RESULT_DIR /
    "10fold_cross_validation_results.csv"
)