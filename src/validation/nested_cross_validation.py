import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import (
    StratifiedKFold
)

from sklearn.ensemble import (
    RandomForestClassifier
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

# =====================================================
# LOAD DATA
# =====================================================

expression_df = pd.read_csv(
    expression_file
)

metadata_df = pd.read_csv(
    metadata_file
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
# FEATURE MATRIX
# =====================================================

X = expression_df.iloc[:,2:].T

y = np.array([
    sample_labels[s]
    for s in X.index
])

gene_names = (
    expression_df["gene_name"]
    .fillna(expression_df["gene_id"])
    .values
)

print("Samples:", X.shape[0])
print("Genes:", X.shape[1])

# =====================================================
# OUTER CV
# =====================================================

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

accuracy_scores = []
auc_scores = []

feature_frequency = {}

fold = 1

for train_idx, test_idx in outer_cv.split(X, y):

    print("\n" + "="*60)
    print(f"OUTER FOLD {fold}")
    print("="*60)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # ==========================================
    # FEATURE SELECTION ONLY ON TRAINING DATA
    # ==========================================

    selector = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )

    selector.fit(
        X_train,
        y_train
    )

    importance_df = pd.DataFrame({
        "gene": gene_names,
        "importance": selector.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    top_features = (
        importance_df
        .head(20)
        ["gene"]
        .tolist()
    )

    # Track stability
    for gene in top_features:

        feature_frequency[gene] = (
            feature_frequency.get(gene, 0)
            + 1
        )

    selected_columns = []

    for gene in top_features:

        gene_idx = np.where(
            gene_names == gene
        )[0][0]

        selected_columns.append(
            gene_idx
        )

    X_train_selected = (
        X_train.iloc[:, selected_columns]
    )

    X_test_selected = (
        X_test.iloc[:, selected_columns]
    )

    # ==========================================
    # MODEL TRAINING
    # ==========================================

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_selected,
        y_train
    )

    predictions = model.predict(
        X_test_selected
    )

    probabilities = model.predict_proba(
        X_test_selected
    )[:,1]

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

    print(
        f"Accuracy = {accuracy:.4f}"
    )

    print(
        f"AUC = {auc:.4f}"
    )

    fold += 1

# =====================================================
# SUMMARY
# =====================================================

print("\n")
print("="*70)
print("NESTED CROSS VALIDATION SUMMARY")
print("="*70)

print(
    f"Mean Accuracy = {np.mean(accuracy_scores):.4f}"
)

print(
    f"Std Accuracy  = {np.std(accuracy_scores):.4f}"
)

print(
    f"Mean AUC      = {np.mean(auc_scores):.4f}"
)

print(
    f"Std AUC       = {np.std(auc_scores):.4f}"
)

# =====================================================
# FEATURE STABILITY
# =====================================================

stability_df = pd.DataFrame({

    "gene_name":
        list(feature_frequency.keys()),

    "selection_count":
        list(feature_frequency.values())
})

stability_df = stability_df.sort_values(
    "selection_count",
    ascending=False
)

RESULT_DIR = (
    BASE_DIR /
    "results"
)

RESULT_DIR.mkdir(
    exist_ok=True
)

stability_df.to_csv(
    RESULT_DIR /
    "nested_cv_feature_stability.csv",
    index=False
)

print("\nTop Stable Biomarkers")

print(
    stability_df.head(20)
)

print(
    "\nSaved:"
)

print(
    RESULT_DIR /
    "nested_cv_feature_stability.csv"
)