"""
=========================================================
DAY 8 - AI DRIVEN BIOMARKER DISCOVERY
Cholangiocarcinoma Biomarker Platform

Random Forest
XGBoost
SHAP
Consensus Biomarkers
=========================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score
)

# --------------------------------------------------
# Optional Packages
# --------------------------------------------------

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ModuleNotFoundError:
    xgb_available = False
    print("WARNING: xgboost not installed.")

try:
    import shap
    shap_available = True
except ModuleNotFoundError:
    shap_available = False
    print("WARNING: shap not installed.")

# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]

EXPRESSION_FILE = (
    BASE_DIR /
    "data" /
    "processed" /
    "expression_logCPM.csv"
)

METADATA_FILE = (
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

# ==================================================
# LOAD DATA
# ==================================================

print("\nLoading expression matrix...")

expression_df = pd.read_csv(
    EXPRESSION_FILE
)

print("Expression Shape:")
print(expression_df.shape)

print("\nLoading metadata...")

metadata_df = pd.read_csv(
    METADATA_FILE
)

print("Metadata Shape:")
print(metadata_df.shape)

# ==================================================
# CREATE LABELS
# ==================================================

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

# ==================================================
# MACHINE LEARNING MATRIX
# ==================================================

print("\nPreparing ML matrix...")

X = expression_df.iloc[:, 2:].T

print("ML Matrix Shape:")
print(X.shape)

missing_samples = [
    sample
    for sample in X.index
    if sample not in sample_labels
]

if len(missing_samples) > 0:

    raise ValueError(
        f"Samples missing in metadata:\n"
        f"{missing_samples}"
    )

y = np.array(
    [
        sample_labels[sample]
        for sample in X.index
    ]
)

print("\nClass Distribution")

print(
    pd.Series(y).value_counts()
)

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:")
print(X_train.shape)

print("Test Shape:")
print(X_test.shape)

# ==================================================
# RANDOM FOREST
# ==================================================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)

rf_predictions = rf.predict(
    X_test
)

rf_probabilities = rf.predict_proba(
    X_test
)[:, 1]

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

rf_auc = roc_auc_score(
    y_test,
    rf_probabilities
)

print("\nRandom Forest Results")
print("----------------------")
print("Accuracy:", rf_accuracy)
print("AUC:", rf_auc)

assert len(expression_df) == len(
    rf.feature_importances_
)

rf_importance = pd.DataFrame({
    "gene_name":
        expression_df["gene_name"],
    "importance":
        rf.feature_importances_
})

rf_importance = rf_importance.sort_values(
    "importance",
    ascending=False
)

rf_importance.to_csv(
    RESULT_DIR /
    "rf_biomarkers.csv",
    index=False
)

print("\nTop 20 RF Biomarkers")

print(
    rf_importance.head(20)
)

# ==================================================
# XGBOOST
# ==================================================

xgb_auc = None
xgb_accuracy = None
xgb_importance = None
consensus = set()

if xgb_available:

    print("\nTraining XGBoost...")

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    xgb.fit(
        X_train,
        y_train
    )

    xgb_predictions = xgb.predict(
        X_test
    )

    xgb_probs = xgb.predict_proba(
        X_test
    )[:, 1]

    xgb_accuracy = accuracy_score(
        y_test,
        xgb_predictions
    )

    xgb_auc = roc_auc_score(
        y_test,
        xgb_probs
    )

    print("\nXGBoost Results")
    print("----------------------")
    print("Accuracy:", xgb_accuracy)
    print("AUC:", xgb_auc)

    xgb_importance = pd.DataFrame({
        "gene_name":
            expression_df["gene_name"],
        "importance":
            xgb.feature_importances_
    })

    xgb_importance = xgb_importance.sort_values(
        "importance",
        ascending=False
    )

    xgb_importance.to_csv(
        RESULT_DIR /
        "xgb_biomarkers.csv",
        index=False
    )

    print("\nTop 20 XGB Biomarkers")

    print(
        xgb_importance.head(20)
    )

    # ==========================================
    # SHAP
    # ==========================================

    if shap_available:

        print("\nRunning SHAP...")

        explainer = shap.TreeExplainer(
            xgb
        )

        shap_values = explainer.shap_values(
            X_test
        )

        shap.summary_plot(
            shap_values,
            X_test,
            show=False
        )

        plt.savefig(
            RESULT_DIR /
            "shap_summary.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "SHAP summary saved."
        )

    rf_top = set(
        rf_importance.head(100)[
            "gene_name"
        ]
    )

    xgb_top = set(
        xgb_importance.head(100)[
            "gene_name"
        ]
    )

    consensus = rf_top.intersection(
        xgb_top
    )

else:

    print(
        "\nSkipping XGBoost."
    )

# ==================================================
# CONSENSUS BIOMARKERS
# ==================================================

consensus_df = pd.DataFrame({
    "gene_name":
        list(consensus)
})

consensus_df.to_csv(
    RESULT_DIR /
    "consensus_biomarkers.csv",
    index=False
)

print(
    "\nConsensus Biomarkers:",
    len(consensus)
)

# ==================================================
# MODEL PERFORMANCE
# ==================================================

metrics_df = pd.DataFrame({
    "Model": [
        "RandomForest",
        "XGBoost"
    ],
    "Accuracy": [
        rf_accuracy,
        xgb_accuracy
    ],
    "AUC": [
        rf_auc,
        xgb_auc
    ]
})

metrics_df.to_csv(
    RESULT_DIR /
    "model_performance.csv",
    index=False
)

# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n")
print("=" * 60)
print("DAY 8 COMPLETE")
print("=" * 60)

print(
    "RF Biomarkers:",
    len(rf_importance)
)

if xgb_importance is not None:

    print(
        "XGB Biomarkers:",
        len(xgb_importance)
    )

print(
    "Consensus Biomarkers:",
    len(consensus)
)

print(
    "\nResults Folder:"
)

print(RESULT_DIR)

print("\nPipeline Finished Successfully")