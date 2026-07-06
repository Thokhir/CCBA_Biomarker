"""
=========================================================
DAY 9
Clinical Biomarker Panel Construction
=========================================================
"""

import pandas as pd
import numpy as np

from pathlib import Path
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(
    exist_ok=True
)

# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading data...")

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

print(
    "Expression Shape:",
    expression_df.shape
)

print(
    "Metadata Shape:",
    metadata_df.shape
)

# =====================================================
# CREATE LABELS
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
# BUILD ML MATRIX
# =====================================================

X = expression_df.iloc[:, 2:].T

missing_samples = [
    s
    for s in X.index
    if s not in sample_labels
]

if len(missing_samples) > 0:

    raise ValueError(
        f"Missing metadata labels:\n"
        f"{missing_samples}"
    )

y = np.array([
    sample_labels[s]
    for s in X.index
])

print("\nML Matrix Shape:")
print(X.shape)

print("\nClass Counts:")
print(
    pd.Series(y).value_counts()
)

# =====================================================
# FEATURE STABILITY ANALYSIS
# =====================================================

print("\nRunning Feature Stability Analysis...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

feature_counter = Counter()

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y),
    start=1
):

    print(
        f"Processing Fold {fold}"
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(
        X_train,
        y_train
    )

    importance = pd.DataFrame({

        "gene_name":
            expression_df["gene_name"],

        "importance":
            rf.feature_importances_
    })

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
    )

    top_genes = (
        importance
        .head(50)
        ["gene_name"]
        .tolist()
    )

    feature_counter.update(
        top_genes
    )

# =====================================================
# STABILITY TABLE
# =====================================================

stability_df = pd.DataFrame(
    feature_counter.items(),
    columns=[
        "gene_name",
        "frequency"
    ]
)

stability_df = (
    stability_df
    .sort_values(
        "frequency",
        ascending=False
    )
)

stability_df.to_csv(
    RESULT_DIR /
    "feature_stability.csv",
    index=False
)

print("\nTop Stable Biomarkers")

print(
    stability_df.head(20)
)

# =====================================================
# RFE
# =====================================================

print(
    "\nRunning Recursive Feature Elimination..."
)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

selector = RFE(
    estimator=rf,
    n_features_to_select=20,
    step=1000
)

selector.fit(
    X,
    y
)

selected_genes = (
    expression_df["gene_name"]
    [selector.support_]
)

selected_df = pd.DataFrame({
    "gene_name":
        selected_genes
})

selected_df.to_csv(
    RESULT_DIR /
    "rfe_selected_genes.csv",
    index=False
)

print(
    "\nRFE Selected Genes:"
)

print(
    selected_df.head(20)
)

# =====================================================
# LOAD DAY 7 + DAY 8 RESULTS
# =====================================================

dea_file = RESULT_DIR / "candidate_biomarkers.csv"
rf_file = RESULT_DIR / "rf_biomarkers.csv"
xgb_file = RESULT_DIR / "xgb_biomarkers.csv"

if (
    dea_file.exists()
    and rf_file.exists()
    and xgb_file.exists()
):

    print("\nBuilding Consensus Clinical Panel...")

    dea = pd.read_csv(dea_file)
    rf_df = pd.read_csv(rf_file)
    xgb_df = pd.read_csv(xgb_file)

    # ----------------------------------------
    # CASE 1
    # DEA returned no significant genes
    # ----------------------------------------

    if len(dea) == 0:

        print(
            "\nWARNING:"
            "\nNo significant DEA genes found."
            "\nUsing top RF + XGB overlap instead."
        )

        rf_top = rf_df.head(200)

        xgb_top = xgb_df.head(200)

        overlap = set(
            rf_top["gene_name"]
        ).intersection(
            set(
                xgb_top["gene_name"]
            )
        )

        final_panel = pd.DataFrame({
            "gene_name": list(overlap)
        })

        final_panel.to_csv(
            RESULT_DIR /
            "clinical_biomarker_panel.csv",
            index=False
        )

        print(
            "\nClinical Panel Size:",
            len(final_panel)
        )

        print(
            final_panel.head(20)
        )

    else:

        # ----------------------------------------
        # NORMAL CONSENSUS
        # ----------------------------------------

        dea["gene_name"] = (
            dea["gene_name"]
            .astype(str)
            .str.upper()
        )

        rf_df["gene_name"] = (
            rf_df["gene_name"]
            .astype(str)
            .str.upper()
        )

        xgb_df["gene_name"] = (
            xgb_df["gene_name"]
            .astype(str)
            .str.upper()
        )

        dea["dea_rank"] = np.arange(
            1,
            len(dea) + 1
        )

        rf_df["rf_rank"] = np.arange(
            1,
            len(rf_df) + 1
        )

        xgb_df["xgb_rank"] = np.arange(
            1,
            len(xgb_df) + 1
        )

        panel = dea.merge(
            rf_df[
                [
                    "gene_name",
                    "rf_rank"
                ]
            ],
            on="gene_name",
            how="inner"
        )

        panel = panel.merge(
            xgb_df[
                [
                    "gene_name",
                    "xgb_rank"
                ]
            ],
            on="gene_name",
            how="inner"
        )

        panel["consensus_score"] = (
            panel["dea_rank"]
            +
            panel["rf_rank"]
            +
            panel["xgb_rank"]
        )

        panel = panel.sort_values(
            "consensus_score"
        )

        final_panel = panel.head(20)

        final_panel.to_csv(
            RESULT_DIR /
            "clinical_biomarker_panel.csv",
            index=False
        )

        print(
            "\nTop Clinical Biomarkers"
        )

        print(
            final_panel.head(20)
        )

else:

    print(
        "\nRequired files missing."
    )