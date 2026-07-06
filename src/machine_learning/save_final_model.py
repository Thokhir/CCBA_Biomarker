from pathlib import Path
import json
import pickle

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

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

MODEL_DIR = (
    BASE_DIR /
    "results" /
    "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

expression_df = pd.read_csv(expression_file)
metadata_df = pd.read_csv(metadata_file)
panel_df = pd.read_csv(panel_file)

if "label" not in metadata_df.columns:

    metadata_df["label"] = metadata_df[
        "sample_type"
    ].apply(
        lambda x: 1 if x == "Primary Tumor" else 0
    )

#######################################################
# Build sample labels
#######################################################

sample_labels = dict(
    zip(
        metadata_df["file_id"],
        metadata_df["label"]
    )
)

#######################################################
# Final Biomarker Panel
#######################################################

panel_genes = panel_df[
    "gene_name"
].tolist()

print("\nFinal Biomarkers")

print(panel_genes)

#######################################################
# Keep only selected genes
#######################################################

selected = expression_df[
    expression_df["gene_name"].isin(
        panel_genes
    )
].copy()

selected = selected.sort_values(
    "gene_name"
)

#######################################################
# Feature Matrix
#######################################################

X = (
    selected
    .iloc[:,2:]
    .T
)

X.columns = selected[
    "gene_name"
].tolist()

#######################################################
# Labels
#######################################################

y = [
    sample_labels[s]
    for s in X.index
]

#######################################################
# Train Final Model
#######################################################

rf = RandomForestClassifier(

    n_estimators=1000,

    random_state=42,

    n_jobs=-1

)

rf.fit(X,y)

#######################################################
# Save Model
#######################################################

with open(
    MODEL_DIR /
    "rf_model.pkl",
    "wb"
) as f:

    pickle.dump(
        rf,
        f
    )

#######################################################
# Save Feature Order
#######################################################

pd.DataFrame({

    "gene_name":X.columns

}).to_csv(

    MODEL_DIR /

    "feature_order.csv",

    index=False

)

#######################################################
# Save Metadata
#######################################################

metadata={

    "algorithm":"RandomForest",

    "n_estimators":1000,

    "random_state":42,

    "training_samples":len(X),

    "features":len(X.columns),

    "panel_genes":panel_genes

}

with open(

    MODEL_DIR /

    "model_metadata.json",

    "w"

) as f:

    json.dump(

        metadata,

        f,

        indent=4

    )

print("\n")

print("="*60)

print("MODEL SAVED")

print("="*60)

print("Model")

print(
    MODEL_DIR /
    "rf_model.pkl"
)

print()

print("Feature Order")

print(
    MODEL_DIR /
    "feature_order.csv"
)

print()

print("Metadata")

print(
    MODEL_DIR /
    "model_metadata.json"
)