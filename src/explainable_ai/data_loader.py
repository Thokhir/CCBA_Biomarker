"""Model, feature-order, training-matrix, and external-cohort loading for Module 9.

Mirrors the exact data construction used by clinical_model_trainer.py (training
matrix) and external_prediction_engine.py (0-filled external feature matrices),
so SHAP/permutation explanations are computed on precisely the data the model
actually saw.
"""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

RF_MODEL_FILE = BASE_DIR / "results" / "trained_model" / "rf_model.pkl"
FEATURE_ORDER_FILE = BASE_DIR / "results" / "trained_model" / "feature_order.csv"
EXPRESSION_FILE = BASE_DIR / "data" / "processed" / "expression_logCPM.csv"
METADATA_FILE = BASE_DIR / "data" / "metadata" / "tcga_chol_metadata.csv"
EXTERNAL_PREDICTIONS_DIR = BASE_DIR / "results" / "external_predictions"

ANNOTATION_COLUMNS = [
    "sample_id", "phenotype", "final_label", "predicted_label",
    "tumor_probability", "prediction_status", "cohort",
]


@dataclass
class ExternalCohortMatrix:
    cohort: str
    annotation: pd.DataFrame
    X: pd.DataFrame
    genes_imputed: list


def load_model_and_feature_order() -> tuple:
    model = joblib.load(RF_MODEL_FILE)
    feature_order = pd.read_csv(FEATURE_ORDER_FILE)["gene_name"].tolist()
    return model, feature_order


def load_training_matrix(feature_order: list) -> tuple:
    """Reconstructs the exact 44-sample x 20-gene matrix the model was fit on.

    expression_logCPM.csv has 18 duplicate gene_name values; the real trainer
    collapses these via a groupby(...).mean() that only keeps gene_name plus
    sample columns, incidentally dropping gene_id. We drop gene_id explicitly
    instead of relying on that side effect.
    """
    expr = pd.read_csv(EXPRESSION_FILE)
    expr["gene_name"] = expr["gene_name"].astype(str).str.upper()
    expr = expr.drop(columns=["gene_id"])

    if expr["gene_name"].duplicated().any():
        sample_columns = expr.columns[1:]
        expr = expr.groupby("gene_name", as_index=False)[sample_columns].mean()

    expr = expr.set_index("gene_name")
    X = expr.loc[feature_order].T

    metadata = pd.read_csv(METADATA_FILE)
    if "label" not in metadata.columns:
        metadata["label"] = metadata["sample_type"].apply(lambda s: 1 if s == "Primary Tumor" else 0)
    label_map = dict(zip(metadata["file_id"], metadata["label"]))
    y = np.array([label_map[s] for s in X.index])

    return X, y


def discover_external_prediction_files() -> list:
    files = sorted(EXTERNAL_PREDICTIONS_DIR.glob("*_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction files found in {EXTERNAL_PREDICTIONS_DIR}")
    return files


def build_external_feature_matrix(prediction_file: Path, feature_order: list) -> ExternalCohortMatrix:
    """Reconstructs the exact 0-filled, feature-order-aligned matrix the
    prediction engine scored for this cohort, and records which genes were
    imputed (missing from the platform, filled with 0) so downstream
    explanations can flag them rather than presenting a fabricated
    contribution as if it were a real measurement.
    """
    cohort = prediction_file.stem.replace("_predictions", "")
    df = pd.read_csv(prediction_file)

    present_genes = [g for g in feature_order if g in df.columns]
    genes_imputed = [g for g in feature_order if g not in df.columns]

    X = pd.DataFrame(index=df.index, columns=feature_order, dtype=float)
    for gene in present_genes:
        X[gene] = df[gene].astype(float)
    for gene in genes_imputed:
        X[gene] = 0.0

    annotation_cols = [c for c in ANNOTATION_COLUMNS if c in df.columns]
    annotation = df[annotation_cols].copy()

    return ExternalCohortMatrix(cohort=cohort, annotation=annotation, X=X, genes_imputed=genes_imputed)


def load_all_external_matrices(feature_order: list) -> dict:
    matrices = {}
    for file in discover_external_prediction_files():
        matrix = build_external_feature_matrix(file, feature_order)
        matrices[matrix.cohort] = matrix
    return matrices


def pool_external_matrices(matrices: dict) -> tuple:
    X_pooled = pd.concat([m.X for m in matrices.values()], ignore_index=True)
    annotation_pooled = pd.concat(
        [m.annotation.assign(cohort=m.cohort) for m in matrices.values()],
        ignore_index=True,
    )

    label_col = "final_label" if "final_label" in annotation_pooled.columns else "phenotype"
    raw = annotation_pooled[label_col].astype(str).str.strip().str.lower()
    label_mapping = {
        "tumor": 1, "primary tumor": 1, "cancer": 1, "cholangiocarcinoma": 1, "cca": 1, "1": 1, "true": 1,
        "normal": 0, "healthy": 0, "control": 0, "adjacent": 0, "0": 0, "false": 0,
    }
    y_pooled = raw.map(label_mapping).astype(int).to_numpy()

    return X_pooled, y_pooled, annotation_pooled
