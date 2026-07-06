"""Wraps model loading, preprocessing, prediction, and SHAP explanation for
the REST API.

Reuses src/dashboard/components/{preprocessing,predictor,shap_runtime}.py
directly - despite living under the dashboard's directory, none of those
three modules import streamlit (confirmed: only pandas/numpy/shap/
dataclasses), so they are genuinely framework-agnostic and importable here
without pulling in a Streamlit dependency for the API service.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "src" / "dashboard"))

from components.preprocessing import build_patient_matrix
from components.predictor import predict_patient
from components.shap_runtime import explain_patient

RESULTS_DIR = BASE_DIR / "results"


class PredictionService:
    """Loads the model once; reused across all requests (mirrors the
    dashboard's @st.cache_resource model-loading pattern, but framework-
    agnostic here since FastAPI has no equivalent decorator)."""

    def __init__(self):
        self.model = joblib.load(RESULTS_DIR / "trained_model" / "rf_model.pkl")
        self.feature_order = pd.read_csv(
            RESULTS_DIR / "trained_model" / "feature_order.csv"
        )["gene_name"].tolist()

    def _validate_gene_expression(self, gene_expression: dict) -> None:
        matched = [g for g in self.feature_order if g in gene_expression]
        if not matched:
            raise ValueError(
                f"None of the {len(self.feature_order)} clinical biomarker panel genes were found "
                "in gene_expression. Expected gene symbols include: " + ", ".join(self.feature_order[:5]) + ", ..."
            )

    def predict(self, sample_id: str, gene_expression: dict) -> dict:
        self._validate_gene_expression(gene_expression)
        raw_row = pd.Series(gene_expression)
        patient_matrix = build_patient_matrix(raw_row, self.feature_order, sample_id)
        prediction = predict_patient(self.model, patient_matrix.X)

        return {
            "sample_id": sample_id,
            "predicted_label": prediction.predicted_label,
            "predicted_class": "Tumor" if prediction.predicted_label == 1 else "Normal",
            "tumor_probability": prediction.tumor_probability,
            "confidence": prediction.confidence,
            "genes_provided": patient_matrix.genes_provided,
            "genes_imputed": patient_matrix.genes_imputed,
        }

    def explain(self, sample_id: str, gene_expression: dict) -> dict:
        self._validate_gene_expression(gene_expression)
        raw_row = pd.Series(gene_expression)
        patient_matrix = build_patient_matrix(raw_row, self.feature_order, sample_id)
        explanation = explain_patient(
            self.model, patient_matrix.X, self.feature_order, patient_matrix.genes_imputed
        )

        contributions = [
            {"gene_name": row["gene_name"], "shap_value": row["shap_value"], "is_imputed": row["is_imputed"]}
            for _, row in explanation.contributions.iterrows()
        ]
        return {
            "sample_id": sample_id,
            "base_value": explanation.base_value,
            "predicted_probability": explanation.predicted_probability,
            "contributions": contributions,
        }
