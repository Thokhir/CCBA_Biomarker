"""Live, single-row SHAP explanation for a freshly uploaded patient.

Uses shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
exactly as in src/explainable_ai/global_explainer.py/local_explainer.py -
cheap (sub-second) for one row, no background dataset needed for tree
ensembles. Imports the class-1 slicing helpers directly from
global_explainer rather than redefining them: getting that slice backwards
(class 0 vs class 1) would silently invert every explanation, and it was
already verified correct (additivity to ~1e-15) in Module 9.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from explainable_ai.global_explainer import class1_values, class1_base_values


@dataclass
class PatientExplanation:
    contributions: pd.DataFrame  # columns: gene_name, shap_value, is_imputed
    base_value: float
    predicted_probability: float


def explain_patient(model, patient_x: pd.DataFrame, feature_order: list, genes_imputed: list) -> PatientExplanation:
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    explanation = explainer(patient_x)

    values = class1_values(explanation)[0]
    base = float(class1_base_values(explanation)[0])
    predicted_probability = float(base + values.sum())

    contributions = pd.DataFrame({
        "gene_name": feature_order,
        "shap_value": values,
        "is_imputed": [g in genes_imputed for g in feature_order],
    })
    contributions["abs_shap_value"] = contributions["shap_value"].abs()
    contributions = contributions.sort_values("abs_shap_value", ascending=False).reset_index(drop=True)

    return PatientExplanation(contributions=contributions, base_value=base,
                               predicted_probability=predicted_probability)
