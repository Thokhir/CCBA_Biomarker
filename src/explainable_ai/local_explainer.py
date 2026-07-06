"""Per-sample SHAP explanations over external validation cohorts.

Computes SHAP values for every external cohort sample (for later per-patient
lookup, e.g. by a future Streamlit app) and selects a small representative
subset - one correctly-classified tumor, one correctly-classified normal, one
misclassified case - for individual waterfall plots.

GSE32225 and GSE89749 predict entirely class 0 (Normal) on this data, so a
"correctly classified tumor" case only exists in GSE26566 - representative
cases are drawn from there.
"""
import numpy as np
import pandas as pd
import shap

try:
    from .data_loader import ExternalCohortMatrix
    from .global_explainer import class1_values, class1_base_values
except ImportError:
    from data_loader import ExternalCohortMatrix
    from global_explainer import class1_values, class1_base_values


def compute_local_shap(model, X: pd.DataFrame) -> shap.Explanation:
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    return explainer(X)


def local_shap_to_frame(explanation: shap.Explanation, matrix: ExternalCohortMatrix, feature_names: list) -> pd.DataFrame:
    values = class1_values(explanation)
    base = class1_base_values(explanation)

    df = pd.DataFrame(values, columns=[f"shap_{g}" for g in feature_names])
    df.insert(0, "sample_id", matrix.annotation["sample_id"].to_numpy())
    df.insert(1, "cohort", matrix.cohort)
    df["base_value"] = base
    df["predicted_label"] = matrix.annotation["predicted_label"].to_numpy()
    df["tumor_probability"] = matrix.annotation["tumor_probability"].to_numpy()
    df["prediction_status"] = matrix.annotation["prediction_status"].to_numpy()
    df["genes_imputed"] = ";".join(matrix.genes_imputed) if matrix.genes_imputed else ""
    return df


def select_representative_cases(matrix: ExternalCohortMatrix) -> pd.DataFrame:
    ann = matrix.annotation.reset_index(drop=True)
    rows = []

    correct_tumor = ann[(ann["prediction_status"] == "Correct") & (ann["predicted_label"] == 1)]
    if not correct_tumor.empty:
        rows.append((correct_tumor.index[0], "correct_tumor"))

    correct_normal = ann[(ann["prediction_status"] == "Correct") & (ann["predicted_label"] == 0)]
    if not correct_normal.empty:
        rows.append((correct_normal.index[0], "correct_normal"))

    incorrect = ann[ann["prediction_status"] == "Incorrect"]
    if not incorrect.empty:
        idx = (incorrect["tumor_probability"] - 0.5).abs().idxmin()
        rows.append((idx, "misclassified"))

    result = ann.loc[[r[0] for r in rows]].copy()
    result["role"] = [r[1] for r in rows]
    result["row_position"] = [r[0] for r in rows]
    return result.reset_index(drop=True)
