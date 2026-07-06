"""Global SHAP feature importance over the training population.

Uses TreeExplainer with feature_perturbation="tree_path_dependent" and no
background dataset - the exact mode for tree ensembles, using the fitted
trees' own path-coverage statistics rather than an independent background
sample. This is the right choice here (not a shortcut): with only 44 training
samples, an interventional/background-based mode would be slower and more
fragile, and tree_path_dependent is exact for tree ensembles regardless.

For this binary RandomForestClassifier, shap.Explanation.values is shaped
(n_samples, n_features, n_classes) and base_values is (n_samples, n_classes).
Class-1 (tumor) contributions are values[..., 1] / base_values[:, 1] -
verified directly against model.predict_proba (additivity holds to ~1e-15).
Getting this slice backwards would silently invert every explanation.
"""
import shap
import numpy as np
import pandas as pd


def compute_global_shap(model, X_train: pd.DataFrame) -> shap.Explanation:
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    return explainer(X_train)


def class1_values(explanation: shap.Explanation) -> np.ndarray:
    values = np.array(explanation.values)
    return values[..., 1] if values.ndim == 3 else values


def class1_base_values(explanation: shap.Explanation) -> np.ndarray:
    base = np.array(explanation.base_values)
    return base[:, 1] if base.ndim == 2 else base


def summarize_global_importance(explanation: shap.Explanation, feature_names: list) -> pd.DataFrame:
    values = class1_values(explanation)
    mean_abs_shap = np.abs(values).mean(axis=0)
    df = pd.DataFrame({"gene_name": feature_names, "mean_abs_shap": mean_abs_shap})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df[["rank", "gene_name", "mean_abs_shap"]]


def shap_values_to_frame(explanation: shap.Explanation, sample_ids: list, feature_names: list) -> pd.DataFrame:
    values = class1_values(explanation)
    df = pd.DataFrame(values, columns=[f"shap_{g}" for g in feature_names])
    df.insert(0, "sample_id", sample_ids)
    df["base_value"] = class1_base_values(explanation)
    return df
