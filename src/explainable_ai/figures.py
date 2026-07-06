"""Publication-quality figure generation for Module 9 (Explainable AI)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

FIGURE_DPI = 300


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_global_bar(importance_df: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    top = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top["gene_name"], top["mean_abs_shap"], color="#3B6FA0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (SHAP)")
    fig.tight_layout()
    _save(fig, out_path)


def plot_beeswarm(explanation: shap.Explanation, feature_names: list, out_path: Path) -> None:
    values = np.array(explanation.values)
    if values.ndim == 3:
        values = values[..., 1]
    data = np.array(explanation.data)
    fig = plt.figure(figsize=(8, 8))
    shap.summary_plot(values, data, feature_names=feature_names, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    _save(fig, out_path)


def plot_waterfall(shap_row: np.ndarray, base_value: float, data_row: np.ndarray,
                    feature_names: list, out_path: Path) -> None:
    explanation = shap.Explanation(
        values=shap_row, base_values=base_value, data=data_row, feature_names=feature_names,
    )
    fig = plt.figure(figsize=(8, 6))
    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    _save(fig, out_path)


def plot_permutation_bar(importance_df: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    top = importance_df.sort_values("mean_importance_auc", ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top["gene_name"], top["mean_importance_auc"],
            xerr=top["std_importance_auc"], color="#A0503B")
    ax.set_xlabel("Permutation Importance (ROC-AUC drop)")
    ax.set_title("Permutation Importance (External Cohorts)")
    fig.tight_layout()
    _save(fig, out_path)
