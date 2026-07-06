"""Publication-quality figure generation for Module 11 (Survival Analysis)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURE_DPI = 300


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_km_curve(kmf_high, kmf_low, gene: str, p_value: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    kmf_high.plot_survival_function(ax=ax, color="#A0503B")
    kmf_low.plot_survival_function(ax=ax, color="#3B6FA0")
    ax.set_xlabel("Days")
    ax.set_ylabel("Survival Probability")
    ax.set_title(f"{gene} (logrank p={p_value:.4f})")
    fig.tight_layout()
    _save(fig, out_path)


def plot_hazard_forest(cox_results: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    top = cox_results.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    y = np.arange(len(top))
    ax.errorbar(
        top["hazard_ratio"], y,
        xerr=[top["hazard_ratio"] - top["hr_lower_95"], top["hr_upper_95"] - top["hazard_ratio"]],
        fmt="o", color="#3B6FA0", ecolor="gray", capsize=3,
    )
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(top["gene_name"])
    ax.set_xlabel("Hazard Ratio (95% CI)")
    ax.set_title("Univariate Cox Hazard Ratios (standardized expression)")
    fig.tight_layout()
    _save(fig, out_path)


def plot_time_roc(roc_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    valid = roc_df.dropna(subset=["auc"])
    ax.plot(valid["horizon_days"], valid["auc"], marker="o", color="#3B6FA0")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (AUC=0.5)")
    ax.set_xlabel("Follow-up Horizon (days)")
    ax.set_ylabel("Landmark-time ROC-AUC")
    ax.set_title("Time-Dependent Discrimination (Risk Score)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)


def plot_nomogram(scales: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3 + len(scales)))
    y_positions = np.arange(len(scales), 0, -1)

    for y, (covariate, scale) in zip(y_positions, scales.items()):
        ax.plot([0, scale["points_max"]], [y, y], color="black", linewidth=2)
        ax.text(-5, y, covariate, ha="right", va="center", fontsize=10)
        low_label = scale["value_min"] if scale["direction"] == "positive" else scale["value_max"]
        high_label = scale["value_max"] if scale["direction"] == "positive" else scale["value_min"]
        ax.text(0, y + 0.15, f"{low_label:.1f}", ha="center", fontsize=8)
        ax.text(scale["points_max"], y + 0.15, f"{high_label:.1f}", ha="center", fontsize=8)

    ax.set_xlim(-30, 105)
    ax.set_ylim(0.5, len(scales) + 0.5)
    ax.set_xlabel("Points")
    ax.set_title("Simplified Nomogram (point-scale per covariate)")
    ax.get_yaxis().set_visible(False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    _save(fig, out_path)
