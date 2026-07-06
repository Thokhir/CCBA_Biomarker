"""Publication-quality figure generation for external validation analytics."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:
    import config


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(roc_tables: dict[str, pd.DataFrame], auc_by_cohort: dict[str, float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=config.FIGSIZE_ROC)
    for cohort, df in roc_tables.items():
        ax.plot(df["False_Positive_Rate"], df["True_Positive_Rate"], linewidth=2,
                label=f"{cohort} (AUC={auc_by_cohort[cohort]:.3f})")
    ax.plot([0, 1], [0, 1], "--", linewidth=1, color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("External Validation ROC Curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


def plot_pr_curves(pr_tables: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=config.FIGSIZE_PR)
    for cohort, df in pr_tables.items():
        ax.plot(df["Recall"], df["Precision"], linewidth=2, label=cohort)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


def plot_calibration_curves(cal_tables: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=config.FIGSIZE_CALIBRATION)
    for cohort, df in cal_tables.items():
        ax.plot(df["Mean_Predicted_Probability"], df["Fraction_of_Positives"],
                marker="o", linewidth=2, label=cohort)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration Curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


def plot_confusion_matrix(cohort: str, tn: int, fp: int, fn: int, tp: int, out_path: Path) -> None:
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=config.FIGSIZE_CONFUSION)
    im = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], ["Normal", "Tumor"])
    ax.set_yticks([0, 1], ["Normal", "Tumor"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(cohort)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                    fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)


def plot_bootstrap_auc_histogram(cohort: str, auc_values: list, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=config.FIGSIZE_BOOTSTRAP_HIST)
    ax.hist(auc_values, bins=config.BOOTSTRAP_HIST_BINS)
    ax.set_xlabel("ROC AUC")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{cohort} Bootstrap ROC AUC")
    fig.tight_layout()
    _save(fig, out_path)
