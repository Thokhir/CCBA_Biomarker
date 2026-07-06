"""Publication-quality figure generation for Module 12 (Drug Target Discovery)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIGURE_DPI = 300


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_target_priority(priority_df: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    top = priority_df.head(top_n).iloc[::-1]
    components = ["score_predictive", "score_external", "score_biological", "score_survival"]
    colors = ["#3B6FA0", "#A0503B", "#5B8C5A", "#8A5BA0"]

    fig, ax = plt.subplots(figsize=(9, 8))
    left = pd.Series(0.0, index=top.index)
    for component, color in zip(components, colors):
        ax.barh(top["gene_name"], top[component], left=left, color=color, label=component)
        left += top[component]

    ax.set_xlabel("Composite Target Priority Score (sum of 4 normalized components)")
    ax.set_title("Target Prioritization")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)


def plot_drug_network(candidates_df: pd.DataFrame, out_path: Path) -> None:
    """Simple two-column layout: genes on the left, drugs on the right,
    lines connecting known interactions. Manual layout (no networkx),
    consistent with Module 10's approach for small, sparse graphs."""
    import numpy as np

    genes = candidates_df["gene_name"].unique().tolist()
    drugs = candidates_df["drug_name"].unique().tolist()

    gene_y = {g: i for i, g in enumerate(genes)}
    drug_y = {d: i * len(genes) / max(len(drugs), 1) for i, d in enumerate(drugs)}

    fig, ax = plt.subplots(figsize=(10, max(6, len(drugs) * 0.3)))
    for _, row in candidates_df.iterrows():
        ax.plot([0, 1], [gene_y[row["gene_name"]], drug_y[row["drug_name"]]], color="gray", linewidth=0.8, zorder=1)

    for gene, y in gene_y.items():
        ax.scatter(0, y, s=200, color="#3B6FA0", zorder=2)
        ax.text(-0.05, y, gene, ha="right", va="center", fontsize=8)
    for drug, y in drug_y.items():
        ax.scatter(1, y, s=100, color="#A0503B", zorder=2)
        ax.text(1.05, y, drug, ha="left", va="center", fontsize=7)

    ax.set_xlim(-0.5, 1.5)
    ax.axis("off")
    ax.set_title("Gene-Drug Interaction Network (DGIdb + Open Targets)")
    fig.tight_layout()
    _save(fig, out_path)
