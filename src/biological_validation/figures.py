"""Publication-quality figure generation for Module 10 (Biological Validation)."""
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


def _neg_log10_padj(df: pd.DataFrame) -> np.ndarray:
    padj = df["Adjusted P-value"].to_numpy(dtype=float)
    padj = np.clip(padj, 1e-300, None)
    return -np.log10(padj)


def plot_enrichment_barplot(df: pd.DataFrame, out_path: Path, title: str, top_n: int = 15) -> None:
    top = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top["Term"], _neg_log10_padj(top), color="#3B6FA0")
    ax.axvline(-np.log10(0.05), color="red", linestyle="--", linewidth=1, label="padj = 0.05")
    ax.set_xlabel("-log10(Adjusted P-value)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)


def plot_go_dotplot(df: pd.DataFrame, out_path: Path, title: str, top_n: int = 15) -> None:
    top = df.head(top_n).iloc[::-1].copy()
    top["overlap_count"] = top["Overlap"].str.split("/").str[0].astype(int)

    fig, ax = plt.subplots(figsize=(9, 8))
    scatter = ax.scatter(_neg_log10_padj(top), range(len(top)), s=top["overlap_count"] * 40,
                          c=top["Adjusted P-value"], cmap="viridis_r")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["Term"])
    ax.axvline(-np.log10(0.05), color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("-log10(Adjusted P-value)")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label="Adjusted P-value")
    fig.tight_layout()
    _save(fig, out_path)


def plot_string_network(network_df: pd.DataFrame, degree_df: pd.DataFrame, out_path: Path) -> None:
    genes = degree_df["gene_name"].tolist()
    n = len(genes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = {gene: (np.cos(a), np.sin(a)) for gene, a in zip(genes, angles)}

    fig, ax = plt.subplots(figsize=(9, 9))
    for _, row in network_df.iterrows():
        x1, y1 = positions[row["preferredName_A"]]
        x2, y2 = positions[row["preferredName_B"]]
        ax.plot([x1, x2], [y1, y2], color="gray", linewidth=1.5, zorder=1)

    degree_map = dict(zip(degree_df["gene_name"], degree_df["string_degree"]))
    for gene, (x, y) in positions.items():
        is_hub = degree_map.get(gene, 0) >= 2
        ax.scatter(x, y, s=400, color="#A0503B" if is_hub else "#3B6FA0", zorder=2, edgecolors="black")
        ax.annotate(gene, (x, y), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"STRING PPI Network ({len(network_df)} edges, {n} genes)\nHub genes (degree >= 2) in red")
    fig.tight_layout()
    _save(fig, out_path)
