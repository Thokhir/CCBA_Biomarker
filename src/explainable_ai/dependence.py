"""SHAP dependence analysis for the top globally-important genes.

Shows how a gene's SHAP contribution varies with its own expression value
across the training population - complements the global bar/beeswarm plots
(which show overall magnitude) with the shape of each gene's relationship
to the model's output.
"""
import pandas as pd
import shap


def select_top_genes(importance_df: pd.DataFrame, top_n: int = 5) -> list:
    return importance_df.head(top_n)["gene_name"].tolist()
