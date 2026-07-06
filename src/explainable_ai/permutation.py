"""Model-agnostic permutation importance, computed on held-out external
cohort data rather than training data.

Permutation importance on training data for a bagging model like Random
Forest reuses in-bag samples the trees have already seen and is known to be
optimistically biased; pooled external cohort data gives an honest, held-out
estimate. Scored primarily by ROC-AUC rather than accuracy, since two of the
three external cohorts predict a single class - accuracy-based permutation
importance would be degenerate there, while AUC still uses the continuous
probability ranking.
"""
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

N_REPEATS = 30
RANDOM_STATE = 42


def compute_permutation_importance(model, X: pd.DataFrame, y: np.ndarray) -> dict:
    results = {}
    for scoring in ("roc_auc", "balanced_accuracy"):
        results[scoring] = permutation_importance(
            model, X, y,
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
            scoring=scoring,
        )
    return results


def summarize_permutation_importance(results: dict, feature_names: list) -> pd.DataFrame:
    df = pd.DataFrame({
        "gene_name": feature_names,
        "mean_importance_auc": results["roc_auc"].importances_mean,
        "std_importance_auc": results["roc_auc"].importances_std,
        "mean_importance_balanced_accuracy": results["balanced_accuracy"].importances_mean,
        "std_importance_balanced_accuracy": results["balanced_accuracy"].importances_std,
    })
    df = df.sort_values("mean_importance_auc", ascending=False).reset_index(drop=True)
    df["rank_auc"] = np.arange(1, len(df) + 1)
    return df


def compare_rankings(shap_df: pd.DataFrame, permutation_df: pd.DataFrame, rf_importance_df: pd.DataFrame) -> tuple:
    merged = (
        shap_df[["gene_name", "rank"]].rename(columns={"rank": "shap_rank"})
        .merge(permutation_df[["gene_name", "rank_auc"]].rename(columns={"rank_auc": "permutation_rank"}),
               on="gene_name")
        .merge(rf_importance_df[["gene_name", "rank"]].rename(columns={"rank": "rf_builtin_rank"}),
               on="gene_name")
    )

    corr_shap_perm, _ = spearmanr(merged["shap_rank"], merged["permutation_rank"])
    corr_shap_rf, _ = spearmanr(merged["shap_rank"], merged["rf_builtin_rank"])
    corr_perm_rf, _ = spearmanr(merged["permutation_rank"], merged["rf_builtin_rank"])

    correlations = {
        "spearman_shap_vs_permutation": float(corr_shap_perm),
        "spearman_shap_vs_rf_builtin": float(corr_shap_rf),
        "spearman_permutation_vs_rf_builtin": float(corr_perm_rf),
    }
    return merged, correlations
