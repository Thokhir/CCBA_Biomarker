"""Cross-cohort SHAP comparison: are biomarker contributions consistent
between the TCGA training population and the independent GEO cohorts, or
does the model rely on different genes once distribution shift is present?
"""
import pandas as pd

TRAINING_SOURCE_LABEL = "TCGA_training"


def build_comparison_frame(train_shap_df: pd.DataFrame, external_shap_all_df: pd.DataFrame,
                            feature_names: list, top_genes: list) -> pd.DataFrame:
    rows = []

    for gene in top_genes:
        col = f"shap_{gene}"
        train_values = train_shap_df[col]
        rows.append({
            "gene_name": gene, "source": TRAINING_SOURCE_LABEL,
            "mean_shap": train_values.mean(), "std_shap": train_values.std(), "n_samples": len(train_values),
        })

        for cohort in sorted(external_shap_all_df["cohort"].unique()):
            cohort_values = external_shap_all_df.loc[external_shap_all_df["cohort"] == cohort, col]
            rows.append({
                "gene_name": gene, "source": cohort,
                "mean_shap": cohort_values.mean(), "std_shap": cohort_values.std(), "n_samples": len(cohort_values),
            })

    return pd.DataFrame(rows)
