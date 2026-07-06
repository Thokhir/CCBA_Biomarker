"""Composite prognostic risk score, built from the small set of genes whose
univariate Cox p-value is below a lenient exploratory threshold (0.10, not
0.05 - with only 18 events, few genes reach conventional significance, and a
completely empty risk score would be less useful than a clearly-labeled
exploratory one built from the top nominal hits).

The composite score itself becomes ONE covariate, which can then be combined
with a single clinical covariate (age) in a 2-variable multivariate Cox
model - respecting the ~10-events-per-covariate heuristic for this cohort's
18 observed deaths, rather than fitting all candidate genes simultaneously.
"""
import numpy as np
import pandas as pd

from lifelines import CoxPHFitter

RISK_GENE_PVALUE_THRESHOLD = 0.10


def select_risk_genes(cox_results: pd.DataFrame, threshold: float = RISK_GENE_PVALUE_THRESHOLD) -> list:
    return cox_results.loc[cox_results["p_value"] < threshold, "gene_name"].tolist()


def compute_risk_score(cohort: pd.DataFrame, cox_results: pd.DataFrame, risk_genes: list) -> pd.Series:
    """Weighted sum of standardized expression, weighted by each gene's
    univariate Cox coefficient (coefficients already fit on standardized
    expression, so weights are directly comparable)."""
    score = pd.Series(0.0, index=cohort.index)
    for gene in risk_genes:
        coef = cox_results.loc[cox_results["gene_name"] == gene, "coef"].iloc[0]
        standardized = (cohort[gene] - cohort[gene].mean()) / cohort[gene].std()
        score += coef * standardized
    return score


def fit_risk_score_cox(cohort_with_score: pd.DataFrame, extra_covariates: list = None) -> CoxPHFitter:
    covariates = ["risk_score"] + (extra_covariates or [])
    data = cohort_with_score[covariates + ["OS_time", "OS_status"]].dropna().copy()
    cph = CoxPHFitter()
    cph.fit(data, duration_col="OS_time", event_col="OS_status")
    return cph


if __name__ == "__main__":
    try:
        from . import survival_data_loader as sdl
        from . import cox_regression as cr
    except ImportError:
        import survival_data_loader as sdl
        import cox_regression as cr

    panel_genes = sdl.load_panel_genes()
    survival_cohort = sdl.load_survival_cohort()
    cox_results = cr.run_univariate_cox(survival_cohort, panel_genes)

    risk_genes = select_risk_genes(cox_results)
    print(f"Genes included in risk score (Cox p < {RISK_GENE_PVALUE_THRESHOLD}): {risk_genes}")

    survival_cohort["risk_score"] = compute_risk_score(survival_cohort, cox_results, risk_genes)
    cph = fit_risk_score_cox(survival_cohort, extra_covariates=["age_at_index"])
    print(cph.summary[["coef", "exp(coef)", "p"]])
