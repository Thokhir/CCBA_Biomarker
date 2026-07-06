"""Univariate Cox proportional hazards regression, one gene at a time.

Only univariate models are fit per gene: with 18 observed deaths, the
conventional events-per-variable heuristic (~10 events/covariate) supports
at most 1-2 covariates reliably. A single 20-gene multivariate Cox model
would be badly overfit and is not attempted here - see risk_score.py for how
a small number of the most informative genes are instead combined into one
composite covariate for a defensible 2-variable multivariate model.
"""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def run_univariate_cox(cohort: pd.DataFrame, genes: list) -> pd.DataFrame:
    records = []
    for gene in genes:
        data = cohort[[gene, "OS_time", "OS_status"]].copy()
        data[gene] = (data[gene] - data[gene].mean()) / data[gene].std()  # standardized for comparable HRs

        cph = CoxPHFitter()
        cph.fit(data, duration_col="OS_time", event_col="OS_status")
        summary = cph.summary.loc[gene]

        records.append({
            "gene_name": gene,
            "hazard_ratio": summary["exp(coef)"],
            "hr_lower_95": summary["exp(coef) lower 95%"],
            "hr_upper_95": summary["exp(coef) upper 95%"],
            "coef": summary["coef"],
            "p_value": summary["p"],
        })

    return pd.DataFrame(records).sort_values("p_value").reset_index(drop=True)


def fit_multivariate_cox(cohort: pd.DataFrame, covariates: list) -> CoxPHFitter:
    data = cohort[covariates + ["OS_time", "OS_status"]].dropna().copy()
    cph = CoxPHFitter()
    cph.fit(data, duration_col="OS_time", event_col="OS_status")
    return cph


if __name__ == "__main__":
    try:
        from . import survival_data_loader as sdl
    except ImportError:
        import survival_data_loader as sdl

    panel_genes = sdl.load_panel_genes()
    survival_cohort = sdl.load_survival_cohort()
    cox_results = run_univariate_cox(survival_cohort, panel_genes)
    print(cox_results)
    print(f"\nGenes with Cox p < 0.05: {(cox_results['p_value'] < 0.05).sum()} / {len(cox_results)}")
