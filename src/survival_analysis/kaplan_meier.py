"""Kaplan-Meier survival curves and log-rank tests, per clinical-panel gene:
high vs. low expression (median split), one univariate test per gene.
"""
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def run_km_per_gene(cohort: pd.DataFrame, genes: list) -> pd.DataFrame:
    records = []
    for gene in genes:
        group_col = f"{gene}_group"
        high = cohort[cohort[group_col] == "High"]
        low = cohort[cohort[group_col] == "Low"]

        result = logrank_test(
            high["OS_time"], low["OS_time"],
            event_observed_A=high["OS_status"], event_observed_B=low["OS_status"],
        )

        kmf_high = KaplanMeierFitter().fit(high["OS_time"], high["OS_status"])
        kmf_low = KaplanMeierFitter().fit(low["OS_time"], low["OS_status"])

        records.append({
            "gene_name": gene,
            "n_high": len(high), "n_low": len(low),
            "events_high": int(high["OS_status"].sum()), "events_low": int(low["OS_status"].sum()),
            "median_survival_high": kmf_high.median_survival_time_,
            "median_survival_low": kmf_low.median_survival_time_,
            "logrank_p_value": result.p_value,
        })

    return pd.DataFrame(records).sort_values("logrank_p_value").reset_index(drop=True)


def fit_km_curves(cohort: pd.DataFrame, gene: str) -> tuple:
    group_col = f"{gene}_group"
    high = cohort[cohort[group_col] == "High"]
    low = cohort[cohort[group_col] == "Low"]
    kmf_high = KaplanMeierFitter().fit(high["OS_time"], high["OS_status"], label="High expression")
    kmf_low = KaplanMeierFitter().fit(low["OS_time"], low["OS_status"], label="Low expression")
    return kmf_high, kmf_low


if __name__ == "__main__":
    try:
        from . import survival_data_loader as sdl
    except ImportError:
        import survival_data_loader as sdl

    panel_genes = sdl.load_panel_genes()
    survival_cohort = sdl.add_expression_groups(sdl.load_survival_cohort(), panel_genes)
    km_results = run_km_per_gene(survival_cohort, panel_genes)
    print(km_results)
    print(f"\nGenes with logrank p < 0.05: {(km_results['logrank_p_value'] < 0.05).sum()} / {len(km_results)}")
