"""Time-dependent (landmark) ROC-AUC for the risk score at fixed follow-up
horizons.

scikit-survival (which provides IPCW-based cumulative_dynamic_auc) fails to
build on this machine - no C++ compiler available for its Cython/ecos
dependencies. Instead, this uses the simpler, well-established landmark-time
approach: at each horizon t, define a binary "event by time t" outcome for
patients with either an observed event before t or follow-up reaching at
least t, and compute standard ROC-AUC via sklearn on the risk score.
Patients censored before t (unknown status at t) are excluded from that
horizon's AUC - the accepted trade-off of the landmark method versus full
IPCW weighting, and reasonable given only 34 patients total.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def compute_landmark_auc(cohort: pd.DataFrame, risk_col: str, horizons: list) -> pd.DataFrame:
    """For each horizon t: excludes patients censored before t (unknown
    status at t); includes everyone else, labeling y=1 if their event
    occurred strictly before t, else y=0 (survived to/past t, whether
    later censored or later had the event)."""
    records = []
    for horizon in horizons:
        evaluable = cohort[(cohort["OS_time"] >= horizon) | (cohort["OS_status"] == 1)].copy()
        y_true = ((evaluable["OS_time"] < horizon) & (evaluable["OS_status"] == 1)).astype(int)

        if y_true.nunique() < 2 or len(evaluable) < 10:
            records.append({"horizon_days": horizon, "n_evaluable": len(evaluable),
                             "n_events_by_horizon": int(y_true.sum()), "auc": np.nan})
            continue

        auc = roc_auc_score(y_true, evaluable[risk_col])
        records.append({"horizon_days": horizon, "n_evaluable": len(evaluable),
                         "n_events_by_horizon": int(y_true.sum()), "auc": auc})

    return pd.DataFrame(records)


def default_horizons(cohort: pd.DataFrame, n_horizons: int = 4) -> list:
    max_time = cohort["OS_time"].max()
    return [round(max_time * f) for f in np.linspace(0.2, 0.8, n_horizons)]


if __name__ == "__main__":
    try:
        from . import survival_data_loader as sdl
        from . import cox_regression as cr
        from . import risk_score as rs
    except ImportError:
        import survival_data_loader as sdl
        import cox_regression as cr
        import risk_score as rs

    panel_genes = sdl.load_panel_genes()
    survival_cohort = sdl.load_survival_cohort()
    cox_results = cr.run_univariate_cox(survival_cohort, panel_genes)
    risk_genes = rs.select_risk_genes(cox_results)
    survival_cohort["risk_score"] = rs.compute_risk_score(survival_cohort, cox_results, risk_genes)

    horizons = default_horizons(survival_cohort)
    print(f"Horizons (days): {horizons}")
    roc_df = compute_landmark_auc(survival_cohort, "risk_score", horizons)
    print(roc_df)
