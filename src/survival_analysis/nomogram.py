"""Simple nomogram-style point-scale for the final 2-covariate Cox model
(risk_score + age). Deliberately minimal: with only 18 observed deaths, the
underlying model itself is restricted to 2 covariates (see risk_score.py),
so a classic multi-axis nomogram would only be showing 2 lines anyway - this
builds the standard "points proportional to coefficient x covariate range"
construction directly from the fitted Cox model, without any external
nomogram library.
"""
import numpy as np
import pandas as pd


def build_nomogram_points(cph, cohort: pd.DataFrame, covariates: list, n_points_scale: int = 100) -> dict:
    """Returns per-covariate point-scale data: for each covariate, the
    observed value range mapped linearly to 0-n_points_scale, scaled by the
    covariate's relative hazard coefficient magnitude (largest |coef| gets
    the full point scale; others are scaled proportionally)."""
    coefs = cph.summary["coef"]
    max_abs_coef = coefs.abs().max()

    scales = {}
    for covariate in covariates:
        coef = coefs[covariate]
        value_min, value_max = cohort[covariate].min(), cohort[covariate].max()
        points_max = n_points_scale * (abs(coef) / max_abs_coef)
        scales[covariate] = {
            "value_min": value_min, "value_max": value_max,
            "points_min": 0.0, "points_max": points_max,
            "coef": coef, "direction": "positive" if coef > 0 else "negative",
        }
    return scales


def points_for_value(scale: dict, value: float) -> float:
    value_min, value_max = scale["value_min"], scale["value_max"]
    if value_max == value_min:
        return 0.0
    fraction = (value - value_min) / (value_max - value_min)
    if scale["direction"] == "negative":
        fraction = 1 - fraction
    return fraction * scale["points_max"]


def total_points_table(cohort: pd.DataFrame, scales: dict) -> pd.DataFrame:
    df = cohort.copy()
    total = pd.Series(0.0, index=df.index)
    for covariate, scale in scales.items():
        points_col = f"{covariate}_points"
        df[points_col] = df[covariate].apply(lambda v: points_for_value(scale, v))
        total += df[points_col]
    df["total_points"] = total
    return df


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

    cph = rs.fit_risk_score_cox(survival_cohort, extra_covariates=["age_at_index"])
    scales = build_nomogram_points(cph, survival_cohort, ["risk_score", "age_at_index"])
    print(scales)

    points_df = total_points_table(survival_cohort, scales)
    print(points_df[["case_id", "risk_score", "age_at_index", "total_points"]].head())
