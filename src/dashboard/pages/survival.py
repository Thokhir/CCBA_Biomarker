"""Survival analysis browsing (Module 11) + active patient's risk placement.

Risk score uses ITIH4 only (the sole gene with univariate Cox p<0.10 in
Hazard_Ratios.csv - PACC1's p=0.121 does not pass that threshold, confirmed
directly; do not add PACC1 back in here).
"""
import sys
from pathlib import Path

import streamlit as st

from components.header import render_header, render_sidebar_patient_badge
from components.static_gallery import render_figure
from components import plots
from utils.caching import load_csv
from utils.session_manager import get_session, has_active_patient

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from survival_analysis.survival_data_loader import load_survival_cohort

RISK_GENES = ["ITIH4"]


def _compute_patient_risk_score(patient_matrix, cohort) -> float:
    hazard_ratios = load_csv("survival/Hazard_Ratios.csv")
    score = 0.0
    for gene in RISK_GENES:
        coef = hazard_ratios.loc[hazard_ratios["gene_name"] == gene, "coef"].iloc[0]
        gene_value = float(patient_matrix.X[gene].iloc[0])
        mean, std = cohort[gene].mean(), cohort[gene].std()
        score += coef * (gene_value - mean) / std
    return score


def render() -> None:
    render_header("Survival")
    render_sidebar_patient_badge()

    st.warning(
        "Survival findings are exploratory and hypothesis-generating, based on 34 patients with "
        "only 18 observed deaths. This risk placement is not a validated clinical-grade prognostic output."
    )

    col1, col2 = st.columns(2)
    with col1:
        render_figure("survival/figures/Hazard_Forest.png", "Univariate Cox hazard ratios")
        render_figure("survival/figures/TimeROC.png", "Time-dependent (landmark) ROC-AUC")
    with col2:
        render_figure("survival/figures/Nomogram.png", "Simplified nomogram (risk score + age)")
        render_figure("survival/figures/KM_CDH3.png", "Kaplan-Meier: CDH3 (logrank p=0.031)")

    with st.expander("View Kaplan-Meier statistics and hazard ratios"):
        st.dataframe(load_csv("survival/KM_statistics.csv"), use_container_width=True)
        st.dataframe(load_csv("survival/Hazard_Ratios.csv"), use_container_width=True)

    if has_active_patient():
        session = get_session()
        if all(g in session.patient_matrix.X.columns for g in RISK_GENES):
            cohort = load_survival_cohort()
            patient_risk_score = _compute_patient_risk_score(session.patient_matrix, cohort)
            session.risk_placement = patient_risk_score

            risk_scores = load_csv("survival/Risk_Score.csv")["risk_score"]
            st.markdown("### This patient's risk placement")
            st.plotly_chart(
                plots.risk_placement_histogram(risk_scores.tolist(), patient_risk_score),
                use_container_width=True,
            )


render()
