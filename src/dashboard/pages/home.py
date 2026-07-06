"""Landing page: platform overview and headline KPIs from cached CSVs."""
import streamlit as st

from components.header import render_header, render_sidebar_patient_badge
from utils.caching import load_csv, load_json


def render() -> None:
    render_header("Home")
    render_sidebar_patient_badge()

    st.markdown(
        "An AI-powered translational biomarker discovery platform for cholangiocarcinoma (CCA): "
        "from raw TCGA/GEO data to a validated diagnostic model, explainable predictions, "
        "biological interpretation, prognostic context, and therapeutic leads."
    )

    overall = load_csv("external_validation/overall_metrics.csv").iloc[0]
    repurposing = load_csv("drug_targets/DrugRepurposing.csv")
    survival_summary = load_json("survival/reports/survival_summary.json")

    with st.container(key="home-kpi-card"):
        cols = st.columns(4)
        cols[0].metric("Biomarker panel", "20 genes")
        cols[1].metric("External ROC-AUC", f"{overall['ROC_AUC']:.3f}")
        cols[2].metric("External cohorts", "3 GEO datasets")
        cols[3].metric("Survival cohort", f"{survival_summary['cohort_size']} patients")

    with st.container(key="home-top-candidate-card"):
        top_candidate = repurposing.iloc[0]
        st.markdown(
            f"**Top repurposing candidate:** {top_candidate['gene_name']} → {top_candidate['drug_name']} "
            f"(priority score {top_candidate['target_priority_score']:.3f})"
        )

    st.info("Start with **Prediction** in the sidebar to score a new patient sample.")


render()
