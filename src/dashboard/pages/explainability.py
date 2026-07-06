"""SHAP explanation: this patient (live) + global model behavior (precomputed)."""
import streamlit as st

from components.header import render_header, render_sidebar_patient_badge
from components.shap_runtime import explain_patient
from components import plots
from components.static_gallery import render_figure
from utils.caching import get_model_and_feature_order, load_csv
from utils.session_manager import get_session, has_active_patient


def render() -> None:
    render_header("Explainability")
    render_sidebar_patient_badge()

    if not has_active_patient():
        st.info("Upload a patient sample on the **Prediction** page first.")
        st.stop()

    session = get_session()
    model, feature_order = get_model_and_feature_order()

    tab_patient, tab_global = st.tabs(["This Patient", "Global Model Behavior"])

    with tab_patient:
        if session.explanation is None:
            session.explanation = explain_patient(
                model, session.patient_matrix.X, feature_order, session.patient_matrix.genes_imputed
            )

        st.plotly_chart(
            plots.shap_waterfall(session.explanation.contributions, session.explanation.base_value),
            use_container_width=True,
        )
        st.caption(
            "Amber bars indicate a gene that was not measured for this patient (imputed as 0) - "
            "its contribution reflects a fabricated value, not a real measurement."
        )
        st.dataframe(session.explanation.contributions, use_container_width=True)

    with tab_global:
        st.markdown("How the model behaves across the TCGA training population (44 samples):")
        col1, col2 = st.columns(2)
        with col1:
            render_figure("explainability/global/shap_global_bar.png", "Global SHAP importance (mean |value|)")
        with col2:
            render_figure("explainability/global/shap_summary_beeswarm.png", "SHAP beeswarm (training population)")

        st.markdown("Dependence plots for the top 5 globally important genes:")
        top5_genes = load_csv("explainability/global/shap_global_importance.csv").head(5)["gene_name"].tolist()
        dep_cols = st.columns(5)
        for col, gene in zip(dep_cols, top5_genes):
            with col:
                render_figure(f"explainability/global/dependence_{gene}.png", gene)


render()
