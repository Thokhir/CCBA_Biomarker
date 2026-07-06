"""Generate + download the single-patient clinical decision-support PDF."""
import streamlit as st

from components.header import render_header, render_sidebar_patient_badge
from components.report_generator import generate_patient_report
from components import plots
from utils.caching import load_csv
from utils.session_manager import get_session, has_active_patient


def render() -> None:
    render_header("Reports")
    render_sidebar_patient_badge()

    if not has_active_patient():
        st.info("Upload a patient sample on the **Prediction** page first.")
        st.stop()

    session = get_session()

    st.markdown(
        "This report covers **this patient only**: prediction, top contributing biomarkers, "
        "relevant pathway/drug context, and a model-validation footnote. It is not a whole-platform "
        "research summary (see the platform's other pages for that)."
    )

    if st.button("Generate PDF Report", type="primary"):
        if session.explanation is None:
            from components.shap_runtime import explain_patient
            from utils.caching import get_model_and_feature_order
            model, feature_order = get_model_and_feature_order()
            session.explanation = explain_patient(
                model, session.patient_matrix.X, feature_order, session.patient_matrix.genes_imputed
            )

        biomarker_annotation = load_csv("pathway_analysis/Biomarker_Annotation.csv")
        therapeutic_priority = load_csv("drug_targets/TherapeuticPriority.csv")
        drug_repurposing = load_csv("drug_targets/DrugRepurposing.csv")
        overall_metrics = load_csv("external_validation/overall_metrics.csv").iloc[0]
        waterfall_fig = plots.shap_waterfall(session.explanation.contributions, session.explanation.base_value)

        pdf_bytes = generate_patient_report(
            session, biomarker_annotation, therapeutic_priority, drug_repurposing, overall_metrics, waterfall_fig
        )

        st.download_button(
            "Download PDF Report",
            data=pdf_bytes,
            file_name=f"cca_report_{session.sample_id}.pdf",
            mime="application/pdf",
        )
        st.success(f"Report generated ({len(pdf_bytes) / 1024:.0f} KB).")


render()
