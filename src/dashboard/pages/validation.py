"""About / Model Validation - the platform's credibility page (Module 8)."""
import streamlit as st

from components.header import render_header, render_sidebar_patient_badge
from components.static_gallery import render_figure
from utils.caching import load_csv, load_json


def render() -> None:
    render_header("Model Validation")
    render_sidebar_patient_badge()

    metadata = load_json("trained_model/model_metadata.json")
    overall = load_csv("external_validation/overall_metrics.csv").iloc[0]
    cohorts = load_csv("external_validation/cohort_metrics.csv")

    with st.container(key="validation-summary-card"):
        st.markdown(
            f"**Model:** Random Forest ({metadata['parameters']['n_estimators']} trees, "
            f"class_weight=\"{metadata['parameters']['class_weight']}\"), trained on "
            f"{metadata['training_samples']} TCGA-CHOL samples ({metadata['training_features']}-gene clinical panel). "
            f"Trained: {metadata['training_date']}."
        )
        cols = st.columns(4)
        cols[0].metric("Overall ROC-AUC", f"{overall['ROC_AUC']:.3f}")
        cols[1].metric("Overall Sensitivity", f"{overall['Sensitivity']:.3f}")
        cols[2].metric("Overall Specificity", f"{overall['Specificity']:.3f}")
        cols[3].metric("External samples", int(overall["Samples"]))

    st.markdown("### Per-cohort breakdown")
    st.dataframe(cohorts, use_container_width=True, hide_index=True)
    st.warning(
        "GSE32225 and GSE89749 predict entirely class 0 (Normal) on this data, despite being mostly "
        "labeled Tumor - stated plainly here rather than hidden. GSE26566 is the only external cohort "
        "with a realistic mix of both predicted classes. See the Reports/Explainability pages for how "
        "this affects confidence in any single prediction."
    )

    st.markdown("### Validation figures")
    col1, col2, col3 = st.columns(3)
    with col1:
        render_figure("external_validation/figures/roc_curves.png", "ROC curves per cohort")
    with col2:
        render_figure("external_validation/figures/precision_recall_curves.png", "Precision-Recall curves")
    with col3:
        render_figure("external_validation/figures/calibration_curves.png", "Calibration curves")


render()
