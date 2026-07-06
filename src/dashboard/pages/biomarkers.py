"""20-gene clinical biomarker panel browser (Module 10's per-gene annotation)."""
import streamlit as st

from components.header import render_header, render_sidebar_patient_badge
from components.biomarker_cards import render_gene_card
from utils.caching import load_csv
from utils.session_manager import get_session, has_active_patient


def render() -> None:
    render_header("Biomarkers")
    render_sidebar_patient_badge()

    annotation = load_csv("pathway_analysis/Biomarker_Annotation.csv")

    st.dataframe(annotation, use_container_width=True, hide_index=True)

    st.markdown("### Gene detail")
    genes_imputed = set()
    if has_active_patient():
        genes_imputed = set(get_session().patient_matrix.genes_imputed)

    selected_gene = st.selectbox("Select a gene", annotation["gene_name"].tolist())
    row = annotation[annotation["gene_name"] == selected_gene].iloc[0]
    status = "imputed" if selected_gene in genes_imputed else None
    render_gene_card(row, patient_status=status)


render()
