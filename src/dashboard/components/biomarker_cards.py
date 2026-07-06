"""Renders one gene's row from Biomarker_Annotation.csv as a styled card."""
import pandas as pd
import streamlit as st


def render_gene_card(row: pd.Series, patient_status: str = None) -> None:
    with st.container(key=f"gene-card-{row['gene_name']}"):
        header = f"**{row['gene_name']}**"
        if patient_status == "imputed":
            header += ' <span class="gene-imputed-flag">(not measured in this patient - imputed)</span>'
        st.markdown(header, unsafe_allow_html=True)

        cols = st.columns(4)
        cols[0].metric("GO BP terms", int(row.get("n_GO_BP_terms", 0)))
        cols[1].metric("STRING degree", int(row.get("string_degree", 0)))
        cols[2].metric("Hub gene", "Yes" if row.get("is_hub") else "No")
        cols[3].metric("CCA PubMed hits", int(row.get("cca_hit_count", 0)) if pd.notna(row.get("cca_hit_count")) else 0)

        if pd.notna(row.get("top_disease_name")):
            st.caption(f"Top disease association: {row['top_disease_name']} (score={row.get('top_disease_score', 0):.2f})")
        if row.get("novel_in_cca_literature"):
            st.caption("Novel/understudied candidate in the cholangiocarcinoma literature.")
