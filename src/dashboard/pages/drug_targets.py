"""Drug target discovery browsing (Module 12)."""
import streamlit as st

from components.header import render_header, render_sidebar_patient_badge
from components.static_gallery import render_figure
from utils.caching import load_csv


def render() -> None:
    render_header("Drug Targets")
    render_sidebar_patient_badge()

    priority = load_csv("drug_targets/TherapeuticPriority.csv")
    repurposing = load_csv("drug_targets/DrugRepurposing.csv")
    biomarker_drug = load_csv("drug_targets/BiomarkerDrugTable.csv")

    tab1, tab2, tab3 = st.tabs(["Priority Ranking", "Drug Repurposing Candidates", "Druggability Detail"])

    with tab1:
        st.dataframe(
            priority[["gene_name", "target_priority_score", "score_predictive",
                      "score_external", "score_biological", "score_survival"]],
            use_container_width=True, hide_index=True,
        )
        render_figure("drug_targets/figures/TargetPriority.png", "Composite target priority score")

    with tab2:
        top = repurposing.iloc[0]
        with st.container(key="top-repurposing-card"):
            st.markdown(
                f"**Top candidate:** {top['gene_name']} → {top['drug_name']} "
                f"({top['source']}, priority={top['target_priority_score']:.3f})"
            )
        st.dataframe(repurposing, use_container_width=True, hide_index=True)
        render_figure("drug_targets/figures/DrugNetwork.png", "Gene-drug interaction network")

    with tab3:
        st.dataframe(biomarker_drug, use_container_width=True, hide_index=True)
        st.caption("DrugBank is not integrated - its API now requires a paid academic license. "
                   "DGIdb and Open Targets (both free) provide the drug-gene evidence shown here.")


render()
