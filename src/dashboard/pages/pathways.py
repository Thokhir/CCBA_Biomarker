"""GO/KEGG/Reactome/WikiPathways/STRING browsing (Module 10)."""
import streamlit as st

from components.header import render_header, render_sidebar_patient_badge
from components.static_gallery import render_figure
from utils.caching import load_csv


def _enrichment_tab(library_name: str, csv_path: str, figure_paths: list) -> None:
    df = load_csv(csv_path)
    n_sig = int((df["Adjusted P-value"] < 0.05).sum())
    st.caption(f"{len(df)} terms tested, {n_sig} significant at padj<0.05.")
    cols = st.columns(len(figure_paths))
    for col, fig_path in zip(cols, figure_paths):
        with col:
            render_figure(fig_path, library_name)
    with st.expander("View underlying data table"):
        st.dataframe(df, use_container_width=True)


def render() -> None:
    render_header("Pathways")
    render_sidebar_patient_badge()

    st.markdown(
        "None of these enrichment analyses reach significance after multiple-testing correction "
        "(a real, expected result - the panel was selected via independent statistical/ML consensus "
        "scoring, not pathway curation). Nominal top hits are shown for reference."
    )

    tabs = st.tabs(["GO: Biological Process", "GO: Cellular Component", "GO: Molecular Function",
                    "KEGG", "Reactome", "WikiPathways", "STRING Network"])

    with tabs[0]:
        _enrichment_tab("GO BP", "pathway_analysis/GO_BP.csv",
                         ["pathway_analysis/figures/GO_BP_barplot.png", "pathway_analysis/figures/GO_BP_dotplot.png"])
    with tabs[1]:
        _enrichment_tab("GO CC", "pathway_analysis/GO_CC.csv",
                         ["pathway_analysis/figures/GO_CC_barplot.png", "pathway_analysis/figures/GO_CC_dotplot.png"])
    with tabs[2]:
        _enrichment_tab("GO MF", "pathway_analysis/GO_MF.csv",
                         ["pathway_analysis/figures/GO_MF_barplot.png", "pathway_analysis/figures/GO_MF_dotplot.png"])
    with tabs[3]:
        _enrichment_tab("KEGG", "pathway_analysis/KEGG.csv", ["pathway_analysis/figures/KEGG.png"])
    with tabs[4]:
        _enrichment_tab("Reactome", "pathway_analysis/Reactome.csv", ["pathway_analysis/figures/Reactome.png"])
    with tabs[5]:
        _enrichment_tab("WikiPathways", "pathway_analysis/WikiPathways.csv",
                         ["pathway_analysis/figures/WikiPathways.png"])
    with tabs[6]:
        network = load_csv("pathway_analysis/STRING_network.csv")
        degree = load_csv("pathway_analysis/STRING_degree.csv")
        n_hubs = int(degree["is_hub"].sum())
        st.caption(f"{len(network)} edges among {len(degree)} genes, {n_hubs} hub genes (degree >= 2).")
        render_figure("pathway_analysis/figures/STRING_network.png", "STRING protein-protein interaction network")
        with st.expander("View degree centrality table"):
            st.dataframe(degree, use_container_width=True)


render()
