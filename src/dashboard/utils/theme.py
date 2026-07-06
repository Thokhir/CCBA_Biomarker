"""Color palette and custom CSS injection for the CCA-BDP dashboard.

The Streamlit theme itself lives in .streamlit/config.toml; this module
adds the polish config.toml can't express (card styling, hiding default
chrome, header treatment).
"""
PRIMARY = "#0F6E68"
SECONDARY_BG = "#F4F7F7"
TEXT = "#1A2B2A"
WARNING_ACCENT = "#B8860B"
ALERT_ACCENT = "#8C1D18"
BORDER = "#E0E6E5"

CUSTOM_CSS = f"""
<style>
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

/* Card containers use st.container(key="...-card") so content actually
nests inside one real DOM element - unsafe_allow_html div tags split
across separate st.markdown calls do NOT nest their content, since each
st.markdown call is its own isolated DOM node. */
div[class*="st-key-"][class*="-card"] {{
    background: {SECONDARY_BG};
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    border: 1px solid {BORDER};
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-bottom: 1rem;
}}

.dashboard-header {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid {PRIMARY};
}}

.dashboard-header h1 {{
    font-size: 1.5rem;
    margin: 0;
    color: {TEXT};
}}

.active-patient-badge {{
    background: {PRIMARY};
    color: white;
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    font-size: 0.85rem;
    margin-top: 1rem;
}}

.gene-imputed-flag {{
    color: {WARNING_ACCENT};
    font-weight: 600;
}}
</style>
"""


def inject_custom_css() -> None:
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
