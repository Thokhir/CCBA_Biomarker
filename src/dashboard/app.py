"""CCA-BDP Clinical Decision Support System - entry point.

Sets up page config, theme/CSS, sys.path for cross-package imports (no
__init__.py exists anywhere in this repo, matching every other module),
and routes pages via st.navigation.
"""
import sys
from pathlib import Path

import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent
BASE_DIR = DASHBOARD_DIR.parents[1]
sys.path.insert(0, str(DASHBOARD_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

from utils.theme import inject_custom_css
from utils.session_manager import get_session

st.set_page_config(
    page_title="CCA-BDP Clinical Decision Support",
    page_icon=":material/science:",
    layout="wide",
)
inject_custom_css()
get_session()

pages = {
    "Overview": [
        st.Page("pages/home.py", title="Home", icon=":material/home:"),
    ],
    "Clinical Workflow": [
        st.Page("pages/prediction.py", title="Prediction", icon=":material/monitor_heart:"),
        st.Page("pages/explainability.py", title="Explainability", icon=":material/psychology:"),
        st.Page("pages/reports.py", title="Reports", icon=":material/description:"),
    ],
    "Research Evidence": [
        st.Page("pages/biomarkers.py", title="Biomarkers", icon=":material/biotech:"),
        st.Page("pages/pathways.py", title="Pathways", icon=":material/hub:"),
        st.Page("pages/survival.py", title="Survival", icon=":material/timeline:"),
        st.Page("pages/drug_targets.py", title="Drug Targets", icon=":material/medication:"),
        st.Page("pages/validation.py", title="Model Validation", icon=":material/verified:"),
    ],
}

navigation = st.navigation(pages)
navigation.run()
