"""Shared page header, consistent across every page."""
import streamlit as st

from utils.session_manager import get_session, has_active_patient


def render_header(page_title: str) -> None:
    st.markdown(
        f'<div class="dashboard-header"><h1>CCA-BDP &middot; {page_title}</h1></div>',
        unsafe_allow_html=True,
    )


def render_sidebar_patient_badge() -> None:
    if has_active_patient():
        session = get_session()
        st.sidebar.markdown(
            f'<div class="active-patient-badge">Active patient: {session.sample_id}<br>'
            f'Prediction: {"Tumor" if session.prediction.predicted_label == 1 else "Normal"} '
            f'({session.prediction.tumor_probability:.1%})</div>',
            unsafe_allow_html=True,
        )
