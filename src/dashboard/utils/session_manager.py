"""Per-user session state for the active patient's upload/prediction/
explanation/risk-placement results, so navigating between pages doesn't
lose or recompute context.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import streamlit as st

SESSION_KEY = "patient_session"


@dataclass
class PatientSession:
    sample_id: Optional[str] = None
    raw_upload_df: object = None
    patient_matrix: object = None
    prediction: object = None
    explanation: object = None
    risk_placement: object = None
    uploaded_at: Optional[datetime] = None
    upload_file_id: Optional[str] = None


def get_session() -> PatientSession:
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = PatientSession()
    return st.session_state[SESSION_KEY]


def reset_session() -> None:
    st.session_state[SESSION_KEY] = PatientSession()


def has_active_patient() -> bool:
    session = get_session()
    return session.prediction is not None
