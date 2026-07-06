"""File uploader widget + downloadable CSV template, generated directly
from feature_order.csv so the template can never drift from the model.
"""
import io

import pandas as pd
import streamlit as st


def build_template_csv(feature_order: list) -> bytes:
    columns = ["sample_id"] + feature_order
    example_row = ["patient_1"] + [5.0] * len(feature_order)
    df = pd.DataFrame([example_row], columns=columns)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def render_uploader(feature_order: list):
    st.download_button(
        "Download CSV template",
        data=build_template_csv(feature_order),
        file_name="cca_patient_template.csv",
        mime="text/csv",
        help="One row per patient, one column per gene. Missing genes are treated as not measured.",
    )
    return st.file_uploader("Upload patient expression CSV", type=["csv"])
