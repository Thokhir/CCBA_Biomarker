"""Upload a patient sample -> preprocess -> predict -> show probability."""
from datetime import datetime

import pandas as pd
import streamlit as st

from components.header import render_header, render_sidebar_patient_badge
from components.uploader import render_uploader
from components.preprocessing import build_patient_matrix
from components.predictor import predict_patient
from components import plots
from utils.caching import get_model_and_feature_order
from utils.session_manager import get_session, reset_session
from utils.validation import validate_upload


def render() -> None:
    render_header("Prediction")

    model, feature_order = get_model_and_feature_order()
    uploaded_file = render_uploader(feature_order)

    if uploaded_file is not None:
        session = get_session()
        if session.upload_file_id != uploaded_file.file_id:
            reset_session()
            session = get_session()
            session.upload_file_id = uploaded_file.file_id

            raw_df = pd.read_csv(uploaded_file)
            validation = validate_upload(raw_df, feature_order)

            for error in validation.errors:
                st.error(error)

            if validation.is_valid:
                for warning in validation.warnings:
                    st.warning(warning)

                raw_row = raw_df.iloc[0]
                sample_id = str(raw_row["sample_id"]) if "sample_id" in raw_row else "patient_1"

                patient_matrix = build_patient_matrix(raw_row, feature_order, sample_id)
                prediction = predict_patient(model, patient_matrix.X)

                session.sample_id = sample_id
                session.raw_upload_df = raw_df
                session.patient_matrix = patient_matrix
                session.prediction = prediction
                session.uploaded_at = datetime.now()

    render_sidebar_patient_badge()

    session = get_session()
    if session.prediction is not None:
        with st.container(key="prediction-result-card"):
            st.subheader(f"Result for {session.sample_id}")
            st.dataframe(session.raw_upload_df, use_container_width=True)

            st.plotly_chart(
                plots.probability_gauge(session.prediction.tumor_probability, session.prediction.predicted_label),
                use_container_width=True,
            )
            st.metric("Confidence", session.prediction.confidence)

            if session.patient_matrix.genes_imputed:
                st.caption(
                    f"Imputed (not measured) genes: {', '.join(session.patient_matrix.genes_imputed)}"
                )

        st.success("Prediction complete. Continue to **Explainability** to see why, or **Reports** to download a PDF.")


render()
