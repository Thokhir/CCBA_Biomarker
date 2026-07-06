"""Centralized cached loaders so every page reads precomputed artifacts the
same way, instead of scattering pd.read_csv calls with inconsistent caching.

Model loading uses @st.cache_resource (shared, unpickled object - must not
be copied/re-hashed). Every CSV/JSON/PNG read from Modules 8/9/10/11/12 uses
@st.cache_data (pure, immutable-during-session reads). Live per-upload
computation (prediction/SHAP/risk placement) is never cached here.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[3]
RESULTS_DIR = BASE_DIR / "results"


@st.cache_resource
def get_model_and_feature_order():
    model = joblib.load(RESULTS_DIR / "trained_model" / "rf_model.pkl")
    feature_order = pd.read_csv(RESULTS_DIR / "trained_model" / "feature_order.csv")["gene_name"].tolist()
    return model, feature_order


@st.cache_data
def load_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / relative_path)


@st.cache_data
def load_json(relative_path: str) -> dict:
    with open(RESULTS_DIR / relative_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_image_bytes(relative_path: str) -> bytes:
    return (RESULTS_DIR / relative_path).read_bytes()


def image_exists(relative_path: str) -> bool:
    return (RESULTS_DIR / relative_path).exists()
