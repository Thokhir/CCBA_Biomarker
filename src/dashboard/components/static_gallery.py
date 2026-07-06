"""Displays pre-rendered 300 DPI PNGs from Modules 8/10/11/12 grouped in
tabs/expanders, so browsing research results never dumps 40+ images in one
long scroll.
"""
import streamlit as st

from utils.caching import load_image_bytes, image_exists


def render_figure(relative_path: str, caption: str) -> None:
    if not image_exists(relative_path):
        st.info(f"Figure not found: {relative_path}")
        return
    st.image(load_image_bytes(relative_path), caption=caption, use_container_width=True)


def render_figure_tabs(tab_specs: list) -> None:
    """tab_specs: list of (tab_label, relative_path, caption)."""
    tabs = st.tabs([spec[0] for spec in tab_specs])
    for tab, (label, relative_path, caption) in zip(tabs, tab_specs):
        with tab:
            render_figure(relative_path, caption)
