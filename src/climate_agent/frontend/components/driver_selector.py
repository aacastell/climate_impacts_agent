import streamlit as st

DRIVER_OPTIONS = [
    "Avg. temperature change",
    "Avg. precipitation change",
    "Heat extreme days",
    "Precipitation extreme days",
]


def render_driver_selector() -> str:
    return st.selectbox("Show driver", DRIVER_OPTIONS)
