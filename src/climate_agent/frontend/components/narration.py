import streamlit as st


def render_narration(response: dict) -> None:
    st.subheader("Narration")
    st.write(response["narration"])
