import streamlit as st


def render_header() -> None:
    st.title("Climate Impacts Agent")
    st.caption(
        "Explore how a level of global warming would affect an ISIMIP sector in a "
        "region. Backend responses are currently stubbed — real tools aren't wired "
        "in yet."
    )
