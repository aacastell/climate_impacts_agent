import streamlit as st


def render_query_bar() -> tuple[str, bool]:
    with st.form("query_form"):
        query = st.text_input(
            "Ask a question",
            placeholder="e.g. How would 2°C of warming affect agriculture in Tolima, Colombia?",
        )
        ask = st.form_submit_button("Ask", type="primary")
    return query, ask
