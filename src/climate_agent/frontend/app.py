import streamlit as st

from climate_agent.frontend.api_client import QueryError, fetch_query
from climate_agent.frontend.components.driver_selector import render_driver_selector
from climate_agent.frontend.components.header import render_header
from climate_agent.frontend.components.maps import render_maps
from climate_agent.frontend.components.narration import render_narration
from climate_agent.frontend.components.query_bar import render_query_bar

st.set_page_config(page_title="Climate Impacts Agent", layout="wide")

render_header()

query_text, ask = render_query_bar()

if ask:
    try:
        result = fetch_query(query_text)
    except QueryError as e:
        st.error(f"Couldn't reach the backend: {e}")
    else:
        st.session_state["submitted"] = True
        st.session_state["response"] = result

if st.session_state.get("submitted"):
    response = st.session_state["response"]
    render_narration(response)
    driver = render_driver_selector()
    render_maps(response, driver)
else:
    st.info("Ask a question to see impacts, drivers, and narration.")
