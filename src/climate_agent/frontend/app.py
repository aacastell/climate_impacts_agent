import pandas as pd
import streamlit as st

MOCK_REGIONS = {
    "Colombia": (4.5, -74.0),
    "Mexico": (23.6, -102.5),
    "United States": (39.8, -98.6),
    "China": (35.0, 105.0),
    "Brazil": (-14.2, -51.9),
}
MOCK_GWLS = ["1.5°C", "2°C", "3°C"]
MOCK_SECTORS = ["Agriculture", "Biome"]


def get_mock_response(region: str, gwl: str, sector: str) -> dict:
    """Placeholder for the eventual API call. Same shape as what the real
    backend will return, so swapping this out later doesn't require
    touching the rendering code below."""
    lat, lon = MOCK_REGIONS[region]
    sector_impact_text = {
        "Agriculture": "Maize yield projected to decline (placeholder — real tool not wired yet).",
        "Biome": "Elevated probability of biome transition toward a drier vegetation class (placeholder — real tool not wired yet).",
    }[sector]
    narration = (
        f"At {gwl} of additional warming above today's climate, {region} shows "
        f"meaningful {sector.lower()} impacts alongside intensifying heat and "
        f"precipitation extremes. (Placeholder narration — real tools, RAG "
        f"grounding, and model layer aren't wired in yet.)"
    )
    return {
        "region": region,
        "gwl": gwl,
        "sector": sector,
        "lat": lat,
        "lon": lon,
        "heat_context": "placeholder — hazard tool not wired yet.",
        "precip_context": "placeholder — hazard tool not wired yet.",
        "sector_impact": sector_impact_text,
        "narration": narration,
    }


st.set_page_config(page_title="Climate Impacts Agent", layout="wide")
st.title("Climate Impacts Agent")
st.caption(
    "Explore how a level of global warming would affect an ISIMIP sector in a "
    "country. Running against mock data for now — no backend wired up yet."
)

with st.sidebar:
    st.subheader("Query")
    region = st.selectbox("Country", list(MOCK_REGIONS.keys()))
    gwl = st.selectbox("Warming level (above today)", MOCK_GWLS)
    sector = st.selectbox("Sector", MOCK_SECTORS)
    ask = st.button("Ask", type="primary")

if ask:
    st.session_state["submitted"] = True
    st.session_state["response"] = get_mock_response(region, gwl, sector)

if st.session_state.get("submitted"):
    response = st.session_state["response"]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Map")
        st.map(pd.DataFrame({"lat": [response["lat"]], "lon": [response["lon"]]}), zoom=3)
        st.caption(f"{response['region']} — {response['sector']} at {response['gwl']}")

    with col2:
        st.subheader("Context")
        st.markdown(f"**Heat:** {response['heat_context']}")
        st.markdown(f"**Precipitation:** {response['precip_context']}")
        st.markdown(f"**{response['sector']} impact:** {response['sector_impact']}")

    st.subheader("Narration")
    st.write(response["narration"])
else:
    st.info("Select a country, warming level, and sector, then click Ask.")
