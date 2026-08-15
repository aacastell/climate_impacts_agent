import streamlit as st

from climate_agent.frontend.mapping import (
    DRIVER_COLORS,
    IMPACT_COLOR,
    grid_to_dataframe,
    render_grid_map,
)


def render_maps(response: dict, driver: str) -> None:
    map_col1, map_col2 = st.columns([1, 1])

    with map_col1:
        st.subheader("Impact")
        impact_df = grid_to_dataframe(response["impact_grid"], IMPACT_COLOR)
        render_grid_map(response["bbox"], impact_df)
        st.caption(f"{response['region']} — {response['sector']} at {response['gwl']}")
        st.markdown(f"**{response['sector']} impact:** {response['sector_impact']}")

    with map_col2:
        st.subheader("Drivers")
        driver_df = grid_to_dataframe(response["driver_grids"][driver], DRIVER_COLORS[driver])
        render_grid_map(response["bbox"], driver_df)
        st.caption(f"{driver}: {response['drivers'][driver]}")
        st.markdown(f"**{driver}:** {response['driver_context'][driver]}")
