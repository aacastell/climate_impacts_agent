import pandas as pd
import pydeck as pdk
import streamlit as st

DRIVER_COLORS = {
    "Avg. temperature change": "#e63946",
    "Avg. precipitation change": "#1d3557",
    "Heat extreme days": "#ff6b35",
    "Precipitation extreme days": "#457b9d",
}

IMPACT_COLOR = "#6a0dad"


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def grid_to_dataframe(grid: list[dict], color_hex: str) -> pd.DataFrame:
    df = pd.DataFrame(grid)
    vmin, vmax = df["value"].min(), df["value"].max()
    span = vmax - vmin or 1.0
    r, g, b = hex_to_rgb(color_hex)
    base = 245

    def to_rgba(value: float) -> list[int]:
        t = (value - vmin) / span
        return [
            int(base + (r - base) * t),
            int(base + (g - base) * t),
            int(base + (b - base) * t),
            180,
        ]

    df["color"] = df["value"].apply(to_rgba)
    return df


def _min_step(values: pd.Series) -> float:
    unique_sorted = sorted(values.unique())
    diffs = [b - a for a, b in zip(unique_sorted, unique_sorted[1:])]
    return min(diffs) if diffs else 1.0


def _cell_polygon(lat: float, lon: float, half_lat: float, half_lon: float) -> list[list[float]]:
    return [
        [lon - half_lon, lat - half_lat],
        [lon + half_lon, lat - half_lat],
        [lon + half_lon, lat + half_lat],
        [lon - half_lon, lat + half_lat],
    ]


def add_cell_polygons(grid_df: pd.DataFrame) -> pd.DataFrame:
    half_lat = _min_step(grid_df["lat"]) / 2
    half_lon = _min_step(grid_df["lon"]) / 2

    grid_df = grid_df.copy()
    grid_df["polygon"] = grid_df.apply(
        lambda row: _cell_polygon(row["lat"], row["lon"], half_lat, half_lon), axis=1
    )
    return grid_df


def render_grid_map(bbox: dict, grid_df: pd.DataFrame) -> None:
    grid_df = add_cell_polygons(grid_df)
    layer = pdk.Layer(
        "PolygonLayer",
        data=grid_df,
        get_polygon="polygon",
        get_fill_color="color",
        filled=True,
        stroked=True,
        get_line_color=[255, 255, 255, 60],
        line_width_min_pixels=1,
        pickable=True,
    )
    center_lat = (bbox["min_lat"] + bbox["max_lat"]) / 2
    center_lon = (bbox["min_lon"] + bbox["max_lon"]) / 2
    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=5)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{value}"}))
