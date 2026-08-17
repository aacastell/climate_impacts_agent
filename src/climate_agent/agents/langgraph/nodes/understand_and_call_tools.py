from climate_agent.agents.langgraph.state import AgentState, ToolCall
from climate_agent.schemas import BBox
from climate_agent.tools.agriculture import compute_agriculture
from climate_agent.tools.biome import compute_biome
from climate_agent.tools.gwl import window_from_gwl
from climate_agent.tools.hazard import compute_hazard_drivers
from climate_agent.tools.region import geocode

MAX_TOOL_ATTEMPTS = 3

SECTOR_TOOLS = {
    "Agriculture": compute_agriculture,
    "Biome": compute_biome,
}


def understand_and_call_tools(state: AgentState) -> dict:
    """Resolve region/gwl/sector from the query and call the tools needed to answer it.

    Loops (via the graph's conditional edge) until resolved or MAX_TOOL_ATTEMPTS is hit, in
    which case it degrades to a default area with a stated assumption rather than failing.
    No real query parsing yet — region/gwl/sector fall back to fixed defaults, so this
    currently always resolves in one pass; the loop and degrade path are real, just not
    reachable until the real router replaces the fallback below.

    Args: state — current graph state.
    Returns: partial state update.
    """
    attempts = state["attempts"] + 1

    region = state.get("region") or "Tolima, Colombia"
    gwl_text = state.get("gwl") or "2°C"
    sector = state.get("sector") or "Agriculture"

    bbox = geocode(region)

    if bbox is None:
        if attempts < MAX_TOOL_ATTEMPTS:
            return {
                "attempts": attempts,
                "status": "resolving",
                "tool_calls": [ToolCall(tool="geocode", args={"region": region}, result=None)],
            }
        return {
            "attempts": attempts,
            "status": "degraded",
            "region": region,
            "bbox": BBox(min_lat=-5.0, min_lon=-5.0, max_lat=5.0, max_lon=5.0),
            "gwl": gwl_text,
            "sector": sector,
            "assumptions": [f"Could not resolve '{region}' to a specific location — showing a default area."],
        }

    # Stopgap: hardcoded 2.0 stands in for real GWL extraction (tool #1, not built yet).
    window = window_from_gwl(2.0)
    gwl = f"{gwl_text} ({window.start_year}-{window.end_year})"

    sector_tool = SECTOR_TOOLS[sector]
    impact_grid, sector_impact = sector_tool(bbox, window)
    driver_grids, drivers = compute_hazard_drivers(bbox, window)

    return {
        "attempts": attempts,
        "status": "ready_to_narrate",
        "region": region,
        "bbox": bbox,
        "gwl": gwl,
        "sector": sector,
        "impact_grid": impact_grid,
        "driver_grids": driver_grids,
        "drivers": drivers,
        "sector_impact": sector_impact,
        "tool_calls": [
            ToolCall(tool="geocode", args={"region": region}, result="ok"),
            ToolCall(tool=f"sector:{sector}", args={"bbox": str(bbox), "window": str(window)}, result="ok"),
            ToolCall(tool="hazard", args={"bbox": str(bbox), "window": str(window)}, result="ok"),
        ],
    }
