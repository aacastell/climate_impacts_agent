from climate_agent.agents.langgraph.state import AgentState, ToolCall
from climate_agent.schemas import BBox
from climate_agent.tools.agriculture import compute_agriculture
from climate_agent.tools.biome import compute_biome
from climate_agent.tools.gwl import (
    window_from_gwl,
    window_from_heat_extremes,
    window_from_precip_extremes,
)
from climate_agent.tools.hazard import compute_hazard_drivers
from climate_agent.tools.language import detect_language
from climate_agent.tools.region import geocode
from climate_agent.tools.router import extract_query_fields

MAX_TOOL_ATTEMPTS = 3

SECTOR_TOOLS = {
    "Agriculture": compute_agriculture,
    "Biome": compute_biome,
}

WINDOW_TOOLS = {
    "gwl": window_from_gwl,
    "heat_extreme": window_from_heat_extremes,
    "precip_extreme": window_from_precip_extremes,
}

DEFAULT_REGION = "Tolima, Colombia"
DEFAULT_SECTOR = "Agriculture"
DEFAULT_GWL_MODE = "gwl"
DEFAULT_GWL_VALUE = 2.0


def _simplify_region(region: str, attempt: int) -> str:
    """Progressively broader interpretation of a comma-separated region description.

    Real alternate-interpretation retry, not a blind repeat: place descriptions commonly go
    specific-to-broad left to right ("the Nile Delta, Egypt"), so later attempts drop leading
    segments, trying broader jurisdictions the geocoder is more likely to recognize. A
    single-segment region (no comma) has nothing to simplify, so every attempt is identical —
    honest, not a fabricated alternative.

    Args: region — original region text. attempt — 1-indexed retry number.
    Returns: region text for this attempt.
    """
    segments = [s.strip() for s in region.split(",")]
    drop = min(attempt - 1, len(segments) - 1)
    return ", ".join(segments[drop:])


def understand_and_call_tools(state: AgentState) -> dict:
    """Resolve region/gwl/sector from the query, via the fine-tuned router (item 14), and call
    the tools needed to answer it.

    Loops (via the graph's conditional edge) until resolved or MAX_TOOL_ATTEMPTS is hit, in
    which case it degrades to a default area with a stated assumption rather than failing. Any
    field the router can't extract or extracts as an unsupported value falls back to a default
    with a stated assumption too — same bounded-resolve-then-degrade pattern used everywhere
    else in this node.

    Args: state — current graph state.
    Returns: partial state update.
    """
    attempts = state["attempts"] + 1
    assumptions: list[str] = []
    language = detect_language(state["query"])

    extracted = extract_query_fields(state["query"]) or {}
    if not extracted:
        assumptions.append("Could not understand the query — using default region, sector, and GWL target.")

    router_region = extracted.get("region") or DEFAULT_REGION
    region = _simplify_region(router_region, attempts)
    if region != router_region:
        assumptions.append(f"Couldn't resolve '{router_region}' directly — trying the broader '{region}'.")

    sector = extracted.get("sector")
    if sector not in SECTOR_TOOLS:
        assumptions.append(f"'{sector}' isn't a supported sector — showing {DEFAULT_SECTOR} instead.")
        sector = DEFAULT_SECTOR

    gwl_mode = extracted.get("gwl_mode")
    if gwl_mode not in WINDOW_TOOLS:
        assumptions.append(f"'{gwl_mode}' isn't a supported warming-level mode — using a direct GWL target instead.")
        gwl_mode = DEFAULT_GWL_MODE

    try:
        gwl_value = float(extracted.get("gwl_value"))
    except (TypeError, ValueError):
        assumptions.append("Could not parse a warming-level value — defaulting to 2.0.")
        gwl_value = DEFAULT_GWL_VALUE

    bbox = geocode(region)
    status = "ready_to_narrate"

    if bbox is None and attempts < MAX_TOOL_ATTEMPTS:
        return {
            "attempts": attempts,
            "status": "resolving",
            "language": language,
            "assumptions": assumptions,
            "tool_calls": [ToolCall(tool="geocode", args={"region": region}, result=None)],
        }

    if bbox is None:
        # Exhausted retries — degrade to a default area, but still compute a real, complete
        # answer for it. Real bug found live (2026-08-17): this used to return early here
        # without ever computing drivers/sector_impact, and the graph routes "degraded" the
        # same as "ready_to_narrate" into retrieve_context — which crashed on the missing
        # data (AttributeError: 'NoneType' object has no attribute 'items'). "Bounded resolve,
        # else honest degrade" always meant *deliver a full answer* with a stated assumption,
        # never a dead end — a crash is exactly the dead end that pattern was built to avoid.
        bbox = BBox(min_lat=-5.0, min_lon=-5.0, max_lat=5.0, max_lon=5.0)
        status = "degraded"
        assumptions = [
            *assumptions,
            f"Could not resolve '{region}' to a specific location — showing a default area.",
        ]

    window_tool = WINDOW_TOOLS[gwl_mode]
    window = window_tool(gwl_value)
    gwl = f"{gwl_value} [{gwl_mode}] ({window.start_year}-{window.end_year})"

    sector_tool = SECTOR_TOOLS[sector]
    impact_grid, sector_impact, sector_assumptions = sector_tool(bbox, window)
    driver_grids, drivers, hazard_assumptions = compute_hazard_drivers(bbox, window)
    assumptions = assumptions + sector_assumptions + hazard_assumptions

    return {
        "attempts": attempts,
        "status": status,
        "region": region,
        "bbox": bbox,
        "gwl": gwl,
        "sector": sector,
        "language": language,
        "impact_grid": impact_grid,
        "driver_grids": driver_grids,
        "drivers": drivers,
        "sector_impact": sector_impact,
        "assumptions": assumptions,
        "tool_calls": [
            ToolCall(tool="router", args={"query": state["query"]}, result=str(extracted)),
            ToolCall(tool="geocode", args={"region": region}, result="ok"),
            ToolCall(tool=f"sector:{sector}", args={"bbox": str(bbox), "window": str(window)}, result="ok"),
            ToolCall(tool="hazard", args={"bbox": str(bbox), "window": str(window)}, result="ok"),
        ],
    }
