from climate_agent.agents.langgraph.state import AgentState
from climate_agent.guardrails import narration_numbers_grounded

MAX_NARRATION_ATTEMPTS = 2
SECTOR_STEM_LEN = 8  # "agricult" covers agriculture/agricultural; "biome" is unaffected (shorter)


def verify_narration(state: AgentState) -> dict:
    """Check that the narration's claims trace back to resolved state (groundedness).

    Two checks: sector/region are topically mentioned (word-stem prefix, not exact substring —
    a real LLM naturally varies word forms like "agricultural" vs "agriculture", which an exact
    match would wrongly fail), and every stat-scale number in the narration is numerically close
    to a real driver/sector-impact value (item 16's guardrail — a topical check alone can't catch
    a model inventing a plausible-sounding but wrong statistic).

    Loops back to narrate (via the graph's conditional edge) on failure, up to
    MAX_NARRATION_ATTEMPTS, then degrades rather than looping forever.

    Args: state — current graph state.
    Returns: partial state update.
    """
    narration = state["narration"] or ""
    region = state["region"] or ""
    sector = state["sector"] or ""

    topically_grounded = region in narration and sector.lower()[:SECTOR_STEM_LEN] in narration.lower()
    numerically_grounded = narration_numbers_grounded(narration, state["drivers"] or {}, state["sector_impact"] or "")
    grounded = topically_grounded and numerically_grounded

    if grounded or state["narration_attempts"] >= MAX_NARRATION_ATTEMPTS:
        return {"grounded": grounded, "status": "done" if grounded else "degraded"}

    return {"grounded": False, "status": "verifying"}
