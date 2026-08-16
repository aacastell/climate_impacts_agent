from climate_agent.agents.langgraph.state import AgentState

MAX_NARRATION_ATTEMPTS = 2


def verify_narration(state: AgentState) -> dict:
    """Check that the narration's claims trace back to resolved state (groundedness).

    Loops back to narrate (via the graph's conditional edge) on failure, up to
    MAX_NARRATION_ATTEMPTS, then degrades rather than looping forever.

    Args: state — current graph state.
    Returns: partial state update.
    """
    narration = state["narration"] or ""
    region = state["region"] or ""
    sector = state["sector"] or ""

    grounded = region in narration and sector.lower() in narration.lower()

    if grounded or state["narration_attempts"] >= MAX_NARRATION_ATTEMPTS:
        return {"grounded": grounded, "status": "done" if grounded else "degraded"}

    return {"grounded": False, "status": "verifying"}
