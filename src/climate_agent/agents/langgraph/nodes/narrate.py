from climate_agent.agents.langgraph.state import AgentState


def narrate(state: AgentState) -> dict:
    """Write the sector-layer narration from resolved state. Stub: templated text.

    Args: state — current graph state.
    Returns: partial state update.
    """
    region = state["region"]
    gwl = state["gwl"]
    sector = state["sector"]

    narration = (
        f"At {gwl} of additional warming above today's climate, {region} shows "
        f"meaningful {sector.lower()} impacts alongside intensifying heat and "
        f"precipitation extremes. (Stub narration — real tools, RAG grounding, "
        f"and model layer aren't wired in yet.)"
    )
    if state.get("assumptions"):
        narration += " Note: " + " ".join(state["assumptions"])

    return {
        "narration": narration,
        "narration_attempts": state["narration_attempts"] + 1,
        "status": "verifying",
    }
