from climate_agent.agents.langgraph.state import AgentState
from climate_agent.tools.narrator import generate_narration


def narrate(state: AgentState) -> dict:
    """Write the sector-layer narration from resolved state, via a real LLM (Ollama).

    Args: state — current graph state.
    Returns: partial state update.
    """
    narration = generate_narration(
        region=state["region"],
        gwl=state["gwl"],
        sector=state["sector"],
        sector_impact=state["sector_impact"],
        drivers=state["drivers"],
        driver_context=state["driver_context"],
        language=state.get("language") or "English",
    )
    if state.get("assumptions"):
        narration += " Note: " + " ".join(state["assumptions"])

    return {
        "narration": narration,
        "narration_attempts": state["narration_attempts"] + 1,
        "status": "verifying",
    }
