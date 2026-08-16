from climate_agent.agents.langgraph.state import AgentState, ToolCall
from climate_agent.tools.literature import retrieve_driver_context


def retrieve_context(state: AgentState) -> dict:
    """Fetch RAG grounding text for the resolved sector's drivers.

    Args: state — current graph state.
    Returns: partial state update.
    """
    sector = state["sector"]
    drivers = state["drivers"]
    driver_context = retrieve_driver_context(sector, drivers)
    return {
        "driver_context": driver_context,
        "tool_calls": [ToolCall(tool="rag_retrieve", args={"sector": sector}, result="stub")],
    }
