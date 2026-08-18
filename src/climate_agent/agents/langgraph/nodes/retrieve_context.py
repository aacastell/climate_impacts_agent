from climate_agent.agents.langgraph.state import AgentState, ToolCall
from climate_agent.tools.literature import retrieve_driver_context
from climate_agent.tools.narrator import synthesize_driver_context


def retrieve_context(state: AgentState) -> dict:
    """Fetch RAG grounding for the resolved sector's drivers, and synthesize a short,
    query-specific, translated blurb per driver rather than displaying the raw retrieved
    paragraph verbatim.

    Args: state — current graph state.
    Returns: partial state update.
    """
    sector = state["sector"]
    drivers = state["drivers"]
    language = state.get("language") or "English"

    raw_context = retrieve_driver_context(sector, drivers)
    driver_context, synthesized_ok = synthesize_driver_context(sector, drivers, raw_context, language)

    assumptions = []
    if not synthesized_ok:
        assumptions.append("Could not synthesize driver context via the LLM — showing raw source excerpts instead.")

    return {
        "driver_context": driver_context,
        "assumptions": assumptions,
        "tool_calls": [ToolCall(tool="rag_retrieve", args={"sector": sector}, result="ok")],
    }
