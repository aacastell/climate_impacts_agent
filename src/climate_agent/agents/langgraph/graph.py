from langgraph.graph import END, START, StateGraph
from mlflow.entities import SpanType

from climate_agent.agents.langgraph.nodes.narrate import narrate
from climate_agent.agents.langgraph.nodes.retrieve_context import retrieve_context
from climate_agent.agents.langgraph.nodes.understand_and_call_tools import (
    understand_and_call_tools,
)
from climate_agent.agents.langgraph.nodes.verify_narration import verify_narration
from climate_agent.agents.langgraph.state import AgentState
from climate_agent.observability.tracing import traced


def route_after_understanding(state: AgentState) -> str:
    return "understand_and_call_tools" if state["status"] == "resolving" else "retrieve_context"

def route_after_verification(state: AgentState) -> str:
    return "narrate" if state["status"] == "verifying" else END

graph = StateGraph(AgentState)

graph.add_node("understand_and_call_tools", traced(understand_and_call_tools, "understand_and_call_tools", SpanType.AGENT))
graph.add_node("retrieve_context", traced(retrieve_context, "retrieve_context", SpanType.RETRIEVER))
graph.add_node("narrate", traced(narrate, "narrate", SpanType.LLM))
graph.add_node("verify_narration", traced(verify_narration, "verify_narration", SpanType.GUARDRAIL))

graph.add_edge(START, "understand_and_call_tools")
graph.add_conditional_edges("understand_and_call_tools", route_after_understanding)
graph.add_edge("retrieve_context", "narrate")
graph.add_edge("narrate", "verify_narration")
graph.add_conditional_edges("verify_narration", route_after_verification)

compiled_graph = graph.compile()
