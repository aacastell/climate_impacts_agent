import mlflow
from mlflow.entities import SpanType

from climate_agent.agents.langgraph.graph import compiled_graph
from climate_agent.agents.langgraph.state import AgentState
from climate_agent.observability.metrics import record_query_metrics
from climate_agent.schemas import QueryResponse


@mlflow.trace(name="query", span_type=SpanType.CHAIN)
def query(query_text: str) -> QueryResponse:
    """Run the LangGraph orchestration for a query, mapped to the API's response shape.

    Traced end to end via MLflow (root span here, child spans per graph node) — see
    observability/tracing.py and observability/metrics.py.

    Args: query_text — the user's raw query.
    Returns: QueryResponse, built from the graph's final resolved state.
    """
    initial_state: AgentState = {
        "query": query_text,
        "tool_calls": [],
        "attempts": 0,
        "status": "resolving",
        "region": None,
        "bbox": None,
        "gwl": None,
        "sector": None,
        "language": None,
        "impact_grid": None,
        "driver_grids": None,
        "drivers": None,
        "driver_context": None,
        "sector_impact": None,
        "narration": None,
        "narration_attempts": 0,
        "grounded": False,
        "assumptions": [],
    }

    final_state = compiled_graph.invoke(initial_state)
    record_query_metrics(final_state)

    return QueryResponse(
        region=final_state["region"],
        gwl=final_state["gwl"],
        sector=final_state["sector"],
        language=final_state["language"],
        bbox=final_state["bbox"],
        sector_impact=final_state["sector_impact"],
        narration=final_state["narration"],
        impact_grid=final_state["impact_grid"],
        driver_grids=final_state["driver_grids"],
        drivers=final_state["drivers"],
        driver_context=final_state["driver_context"],
    )
