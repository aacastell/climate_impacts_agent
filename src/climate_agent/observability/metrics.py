import mlflow

from climate_agent.agents.langgraph.state import AgentState


def compute_query_metrics(final_state: AgentState) -> dict[str, str]:
    """Derived observability metrics from a completed graph run's final state.

    Args: final_state — the graph's final AgentState after invoke().
    Returns: dict of metric name -> string value (MLflow trace tags are string-valued).
    """
    return {
        "tool_attempts": str(final_state["attempts"]),
        "narration_attempts": str(final_state["narration_attempts"]),
        "degraded": str(final_state["status"] == "degraded"),
        "grounded": str(final_state["grounded"]),
        "assumption_count": str(len(final_state.get("assumptions") or [])),
    }


def record_query_metrics(final_state: AgentState) -> None:
    """Attach derived metrics to the current active MLflow trace as tags.

    Args: final_state — the graph's final AgentState after invoke().
    """
    mlflow.update_current_trace(tags=compute_query_metrics(final_state))
