from collections.abc import Callable

import mlflow


def traced(func: Callable, name: str, span_type: str) -> Callable:
    """Wrap a function with MLflow tracing (auto-captures inputs, outputs, and latency).

    Nested calls to other traced functions automatically become child spans of whichever
    traced function is currently executing, so wrapping the top-level query() function and
    each graph node gives a full per-query trace tree with no extra wiring.

    Args: func — the function to trace. name — display name for the span.
    span_type — an mlflow.entities.SpanType value (e.g. SpanType.AGENT).
    Returns: the traced function, same call signature as func.
    """
    return mlflow.trace(func, name=name, span_type=span_type)
