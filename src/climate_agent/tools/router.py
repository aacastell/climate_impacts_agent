from functools import lru_cache

from climate_agent.ml.tool_router.evaluate import generate_completion, parse_router_output
from climate_agent.ml.tool_router.registry import load_router_model


@lru_cache(maxsize=1)
def _model_and_tokenizer():
    return load_router_model()


def extract_query_fields(query: str) -> dict | None:
    """Extract region/sector/gwl_mode/gwl_value from a free-text query via the fine-tuned router.

    Args: query — the user's raw query text.
    Returns: {"region", "sector", "gwl_mode", "gwl_value"} dict, or None if the model's output
    couldn't be parsed as valid JSON with the expected keys (caller degrades on None).
    """
    model, tokenizer = _model_and_tokenizer()
    completion = generate_completion(model, tokenizer, query)
    return parse_router_output(completion)
