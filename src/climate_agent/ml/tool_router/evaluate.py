import json
from pathlib import Path

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from climate_agent.ml.tool_router.generate_data import EVAL_FILE, SYSTEM_PROMPT

VALID_SECTORS = {"Agriculture", "Biome"}
VALID_MODES = {"gwl", "heat_extreme", "precip_extreme"}
GWL_VALUE_TOLERANCE = 0.5


def _load_eval_examples() -> list[dict]:
    """Read eval.jsonl into a list of {query, target} dicts."""
    return [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line]


def parse_router_output(text: str) -> dict | None:
    """Parse a router completion into a target-shaped dict, or None if it isn't valid JSON.

    Args: text — raw model completion.
    Returns: parsed dict, or None on any parse/shape failure.
    """
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        parsed = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None
    if not all(k in parsed for k in ("region", "sector", "gwl_mode", "gwl_value")):
        return None
    return parsed


def _fields_correct(predicted: dict, target: dict) -> dict[str, bool]:
    """Per-field correctness of a parsed prediction against its target.

    Args: predicted, target — dicts with region/sector/gwl_mode/gwl_value.
    Returns: dict of field name -> bool.
    """
    region_ok = str(predicted.get("region", "")).strip().lower() == target["region"].strip().lower()
    sector_ok = predicted.get("sector") == target["sector"]
    mode_ok = predicted.get("gwl_mode") == target["gwl_mode"]
    try:
        value_ok = abs(float(predicted.get("gwl_value")) - target["gwl_value"]) <= GWL_VALUE_TOLERANCE
    except (TypeError, ValueError):
        value_ok = False
    return {"region": region_ok, "sector": sector_ok, "gwl_mode": mode_ok, "gwl_value": value_ok}


def generate_completion(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, query: str) -> str:
    """Run the router prompt through a model and return its raw text completion.

    Args: model, tokenizer — a loaded causal LM and its tokenizer. query — user question.
    Returns: decoded completion text (assistant turn only).
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    completion_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def evaluate(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> dict:
    """Run every eval example through the model and score field-level + exact-match accuracy.

    Args: model, tokenizer — a loaded causal LM and its tokenizer.
    Returns: {"n": int, "exact_match": float, "field_accuracy": {field: float}, "parse_failures": int}.
    """
    examples = _load_eval_examples()
    field_correct = {"region": 0, "sector": 0, "gwl_mode": 0, "gwl_value": 0}
    exact_match = 0
    parse_failures = 0

    for example in examples:
        completion = generate_completion(model, tokenizer, example["query"])
        predicted = parse_router_output(completion)
        if predicted is None:
            parse_failures += 1
            continue
        correctness = _fields_correct(predicted, example["target"])
        for field, ok in correctness.items():
            field_correct[field] += int(ok)
        if all(correctness.values()):
            exact_match += 1

    n = len(examples)
    return {
        "n": n,
        "exact_match": exact_match / n,
        "field_accuracy": {field: count / n for field, count in field_correct.items()},
        "parse_failures": parse_failures,
    }


def print_report(label: str, results: dict) -> None:
    print(f"=== {label} (n={results['n']}) ===")
    print(f"Exact match: {results['exact_match']:.1%}")
    for field, acc in results["field_accuracy"].items():
        print(f"  {field}: {acc:.1%}")
    print(f"Parse failures: {results['parse_failures']}")
