import json
import os

import httpx

from climate_agent.tools.literature import clean_text

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
NARRATOR_MODEL = "qwen2.5:7b"


def generate_narration(
    region: str,
    gwl: str,
    sector: str,
    sector_impact: str,
    drivers: dict[str, str],
    driver_context: dict[str, str],
    language: str = "English",
) -> str:
    """Generate a short narration grounded in resolved climate-impact data, via a local Ollama LLM.

    Args: region, gwl, sector, sector_impact, drivers, driver_context — resolved state fields.
    language — human-readable language name (from tools/language.py's real detection, not
    guessed) to respond in; source data/labels stay English regardless, the model translates.
    Returns: narration text.
    """
    prompt = (
        f"Write a short (3-4 sentence) narration IN {language.upper()} explaining climate "
        "impacts, grounded ONLY in the data below. Do not invent numbers not shown here.\n\n"
        f"Region: {region}\n"
        f"GWL: {gwl}\n"
        f"Sector: {sector}\n"
        f"Sector impact: {sector_impact}\n"
        f"Drivers: {drivers}\n"
        f"Driver context (real scientific sources): {driver_context}\n"
    )
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": NARRATOR_MODEL, "prompt": prompt, "stream": False},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["response"].strip()

def synthesize_driver_context(
    sector: str,
    drivers: dict[str, str],
    raw_context: dict[str, tuple[str, str]],
    language: str = "English",
) -> tuple[dict[str, str], bool]:
    """Write a short, query-specific sentence per driver, grounded in retrieved NASA source text
    and this query's actual computed value — instead of displaying the raw retrieved paragraph
    verbatim. Matches narration's approach (synthesize from real data, don't just quote it) and
    handles translation in the same call rather than a separate pass. Real UX fix (2026-08-17):
    driver text next to each map used to just dump a scraped NASA paragraph, unrelated to the
    specific region/value being shown, which read as disconnected literature rather than an
    explanation of the map.

    Falls back to a cleaned (untranslated, unsynthesized) version of the raw retrieved text if
    the model's response can't be parsed as the expected JSON shape, so a real, grounded display
    string is always available even if synthesis fails.

    Args: sector — resolved sector name. drivers — driver name -> numeric summary (e.g.
    "+1.35°C vs. baseline."). raw_context — driver name -> (retrieved chunk text, source title),
    from literature.retrieve_driver_context. language — human-readable target language name.
    Returns: (driver_context, succeeded) — driver name -> short markdown-ready sentence with
    source citation, and whether synthesis succeeded.
    """
    fallback = {
        driver: (
            f"{clean_text(raw_context[driver][0])} (Source: {raw_context[driver][1]})"
            if raw_context.get(driver, ("", ""))[0]
            else "No grounding context available for this driver."
        )
        for driver in drivers
    }
    drivers_with_context = {d: s for d, s in drivers.items() if raw_context.get(d, ("", ""))[0]}
    if not drivers_with_context:
        return fallback, True

    # Citation is appended by code, not the model (real bug found live 2026-08-17: asked to
    # reproduce "(Source: <source_title>)" itself, the model silently truncated the title —
    # citation text the code already knows exactly should never depend on the LLM getting it
    # right). The prompt also explicitly requires explaining the real-world CONSEQUENCE, not
    # just restating computed_value — first version just translated the number back, which
    # wasn't wrong, just useless (that's what `drivers`/the map caption already show).
    items = "\n\n".join(
        f'"{driver}": computed_value="{summary}", source_text="{clean_text(raw_context[driver][0], max_chars=600)}"'
        for driver, summary in drivers_with_context.items()
    )
    prompt = (
        f"For each labeled driver below, write ONE short sentence (max 30 words) IN {language.upper()} "
        f"explaining the real-world CONSEQUENCE of this change for {sector.lower()} — effects on "
        "yields, growing conditions, or food security — using the actual findings in source_text. "
        "You may briefly reference computed_value, but do not just restate it: the sentence must "
        "explain what it means, grounded in source_text's real findings. Do not invent findings not "
        "present in source_text. Do not include any source citation or attribution — that is added "
        "separately. Return ONLY a JSON object mapping each driver label to its sentence — no "
        f"commentary, no markdown code fences.\n\n{items}"
    )

    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": NARRATOR_MODEL, "prompt": prompt, "stream": False},
            timeout=120.0,
        )
        response.raise_for_status()
        raw = response.json()["response"].strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        synthesized = json.loads(raw)
        if not set(drivers_with_context).issubset(synthesized):
            return fallback, False
    except (httpx.HTTPError, ValueError, KeyError):
        return fallback, False

    result = dict(fallback)
    result.update({
        driver: f"{synthesized[driver]} (Source: {raw_context[driver][1]})" for driver in drivers_with_context
    })
    return result, True
