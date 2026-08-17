import os

import httpx

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
NARRATOR_MODEL = "qwen2.5:7b"


def generate_narration(
    region: str,
    gwl: str,
    sector: str,
    sector_impact: str,
    drivers: dict[str, str],
    driver_context: dict[str, str],
) -> str:
    """Generate a short narration grounded in resolved climate-impact data, via a local Ollama LLM.

    Args: region, gwl, sector, sector_impact, drivers, driver_context — resolved state fields.
    Returns: narration text.
    """
    prompt = (
        "Write a short (3-4 sentence) narration explaining climate impacts, grounded ONLY in "
        "the data below. Do not invent numbers not shown here.\n\n"
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
