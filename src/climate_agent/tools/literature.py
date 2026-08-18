import re

from climate_agent.rag.index import search

MAX_CONTEXT_CHARS = 320  # fallback display length if synthesis fails and raw text has to be shown

# Real, verified fix (2026-08-17): the old query ("Agriculture impact of avg. precipitation
# change: +2.00mm/day vs. baseline.") was dominated by generic numeric/boilerplate phrasing the
# embedding model latched onto, so most drivers converged on the same generic disclaimer chunk
# regardless of actual topic — confirmed live via direct search() comparisons. Plain topical
# keywords differentiate real results correctly (verified); the numeric summary is still used
# for display (`drivers` dict) but dropped from the retrieval query entirely.
DRIVER_QUERY_TERMS = {
    "Avg. temperature change": "rising temperatures warming climate impact",
    "Avg. precipitation change": "rainfall precipitation change impact",
    "Heat extreme days": "extreme heat waves high temperature days impact",
    "Precipitation extreme days": "drought dry spells low rainfall impact",
}


def clean_text(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Collapse whitespace/newlines from raw scraped text and truncate to a display-friendly length.

    Real corpus chunks (rag/corpus.py) keep unflattened get_text() newlines, so raw display
    breaks mid-sentence, and full chunks run 1000+ characters. Used as the fallback display
    format if LLM synthesis (narrator.synthesize_driver_context) fails and raw retrieved text
    has to be shown as-is instead.

    Args: text — raw retrieved chunk text. max_chars — truncation length.
    Returns: single-line, truncated text (word-boundary cut, "..." suffix if truncated).
    """
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    cut = collapsed[:max_chars].rsplit(" ", 1)[0]
    return f"{cut}..."


def retrieve_driver_context(sector: str, drivers: dict[str, str]) -> dict[str, tuple[str, str]]:
    """Real grounding source per driver, retrieved via semantic search over the NASA corpus.

    Returns raw (chunk_text, source_title) pairs rather than display-ready text — the caller
    (narrator.synthesize_driver_context) turns this into a short, query-specific blurb instead
    of dumping the retrieved paragraph verbatim (real UX bug found live 2026-08-17: raw RAG
    dumps read as disconnected literature, not an explanation of this query's actual numbers).

    Uses topical keywords (DRIVER_QUERY_TERMS) rather than the numeric summary for the retrieval
    query — a raw numeric-heavy query was found (verified live) to dilute retrieval relevance,
    causing unrelated drivers to converge on the same generic chunk. Avoids reusing the same
    chunk across two different drivers in one call for the same reason.

    Args: sector — resolved sector name. drivers — computed driver summaries (used only to know
    which drivers need grounding, not fed into the retrieval query itself).
    Returns: dict mapping driver name -> (chunk_text, source_title); ("", "") if nothing found.
    """
    context: dict[str, tuple[str, str]] = {}
    used_texts: set[str] = set()
    for driver in drivers:
        query_terms = DRIVER_QUERY_TERMS.get(driver, driver.lower())
        query = f"{sector} {query_terms}"
        results = search(query, top_k=5, tag=sector) or search(query, top_k=5)
        pick = next((r for r in results if r["text"] not in used_texts), results[0] if results else None)
        if pick:
            used_texts.add(pick["text"])
            context[driver] = (pick["text"], pick["source_title"])
        else:
            context[driver] = ("", "")
    return context
