DRIVER_CONTEXT_TEXT = {
    "Agriculture": {
        "Avg. temperature change": "Warmer averages shift planting windows (stub — RAG not wired yet).",
        "Avg. precipitation change": "Reduced rainfall raises irrigation demand (stub — RAG not wired yet).",
        "Heat extreme days": "More hot days stress crops during flowering (stub — RAG not wired yet).",
        "Precipitation extreme days": "Longer dry spells compound soil deficits (stub — RAG not wired yet).",
    },
    "Biome": {
        "Avg. temperature change": "Warming shifts species range boundaries (stub — RAG not wired yet).",
        "Avg. precipitation change": "Drying favors drought-adapted vegetation (stub — RAG not wired yet).",
        "Heat extreme days": "Heat extremes raise wildfire risk (stub — RAG not wired yet).",
        "Precipitation extreme days": "Dry spells reduce vegetation resilience (stub — RAG not wired yet).",
    },
}


def retrieve_driver_context(sector: str, drivers: dict[str, str]) -> dict[str, str]:
    """Per-driver grounding text for a sector, anchored to the computed change data.

    Stub: static lookup, ignores drivers (accepted now so the signature won't change once real
    retrieval — querying against actual computed magnitudes, not just sector name — replaces this body).

    Args: sector — resolved sector name (e.g. "Agriculture"). drivers — computed driver summaries
        (e.g. "+2.1°C (stub)"), what a real implementation would use to formulate retrieval queries.
    Returns: dict mapping each driver name to a short markdown-ready context string.
    """
    return DRIVER_CONTEXT_TEXT.get(sector, {})
