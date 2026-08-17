from climate_agent.rag.index import search


def retrieve_driver_context(sector: str, drivers: dict[str, str]) -> dict[str, str]:
    """Per-driver grounding text for a sector, retrieved from a real corpus via semantic search.

    Falls back to an unfiltered (untagged) search if the sector-tagged corpus has no match, so a
    sector with thinner real-source coverage still gets the closest available real grounding
    rather than nothing.

    Args: sector — resolved sector name (e.g. "Agriculture"). drivers — computed driver
    summaries (e.g. "+1.35°C vs. baseline."), used to formulate each driver's retrieval query.
    Returns: dict mapping each driver name to a short markdown-ready context string, citing its
    real source.
    """
    context: dict[str, str] = {}
    for driver, summary in drivers.items():
        query = f"{sector} impact of {driver.lower()}: {summary}"
        results = search(query, top_k=1, tag=sector) or search(query, top_k=1)
        if results:
            top = results[0]
            context[driver] = f"{top['text']} (Source: {top['source_title']})"
        else:
            context[driver] = "No grounding context available for this driver."
    return context
