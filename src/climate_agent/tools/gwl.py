def resolve_gwl(gwl_text: str) -> str:
    """Normalize a free-text warming-level target. Stub: passthrough only.

    Real resolution (per-model year-window crossing, ISIMIP GWL protocol) needs actual per-model
    temperature trajectories this project doesn't have yet — faking that structure now would risk
    locking in a guessed shape rather than the real one, so this stays a minimal passthrough
    until item 10 lands.

    Args: gwl_text — warming level as typed/extracted (e.g. "2°C", "2 degrees").
    Returns: gwl_text, unchanged (whitespace-trimmed).
    """
    return gwl_text.strip()
