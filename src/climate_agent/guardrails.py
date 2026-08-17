import re

MAX_QUERY_LENGTH = 500
INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|prior|above) instructions",
    r"disregard (all |the )?(previous|prior|above) instructions",
    r"you are now",
    r"reveal (your |the )?system prompt",
]

NUMBER_RE = re.compile(r"-?\d+\.?\d*")
YEAR_LIKE_THRESHOLD = 1000  # numbers at/above this magnitude are treated as years, not stats
DEFAULT_TOLERANCE = 0.5


def validate_query(query_text: str) -> str | None:
    """Input guardrail: check a raw query for obvious abuse before it reaches any model.

    Args: query_text — the user's raw query.
    Returns: a rejection reason if the query should be blocked, else None.
    """
    text = (query_text or "").strip()
    if not text:
        return "Query is empty."
    if len(text) > MAX_QUERY_LENGTH:
        return f"Query is too long ({len(text)} characters, max {MAX_QUERY_LENGTH})."

    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return "Query contains a potential prompt-injection pattern."
    return None


def _numbers_in_text(text: str) -> list[float]:
    """Every numeric value mentioned in a text (handles plain and signed decimals)."""
    return [float(match) for match in NUMBER_RE.findall(text)]


def narration_numbers_grounded(
    narration: str, drivers: dict[str, str], sector_impact: str, tolerance: float = DEFAULT_TOLERANCE
) -> bool:
    """Output guardrail: every stat-scale number in the narration must be close to a real
    number drawn from the resolved driver/sector data.

    A topical check (does the narration mention the right region/sector) doesn't catch a model
    inventing a plausible-sounding but wrong statistic — this checks numeric fidelity directly.
    Year-like numbers (>= YEAR_LIKE_THRESHOLD, e.g. "2054-2059" from the GWL window) are excluded
    from the check — they're legitimately part of the prompt context, not driver statistics, and
    checking them against driver-value tolerances would produce false positives.

    Args: narration — generated narration text. drivers, sector_impact — real computed values
    the narration should be grounded in. tolerance — max allowed absolute difference for a match.
    Returns: True if every stat-scale narration number matches some real number within tolerance,
    or if there's nothing real to check against (avoids failing spuriously on an empty state).
    """
    real_numbers = _numbers_in_text(sector_impact) + [n for v in drivers.values() for n in _numbers_in_text(v)]
    if not real_numbers:
        return True

    claimed_numbers = [n for n in _numbers_in_text(narration) if abs(n) < YEAR_LIKE_THRESHOLD]
    return all(any(abs(claimed - real) <= tolerance for real in real_numbers) for claimed in claimed_numbers)
