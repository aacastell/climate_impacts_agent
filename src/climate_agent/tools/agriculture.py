from climate_agent.schemas import BBox, GridCell
from climate_agent.tools.analysis import grid_cells_in_bbox, stub_value

SECTOR_IMPACT_TEXT = "Maize yield projected to decline (stub — real tool not wired yet)."


def compute_agriculture(bbox: BBox, gwl: str) -> tuple[list[GridCell], str]:
    """Change-vs-baseline agriculture impact grid and summary text.

    Stub: random values, ignores gwl (accepted now so the signature won't change once real
    ISIMIP-backed computation replaces this body).

    Args: bbox — target area. gwl — resolved warming level (e.g. "2°C").
    Returns: (impact_grid, sector_impact) — per-cell impact values, and a summary string.
    """
    cells = grid_cells_in_bbox(bbox)
    impact_grid = [
        GridCell(lat=lat, lon=lon, value=stub_value("impact:agriculture", lat, lon, 0.0, 1.0))
        for lat, lon in cells
    ]
    return impact_grid, SECTOR_IMPACT_TEXT
