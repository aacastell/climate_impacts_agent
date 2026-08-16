from climate_agent.schemas import BBox, GridCell
from climate_agent.tools.analysis import grid_cells_in_bbox, stub_value

DRIVER_RANGES = {
    "Avg. temperature change": (0.5, 4.0, "°C"),
    "Avg. precipitation change": (-20.0, 10.0, "% precipitation"),
    "Heat extreme days": (0.0, 40.0, " days/year above 35°C"),
    "Precipitation extreme days": (0.0, 20.0, " consecutive dry days"),
}


def compute_hazard_drivers(bbox: BBox, gwl: str) -> tuple[dict[str, list[GridCell]], dict[str, str]]:
    """Change-vs-baseline grids and summary text for all four climate hazard drivers.

    Stub: random values, ignores gwl (accepted now so the signature won't change once real
    per-GWL computation replaces this body).

    Args: bbox — target area. gwl — resolved warming level (e.g. "2°C").
    Returns: (driver_grids, drivers) — per-driver GridCell lists, and per-driver summary strings.
    """
    cells = grid_cells_in_bbox(bbox)

    driver_grids: dict[str, list[GridCell]] = {}
    drivers: dict[str, str] = {}
    for driver, (low, high, unit) in DRIVER_RANGES.items():
        grid = [
            GridCell(lat=lat, lon=lon, value=stub_value(driver, lat, lon, low, high))
            for lat, lon in cells
        ]
        driver_grids[driver] = grid
        avg = sum(c.value for c in grid) / len(grid)
        sign = "+" if avg >= 0 else ""
        drivers[driver] = f"{sign}{avg:.1f}{unit} (stub)"

    return driver_grids, drivers
