import math
import random

from climate_agent.schemas import BBox, GridCell, QueryResponse

GRID_RESOLUTION_DEG = 0.5  # matches the ISIMIP/GGCMI global simulation grid
MIN_CELLS_PER_SIDE = 3  # ensures a tiny region still renders as a visible patch, not a single point

GEOCODE_LOOKUP = {
    "tolima, colombia": {"min_lat": 3.3, "min_lon": -76.0, "max_lat": 5.4, "max_lon": -74.3},
    "colombia": {"min_lat": -4.2, "min_lon": -79.0, "max_lat": 12.5, "max_lon": -66.9},
    "mexico": {"min_lat": 14.5, "min_lon": -118.4, "max_lat": 32.7, "max_lon": -86.7},
    "united states": {"min_lat": 24.5, "min_lon": -124.8, "max_lat": 49.4, "max_lon": -66.9},
    "china": {"min_lat": 18.2, "min_lon": 73.5, "max_lat": 53.6, "max_lon": 134.8},
    "brazil": {"min_lat": -33.7, "min_lon": -73.9, "max_lat": 5.3, "max_lon": -34.8},
}
DEFAULT_BBOX = {"min_lat": -5.0, "min_lon": -5.0, "max_lat": 5.0, "max_lon": 5.0}

DRIVER_RANGES = {
    "Avg. temperature change": (0.5, 4.0, "°C"),
    "Avg. precipitation change": (-20.0, 10.0, "% precipitation"),
    "Heat extreme days": (0.0, 40.0, " days/year above 35°C"),
    "Precipitation extreme days": (0.0, 20.0, " consecutive dry days"),
}

SECTOR_IMPACT_TEXT = {
    "Agriculture": "Maize yield projected to decline (placeholder — real tool not wired yet).",
    "Biome": "Elevated probability of biome transition toward a drier vegetation class (placeholder — real tool not wired yet).",
}

# Qualitative counterpart to the quantitative `drivers` stats -- in the real
# system this is RAG-grounded, cached per (sector, driver), and anchored to
# the actual resolved numbers. Placeholder text until that's wired in.
DRIVER_CONTEXT_TEXT = {
    "Agriculture": {
        "Avg. temperature change": "Warmer averages shift planting windows and can push heat-sensitive crops past their optimal growth range (placeholder — real RAG grounding not wired yet).",
        "Avg. precipitation change": "Reduced rainfall raises irrigation demand and drought stress during key growth stages (placeholder — real RAG grounding not wired yet).",
        "Heat extreme days": "More days above 35°C accelerate crop water stress and can reduce yields during flowering (placeholder — real RAG grounding not wired yet).",
        "Precipitation extreme days": "Longer dry spells compound soil moisture deficits across a growing season (placeholder — real RAG grounding not wired yet).",
    },
    "Biome": {
        "Avg. temperature change": "Sustained warming shifts species range boundaries and can favor more heat-tolerant vegetation (placeholder — real RAG grounding not wired yet).",
        "Avg. precipitation change": "Drying trends favor a transition toward more drought-adapted vegetation classes (placeholder — real RAG grounding not wired yet).",
        "Heat extreme days": "More frequent heat extremes raise wildfire risk and heat-stress mortality in sensitive species (placeholder — real RAG grounding not wired yet).",
        "Precipitation extreme days": "Extended dry spells reduce vegetation resilience and increase die-back risk (placeholder — real RAG grounding not wired yet).",
    },
}

# No real parser yet, so free-text queries always resolve to these regardless
# of what was typed -- a real router replaces this entirely, not just its values.
DEFAULT_QUERY_REGION = "Tolima, Colombia"
DEFAULT_QUERY_GWL = "2°C"
DEFAULT_QUERY_SECTOR = "Agriculture"


def _geocode(region_text: str) -> BBox:
    bbox = GEOCODE_LOOKUP.get(region_text.strip().lower(), DEFAULT_BBOX)
    return BBox(**bbox)


def _grid_line(min_value: float, max_value: float, resolution: float) -> list[float]:
    # Cell centers on the real global grid sit at a half-resolution offset
    # (e.g. -89.75, -89.25, ... for a 0.5deg grid) — not at round numbers.
    start = math.floor(min_value / resolution) * resolution + resolution / 2
    values = []
    value = start
    while value <= max_value:
        values.append(round(value, 3))
        value += resolution
    return values


def _resolution_for_bbox(bbox: BBox, base_resolution: float, min_cells: int) -> float:
    lat_span = bbox.max_lat - bbox.min_lat
    lon_span = bbox.max_lon - bbox.min_lon
    resolution = base_resolution
    if lat_span > 0:
        resolution = min(resolution, lat_span / min_cells)
    if lon_span > 0:
        resolution = min(resolution, lon_span / min_cells)
    return max(resolution, 1e-4)  # numerical floor (~11m) so a literal point can't yield zero cells


def _grid_cells_in_bbox(bbox: BBox) -> list[tuple[float, float]]:
    resolution = _resolution_for_bbox(bbox, GRID_RESOLUTION_DEG, MIN_CELLS_PER_SIDE)
    lats = _grid_line(bbox.min_lat, bbox.max_lat, resolution)
    lons = _grid_line(bbox.min_lon, bbox.max_lon, resolution)
    return [(lat, lon) for lat in lats for lon in lons]


def _mock_value(seed: str, lat: float, lon: float, low: float, high: float) -> float:
    rng = random.Random(f"{seed}:{lat:.3f}:{lon:.3f}")
    return round(rng.uniform(low, high), 2)


def query(query_text: str) -> QueryResponse:
    region = DEFAULT_QUERY_REGION
    gwl = DEFAULT_QUERY_GWL
    sector = DEFAULT_QUERY_SECTOR

    bbox = _geocode(region)
    cells = _grid_cells_in_bbox(bbox)

    impact_grid = [
        GridCell(lat=lat, lon=lon, value=_mock_value(f"impact:{sector}", lat, lon, 0.0, 1.0))
        for lat, lon in cells
    ]

    driver_grids: dict[str, list[GridCell]] = {}
    drivers: dict[str, str] = {}
    for driver, (low, high, unit) in DRIVER_RANGES.items():
        grid = [
            GridCell(lat=lat, lon=lon, value=_mock_value(driver, lat, lon, low, high))
            for lat, lon in cells
        ]
        driver_grids[driver] = grid
        avg = sum(c.value for c in grid) / len(grid)
        sign = "+" if avg >= 0 else ""
        drivers[driver] = f"{sign}{avg:.1f}{unit} (placeholder)"

    narration = (
        f"At {gwl} of additional warming above today's climate, {region} shows "
        f"meaningful {sector.lower()} impacts alongside intensifying heat and "
        f"precipitation extremes. (Placeholder narration — real tools, RAG "
        f"grounding, and model layer aren't wired in yet.)"
    )

    return QueryResponse(
        region=region,
        gwl=gwl,
        sector=sector,
        bbox=bbox,
        sector_impact=SECTOR_IMPACT_TEXT[sector],
        narration=narration,
        impact_grid=impact_grid,
        driver_grids=driver_grids,
        drivers=drivers,
        driver_context=DRIVER_CONTEXT_TEXT[sector],
    )
