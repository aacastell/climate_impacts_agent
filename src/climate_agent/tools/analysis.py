import math
import random

import numpy as np
import xarray as xr

from climate_agent.schemas import BBox

GRID_RESOLUTION_DEG = 0.5  # matches the ISIMIP/GGCMI global simulation grid
MIN_CELLS_PER_SIDE = 3  # ensures a tiny region still renders as a visible patch, not a single point


def _grid_line(min_value: float, max_value: float, resolution: float) -> list[float]:
    """Cell-center coordinates spanning [min_value, max_value] at the given resolution.

    Args: min_value, max_value — span bounds (degrees). resolution — cell spacing (degrees).
    Returns: sorted list of cell-center values, offset by half a resolution step.
    """
    start = math.floor(min_value / resolution) * resolution + resolution / 2
    values = []
    value = start
    while value <= max_value:
        values.append(round(value, 3))
        value += resolution
    return values


def _resolution_for_bbox(bbox: BBox, base_resolution: float, min_cells: int) -> float:
    """Grid resolution for a bbox, shrunk so small regions still render a visible grid.

    Args: bbox — target area. base_resolution — default spacing (degrees). min_cells — minimum cells per side.
    Returns: resolution in degrees — capped at base_resolution, floored near zero.
    """
    lat_span = bbox.max_lat - bbox.min_lat
    lon_span = bbox.max_lon - bbox.min_lon
    resolution = base_resolution
    if lat_span > 0:
        resolution = min(resolution, lat_span / min_cells)
    if lon_span > 0:
        resolution = min(resolution, lon_span / min_cells)
    return max(resolution, 1e-4)  # numerical floor (~11m) so a literal point can't yield zero cells


def grid_cells_in_bbox(bbox: BBox) -> list[tuple[float, float]]:
    """All (lat, lon) cell centers covering a bbox, at module-default resolution.

    Args: bbox — target area.
    Returns: list of (lat, lon) tuples, one per grid cell.
    """
    resolution = _resolution_for_bbox(bbox, GRID_RESOLUTION_DEG, MIN_CELLS_PER_SIDE)
    lats = _grid_line(bbox.min_lat, bbox.max_lat, resolution)
    lons = _grid_line(bbox.min_lon, bbox.max_lon, resolution)
    return [(lat, lon) for lat in lats for lon in lons]


def sample_at_cells(data: xr.DataArray, cells: list[tuple[float, float]]) -> np.ndarray:
    """Nearest-neighbor sample a 2D (lat, lon) DataArray at each grid cell center.

    Args: data — 2D DataArray indexed by lat, lon. cells — (lat, lon) tuples, e.g. from
    grid_cells_in_bbox.
    Returns: array of sampled values, one per cell, same order as `cells`.
    """
    lats = [lat for lat, lon in cells]
    lons = [lon for lat, lon in cells]
    sampled = data.sel(
        lat=xr.DataArray(lats, dims="points"),
        lon=xr.DataArray(lons, dims="points"),
        method="nearest",
    )
    return sampled.values


def stub_value(seed: str, lat: float, lon: float, low: float, high: float) -> float:
    """Deterministic placeholder value for a grid cell. Delete once real tools (items 9-12) land.

    Args: seed — label for this value stream (e.g. driver name). lat, lon — cell center. low, high — value range.
    Returns: a value in [low, high], stable for the same (seed, lat, lon).
    """
    rng = random.Random(f"{seed}:{lat:.3f}:{lon:.3f}")
    return round(rng.uniform(low, high), 2)
