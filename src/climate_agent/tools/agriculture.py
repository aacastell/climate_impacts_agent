from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr

from climate_agent.schemas import BBox, GridCell, Window
from climate_agent.tools.analysis import (
    clamp_years_to_range,
    grid_cells_in_bbox,
    sample_at_cells,
)

CACHE_DIR = Path("data/cache")
BASELINE_FILE = (
    CACHE_DIR / "lpjml_gfdl-esm4_w5e5_historical_2015soc_default_yield-mai-noirr_global_annual-gs_1850_2014.nc"
)
FUTURE_FILE = (
    CACHE_DIR / "lpjml_gfdl-esm4_w5e5_ssp370_2015soc_2015co2_yield-mai-noirr_global_annual-gs_2015_2100.nc"
)
VARIABLE = "yield-mai-noirr"
BASELINE_YEARS = (2011, 2014)  # matches the climate data scoping decision (item 8)
FUTURE_CACHED_YEARS = (2015, 2100)  # the full future scenario period actually cached
TIME_ORIGIN_YEAR = 1601  # file's time units: "growing seasons since 1601-01-01" (1 unit = 1 year)
BBOX_PAD_DEG = 2.0  # generous margin so nearest-neighbor sampling near the bbox edge still finds real data

SECTOR_LABEL = "Maize yield"


@lru_cache(maxsize=64)
def _mean_over_years(
    path: Path,
    start_year: int,
    end_year: int,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
) -> xr.DataArray:
    """Time-mean of the yield variable over an inclusive calendar-year range, subset to a padded bbox.

    Cached by (path, start_year, end_year, bbox bounds) — loads only the padded bbox slice of
    the global grid, not the full array, matching the memory fix applied to hazard.py's
    _driver_stats (2026-08-17): eager full-globe loads across multiple tools in one request were
    the real cause of a Docker OOM kill. lat is descending (90 -> -90) and lon is ascending
    (-180 -> 180) in the cached files — verified directly. An antimeridian-wrapping bbox
    (min_lon > max_lon) falls back to the full lon range rather than an inverted, empty slice.

    Args: path — cached LPJmL NetCDF file. start_year, end_year — inclusive year range.
    min_lat, min_lon, max_lat, max_lon — bbox to subset to.
    Returns: 2D (lat, lon) DataArray of the time-mean.
    """
    lon_slice = slice(None) if min_lon > max_lon else slice(min_lon - BBOX_PAD_DEG, max_lon + BBOX_PAD_DEG)

    ds = xr.open_dataset(path, decode_times=False)
    start_idx = start_year - TIME_ORIGIN_YEAR
    end_idx = end_year - TIME_ORIGIN_YEAR
    return (
        ds[VARIABLE]
        .sel(
            time=slice(start_idx, end_idx),
            lat=slice(max_lat + BBOX_PAD_DEG, min_lat - BBOX_PAD_DEG),
            lon=lon_slice,
        )
        .mean(dim="time")
    )


def compute_agriculture(bbox: BBox, window: Window) -> tuple[list[GridCell], str, list[str]]:
    """Change-vs-baseline maize yield impact grid and summary, from real cached LPJmL data.

    If the requested window falls outside the cached future period, clamps to the nearest
    cached range and reports it as an assumption (see hazard.py's compute_hazard_drivers for
    why this became a real, reachable path with item 17's trained emulator models).

    Args: bbox — target area. window — target future period.
    Returns: (impact_grid, sector_impact, assumptions) — per-cell absolute yield change in t/ha
    (baseline->future; not % change — near-zero baseline cells make relative change blow up
    and dominate the average, verified against real data), a summary string, and any assumptions
    from window clamping. Cells with no data (ocean, non-arable land) are omitted from the grid.
    """
    cells = grid_cells_in_bbox(bbox)

    used_start, used_end, clamped = clamp_years_to_range(window.start_year, window.end_year, *FUTURE_CACHED_YEARS)
    assumptions = []
    if clamped:
        assumptions.append(
            f"Requested period {window.start_year}-{window.end_year} isn't cached — "
            f"showing agriculture data for {used_start}-{used_end} instead."
        )

    baseline = sample_at_cells(
        _mean_over_years(BASELINE_FILE, *BASELINE_YEARS, bbox.min_lat, bbox.min_lon, bbox.max_lat, bbox.max_lon),
        cells,
    )
    future = sample_at_cells(
        _mean_over_years(FUTURE_FILE, used_start, used_end, bbox.min_lat, bbox.min_lon, bbox.max_lat, bbox.max_lon),
        cells,
    )

    abs_change = future - baseline

    impact_grid = [
        GridCell(lat=lat, lon=lon, value=round(float(value), 2))
        for (lat, lon), value in zip(cells, abs_change)
        if np.isfinite(value)
    ]

    if not impact_grid:
        return [], f"{SECTOR_LABEL}: no data available for this area (non-agricultural land).", assumptions

    avg = sum(c.value for c in impact_grid) / len(impact_grid)
    sign = "+" if avg >= 0 else ""
    return impact_grid, f"{SECTOR_LABEL} change: {sign}{avg:.2f} t/ha vs. baseline.", assumptions
