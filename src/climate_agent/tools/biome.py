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
    CACHE_DIR / "classic_gfdl-esm4_w5e5_historical_2015soc_default_npp-total_global_monthly_1850_2014.nc"
)
FUTURE_FILE = (
    CACHE_DIR / "classic_gfdl-esm4_w5e5_ssp370_2015soc_default_npp-total_global_monthly_2015_2100.nc"
)
VARIABLE = "npp-total"
BASELINE_YEARS = (2011, 2014)  # matches the climate data scoping decision (item 8)
FUTURE_CACHED_YEARS = (2015, 2100)  # the full future scenario period actually cached
TIME_ORIGIN_YEAR = 1601  # file's time units: "days since 1601-01-01", 365_day calendar
DAYS_PER_YEAR = 365
KG_TO_G = 1000
SECONDS_PER_YEAR = DAYS_PER_YEAR * 86400  # matches the file's own 365_day calendar

SECTOR_LABEL = "Net primary productivity"


@lru_cache(maxsize=32)
def _mean_over_years(path: Path, start_year: int, end_year: int) -> xr.DataArray:
    """Time-mean of NPP (monthly resolution) over an inclusive calendar-year range.

    Cached by (path, start_year, end_year) — the baseline args never change, and until item 17
    trains a real GWL emulator, the future window doesn't either, so this collapses the global
    read+average to a one-time cost per process instead of once per query.

    Args: path — cached CLASSIC NetCDF file. start_year, end_year — inclusive year range.
    Returns: 2D (lat, lon) DataArray of the time-mean, in the file's native kg m-2 s-1.
    """
    ds = xr.open_dataset(path, decode_times=False)
    start_idx = (start_year - TIME_ORIGIN_YEAR) * DAYS_PER_YEAR
    end_idx = (end_year + 1 - TIME_ORIGIN_YEAR) * DAYS_PER_YEAR - 1
    return ds[VARIABLE].sel(time=slice(start_idx, end_idx)).mean(dim="time")


def compute_biome(bbox: BBox, window: Window) -> tuple[list[GridCell], str, list[str]]:
    """Change-vs-baseline net primary productivity impact grid and summary, from real cached
    CLASSIC data.

    Was originally scoped to a pft/c4grass land-cover fraction, but that turned out to be
    static under the 2015soc scenario (confirmed against real data — ~0 variance globally
    across the full 1850-2100 span). NPP is a genuinely climate-responsive CLASSIC output
    instead (verified: real seasonal cycle, real baseline-vs-future difference).

    If the requested window falls outside the cached future period, clamps to the nearest
    cached range and reports it as an assumption (see hazard.py's compute_hazard_drivers for
    why this became a real, reachable path with item 17's trained emulator models).

    Args: bbox — target area. window — target future period.
    Returns: (impact_grid, sector_impact, assumptions) — per-cell absolute NPP change in
    g C/m²/yr (baseline->future, converted from the file's native kg m-2 s-1 for readability),
    a summary string, and any assumptions from window clamping. Cells with no data are omitted.
    """
    cells = grid_cells_in_bbox(bbox)

    used_start, used_end, clamped = clamp_years_to_range(window.start_year, window.end_year, *FUTURE_CACHED_YEARS)
    assumptions = []
    if clamped:
        assumptions.append(
            f"Requested period {window.start_year}-{window.end_year} isn't cached — "
            f"showing biome data for {used_start}-{used_end} instead."
        )

    baseline = sample_at_cells(_mean_over_years(BASELINE_FILE, *BASELINE_YEARS), cells)
    future = sample_at_cells(_mean_over_years(FUTURE_FILE, used_start, used_end), cells)

    abs_change = (future - baseline) * KG_TO_G * SECONDS_PER_YEAR

    impact_grid = [
        GridCell(lat=lat, lon=lon, value=round(float(value), 1))
        for (lat, lon), value in zip(cells, abs_change)
        if np.isfinite(value)
    ]

    if not impact_grid:
        return [], f"{SECTOR_LABEL}: no data available for this area.", assumptions

    avg = sum(c.value for c in impact_grid) / len(impact_grid)
    sign = "+" if avg >= 0 else ""
    return impact_grid, f"{SECTOR_LABEL} change: {sign}{avg:.1f} g C/m²/yr vs. baseline.", assumptions
