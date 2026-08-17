from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr

from climate_agent.schemas import BBox, GridCell, Window
from climate_agent.tools.analysis import grid_cells_in_bbox, sample_at_cells

CACHE_DIR = Path("data/cache")
BASELINE_FILE = (
    CACHE_DIR / "lpjml_gfdl-esm4_w5e5_historical_2015soc_default_yield-mai-noirr_global_annual-gs_1850_2014.nc"
)
FUTURE_FILE = (
    CACHE_DIR / "lpjml_gfdl-esm4_w5e5_ssp370_2015soc_2015co2_yield-mai-noirr_global_annual-gs_2015_2100.nc"
)
VARIABLE = "yield-mai-noirr"
BASELINE_YEARS = (2011, 2014)  # matches the climate data scoping decision (item 8)
TIME_ORIGIN_YEAR = 1601  # file's time units: "growing seasons since 1601-01-01" (1 unit = 1 year)

SECTOR_LABEL = "Maize yield"


@lru_cache(maxsize=32)
def _mean_over_years(path: Path, start_year: int, end_year: int) -> xr.DataArray:
    """Time-mean of the yield variable over an inclusive calendar-year range.

    Cached by (path, start_year, end_year) — the baseline args never change, and until item 17
    trains a real GWL emulator, the future window doesn't either, so this collapses the global
    read+average to a one-time cost per process instead of once per query.

    Args: path — cached LPJmL NetCDF file. start_year, end_year — inclusive year range.
    Returns: 2D (lat, lon) DataArray of the time-mean.
    """
    ds = xr.open_dataset(path, decode_times=False)
    start_idx = start_year - TIME_ORIGIN_YEAR
    end_idx = end_year - TIME_ORIGIN_YEAR
    return ds[VARIABLE].sel(time=slice(start_idx, end_idx)).mean(dim="time")


def compute_agriculture(bbox: BBox, window: Window) -> tuple[list[GridCell], str]:
    """Change-vs-baseline maize yield impact grid and summary, from real cached LPJmL data.

    Args: bbox — target area. window — target future period.
    Returns: (impact_grid, sector_impact) — per-cell absolute yield change in t/ha
    (baseline->future; not % change — near-zero baseline cells make relative change blow up
    and dominate the average, verified against real data), and a summary string. Cells with no
    data (ocean, non-arable land) are omitted from the grid.
    """
    cells = grid_cells_in_bbox(bbox)
    baseline = sample_at_cells(_mean_over_years(BASELINE_FILE, *BASELINE_YEARS), cells)
    future = sample_at_cells(_mean_over_years(FUTURE_FILE, window.start_year, window.end_year), cells)

    abs_change = future - baseline

    impact_grid = [
        GridCell(lat=lat, lon=lon, value=round(float(value), 2))
        for (lat, lon), value in zip(cells, abs_change)
        if np.isfinite(value)
    ]

    if not impact_grid:
        return [], f"{SECTOR_LABEL}: no data available for this area (non-agricultural land)."

    avg = sum(c.value for c in impact_grid) / len(impact_grid)
    sign = "+" if avg >= 0 else ""
    return impact_grid, f"{SECTOR_LABEL} change: {sign}{avg:.2f} t/ha vs. baseline."
