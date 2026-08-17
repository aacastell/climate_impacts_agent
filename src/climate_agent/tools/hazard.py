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
TAS_BASELINE_FILE = CACHE_DIR / "gfdl-esm4_r1i1p1f1_w5e5_historical_tas_global_daily_2011_2014.nc"
TAS_FUTURE_FILE = CACHE_DIR / "gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_global_daily_2051_2060.nc"
PR_BASELINE_FILE = CACHE_DIR / "gfdl-esm4_r1i1p1f1_w5e5_historical_pr_global_daily_2011_2014.nc"
PR_FUTURE_FILE = CACHE_DIR / "gfdl-esm4_r1i1p1f1_w5e5_ssp370_pr_global_daily_2051_2060.nc"

BASELINE_YEARS = (2011, 2014)  # matches the climate data scoping decision (item 8)
FUTURE_CACHED_YEARS = (2051, 2060)  # the only future decade actually cached (item 8)
KELVIN_TO_CELSIUS = 273.15
MM_PER_DAY = 86400  # kg m-2 s-1 -> mm/day (1 kg/m2 of water = 1mm depth)
HEAT_EXTREME_THRESHOLD_C = 35.0
DRY_DAY_THRESHOLD_MM = 1.0  # ETCCDI "dry day" convention


def _max_consecutive_true(bool_arr: np.ndarray) -> np.ndarray:
    """Max run length of consecutive True values along axis 0, vectorized over the rest.

    Args: bool_arr — boolean array, shape (time, ...).
    Returns: array of shape bool_arr.shape[1:], the max run length per remaining-axis element.
    """
    counts = np.zeros(bool_arr.shape, dtype=np.int32)
    counts[0] = bool_arr[0]
    for t in range(1, bool_arr.shape[0]):
        counts[t] = (counts[t - 1] + bool_arr[t]) * bool_arr[t]
    return counts.max(axis=0)


def _annual_consecutive_dry_days(is_dry: xr.DataArray) -> xr.DataArray:
    """Average annual max consecutive-dry-day streak, per grid cell.

    Args: is_dry — daily boolean (time, lat, lon) DataArray, spanning one or more full years.
    Returns: 2D (lat, lon) DataArray of the average annual max streak length.
    """
    years = is_dry["time"].dt.year.values
    unique_years = np.unique(years)
    values = is_dry.values
    annual_max = np.stack([_max_consecutive_true(values[years == year]) for year in unique_years])
    return xr.DataArray(
        annual_max.mean(axis=0),
        dims=("lat", "lon"),
        coords={"lat": is_dry["lat"], "lon": is_dry["lon"]},
    )


@lru_cache(maxsize=32)
def _driver_stats(path: Path, variable: str, start_year: int, end_year: int) -> tuple[xr.DataArray, xr.DataArray]:
    """This variable's global mean and extreme-day grid, over an inclusive calendar-year range.

    Cached by (path, variable, start_year, end_year) — the file's internal chunking is one full
    global grid per day (measured: a small padded-bbox slice costs ~4.5s, the whole globe costs
    ~8s — the chunking makes region-first slicing pointless, since decompression cost is
    dominated by number of days, not spatial extent requested). So this loads the full daily
    global grid once, derives every reduction needed from that one in-memory array, then lets it
    be discarded — only the small (lat, lon) reduced grids are cached, not the ~2GB+ raw array,
    to keep peak memory bounded.

    Args: path — cached NetCDF file. variable — "tas" or "pr". start_year, end_year — inclusive
    year range.
    Returns: (mean_grid, extreme_grid) — extreme_grid is average annual heat-extreme-day count
    for tas, average annual consecutive-dry-day streak for pr.
    """
    ds = xr.open_dataset(path)
    raw = ds[variable].sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31")).load()
    n_years = len(np.unique(raw["time"].dt.year.values))

    if variable == "tas":
        celsius = raw - KELVIN_TO_CELSIUS
        mean_grid = celsius.mean(dim="time")
        extreme_grid = (celsius > HEAT_EXTREME_THRESHOLD_C).sum(dim="time") / n_years
    else:
        mm_per_day = raw * MM_PER_DAY
        mean_grid = mm_per_day.mean(dim="time")
        extreme_grid = _annual_consecutive_dry_days(mm_per_day < DRY_DAY_THRESHOLD_MM)

    return mean_grid, extreme_grid


def _driver_from_grids(
    cells: list[tuple[float, float]], baseline_grid: xr.DataArray, future_grid: xr.DataArray, unit: str
) -> tuple[list[GridCell], str]:
    """Sample baseline/future grids at each cell and build the change grid + summary for one driver.

    Args: cells — (lat, lon) points to sample. baseline_grid, future_grid — 2D (lat, lon)
    DataArrays. unit — display unit suffix for the summary string.
    Returns: (grid, summary) — per-cell absolute change, and a summary string.
    """
    baseline_vals = sample_at_cells(baseline_grid, cells)
    future_vals = sample_at_cells(future_grid, cells)
    change = future_vals - baseline_vals

    grid = [
        GridCell(lat=lat, lon=lon, value=round(float(v), 2))
        for (lat, lon), v in zip(cells, change)
        if np.isfinite(v)
    ]
    if not grid:
        return [], "no data available for this area."

    avg = sum(c.value for c in grid) / len(grid)
    sign = "+" if avg >= 0 else ""
    return grid, f"{sign}{avg:.2f}{unit} vs. baseline."


def compute_hazard_drivers(
    bbox: BBox, window: Window
) -> tuple[dict[str, list[GridCell]], dict[str, str], list[str]]:
    """Change-vs-baseline grids and summary text for all four climate hazard drivers, from real
    cached tas/pr data.

    "Precipitation extreme days" is implemented as consecutive dry days (ETCCDI CDD convention,
    <1mm/day) — a real, standard precipitation-extreme index.

    If the requested window falls outside the one decade actually cached (FUTURE_CACHED_YEARS),
    clamps to the nearest cached range and reports it as an assumption — this was unreachable
    with item 10's stub emulator (always returned an in-range window) but is a real, hit path
    now that item 17's trained models can predict windows outside the cache.

    Args: bbox — target area. window — target future period.
    Returns: (driver_grids, drivers, assumptions) — per-driver GridCell lists, per-driver summary
    strings, and any assumptions from window clamping.
    """
    cells = grid_cells_in_bbox(bbox)

    used_start, used_end, clamped = clamp_years_to_range(window.start_year, window.end_year, *FUTURE_CACHED_YEARS)
    assumptions = []
    if clamped:
        assumptions.append(
            f"Requested period {window.start_year}-{window.end_year} isn't cached — "
            f"showing hazard data for {used_start}-{used_end} instead."
        )

    tas_base_mean, tas_base_heat = _driver_stats(TAS_BASELINE_FILE, "tas", *BASELINE_YEARS)
    tas_fut_mean, tas_fut_heat = _driver_stats(TAS_FUTURE_FILE, "tas", used_start, used_end)
    pr_base_mean, pr_base_dry = _driver_stats(PR_BASELINE_FILE, "pr", *BASELINE_YEARS)
    pr_fut_mean, pr_fut_dry = _driver_stats(PR_FUTURE_FILE, "pr", used_start, used_end)

    driver_grids: dict[str, list[GridCell]] = {}
    drivers: dict[str, str] = {}

    driver_grids["Avg. temperature change"], drivers["Avg. temperature change"] = _driver_from_grids(
        cells, tas_base_mean, tas_fut_mean, "°C"
    )
    driver_grids["Avg. precipitation change"], drivers["Avg. precipitation change"] = _driver_from_grids(
        cells, pr_base_mean, pr_fut_mean, " mm/day"
    )
    driver_grids["Heat extreme days"], drivers["Heat extreme days"] = _driver_from_grids(
        cells, tas_base_heat, tas_fut_heat, " days/year above 35°C"
    )
    driver_grids["Precipitation extreme days"], drivers["Precipitation extreme days"] = _driver_from_grids(
        cells, pr_base_dry, pr_fut_dry, " consecutive dry days"
    )

    return driver_grids, drivers, assumptions
