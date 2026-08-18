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
BBOX_PAD_DEG = 2.0  # generous margin so nearest-neighbor sampling near the bbox edge still finds real data



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


@lru_cache(maxsize=64)
def _driver_stats(
    path: Path,
    variable: str,
    start_year: int,
    end_year: int,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """This variable's regional mean and extreme-day grid, over an inclusive calendar-year range,
    subset to a padded bbox.

    Cached by (path, variable, start_year, end_year, bbox bounds) — loads only the padded bbox
    slice of the daily grid, not the full ~1.5-3.8GB global array, since sample_at_cells only
    ever samples points inside the bbox anyway. Real fix (2026-08-17): the previous full-globe
    load was benchmarked on I/O time only (bbox-first was actually a bit faster: 4.5s vs 8s) and
    never accounted for memory — four sequential full-global loads per request (tas/pr x
    baseline/future), each with same-size intermediates (mm_per_day, the consecutive-dry-day
    counts array), transiently needed 8-12GB+ and caused a real OOM kill in Docker.

    lat is descending (90 -> -90) and lon is ascending (-180 -> 180) in the cached files —
    verified directly, not assumed. An antimeridian-wrapping bbox (min_lon > max_lon) falls back
    to the full lon range rather than producing an inverted, empty slice.

    Args: path — cached NetCDF file. variable — "tas" or "pr". start_year, end_year — inclusive
    year range. min_lat, min_lon, max_lat, max_lon — bbox to subset to.
    Returns: (mean_grid, extreme_grid) — extreme_grid is average annual heat-extreme-day count
    for tas, average annual consecutive-dry-day streak for pr.
    """
    lon_slice = slice(None) if min_lon > max_lon else slice(min_lon - BBOX_PAD_DEG, max_lon + BBOX_PAD_DEG)

    ds = xr.open_dataset(path)
    raw = ds[variable].sel(
        time=slice(f"{start_year}-01-01", f"{end_year}-12-31"),
        lat=slice(max_lat + BBOX_PAD_DEG, min_lat - BBOX_PAD_DEG),
        lon=lon_slice,
    ).load()
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

    tas_base_mean, tas_base_heat = _driver_stats(
        TAS_BASELINE_FILE, "tas", *BASELINE_YEARS, bbox.min_lat, bbox.min_lon, bbox.max_lat, bbox.max_lon
    )
    tas_fut_mean, tas_fut_heat = _driver_stats(
        TAS_FUTURE_FILE, "tas", used_start, used_end, bbox.min_lat, bbox.min_lon, bbox.max_lat, bbox.max_lon
    )
    pr_base_mean, pr_base_dry = _driver_stats(
        PR_BASELINE_FILE, "pr", *BASELINE_YEARS, bbox.min_lat, bbox.min_lon, bbox.max_lat, bbox.max_lon
    )
    pr_fut_mean, pr_fut_dry = _driver_stats(
        PR_FUTURE_FILE, "pr", used_start, used_end, bbox.min_lat, bbox.min_lon, bbox.max_lat, bbox.max_lon
    )


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
