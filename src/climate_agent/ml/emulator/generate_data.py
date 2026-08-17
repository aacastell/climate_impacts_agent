import json
from pathlib import Path

import numpy as np
import xarray as xr

from climate_agent.tools.hazard import (
    DRY_DAY_THRESHOLD_MM,
    HEAT_EXTREME_THRESHOLD_C,
    KELVIN_TO_CELSIUS,
    MM_PER_DAY,
    PR_BASELINE_FILE,
    PR_FUTURE_FILE,
    TAS_BASELINE_FILE,
    TAS_FUTURE_FILE,
    _max_consecutive_true,
)

DATA_DIR = Path(__file__).parent / "data"
TRAINING_DATA_FILE = DATA_DIR / "training_points.json"


def _yearly_global_mean_tas(*paths: Path) -> dict[int, float]:
    """Global-mean temperature (°C) for every real calendar year across the given cached tas files.

    Args: paths — one or more cached tas NetCDF files (e.g. baseline and future).
    Returns: {year: global_mean_celsius}.
    """
    points: dict[int, float] = {}
    for path in paths:
        ds = xr.open_dataset(path)
        celsius = ds["tas"] - KELVIN_TO_CELSIUS
        yearly = celsius.groupby("time.year").mean(dim="time")
        global_mean = yearly.mean(dim=["lat", "lon"], skipna=True)
        points.update({int(y): float(v) for y, v in zip(global_mean["year"].values, global_mean.values)})
    return points


def _yearly_global_mean_heat_extreme_days(*paths: Path) -> dict[int, float]:
    """Global-mean count of days above HEAT_EXTREME_THRESHOLD_C for every real calendar year.

    Args: paths — one or more cached tas NetCDF files.
    Returns: {year: global_mean_day_count}.
    """
    points: dict[int, float] = {}
    for path in paths:
        ds = xr.open_dataset(path)
        celsius = ds["tas"] - KELVIN_TO_CELSIUS
        is_hot = celsius > HEAT_EXTREME_THRESHOLD_C
        yearly_counts = is_hot.groupby("time.year").sum(dim="time")
        global_mean = yearly_counts.mean(dim=["lat", "lon"], skipna=True)
        points.update({int(y): float(v) for y, v in zip(global_mean["year"].values, global_mean.values)})
    return points


def _yearly_global_mean_dry_days(*paths: Path) -> dict[int, float]:
    """Global-mean max consecutive-dry-day streak for every real calendar year.

    Args: paths — one or more cached pr NetCDF files.
    Returns: {year: global_mean_streak_length}.
    """
    points: dict[int, float] = {}
    for path in paths:
        ds = xr.open_dataset(path)
        mm_per_day = ds["pr"] * MM_PER_DAY
        is_dry = (mm_per_day < DRY_DAY_THRESHOLD_MM).values
        years = mm_per_day["time"].dt.year.values
        for year in np.unique(years):
            annual_max = _max_consecutive_true(is_dry[years == year])
            points[int(year)] = float(np.nanmean(annual_max))
    return points


def build_training_points() -> dict[str, dict[str, float]]:
    """Real (year -> value) training points for all 3 emulator models, from the cached data.

    Only 14 real years exist (2011-2014 baseline + 2051-2060 future, per the item-8 data
    scoping) — no fabricated points. "gwl" values are global-mean temperature relative to the
    2011-2014 cached baseline, NOT true pre-industrial (that data isn't cached) — an explicit,
    documented scoping choice, not a silent assumption.

    Returns: {"gwl": {year: celsius}, "heat_extreme": {year: day_count}, "precip_extreme": {year: streak_length}}
    (dict keys are strings after JSON round-trip; int(year) when consumed).
    """
    tas_points = _yearly_global_mean_tas(TAS_BASELINE_FILE, TAS_FUTURE_FILE)
    baseline_mean = sum(v for y, v in tas_points.items() if 2011 <= y <= 2014) / 4
    gwl_points = {year: value - baseline_mean for year, value in tas_points.items()}

    return {
        "gwl": gwl_points,
        "heat_extreme": _yearly_global_mean_heat_extreme_days(TAS_BASELINE_FILE, TAS_FUTURE_FILE),
        "precip_extreme": _yearly_global_mean_dry_days(PR_BASELINE_FILE, PR_FUTURE_FILE),
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    points = build_training_points()
    TRAINING_DATA_FILE.write_text(json.dumps(points, indent=2))
    for mode, series in points.items():
        print(f"{mode}: {len(series)} real points -> {sorted(series.items())}")


if __name__ == "__main__":
    main()
