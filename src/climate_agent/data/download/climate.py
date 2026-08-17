import argparse
from pathlib import Path

from climate_agent.data.download.client import (
    download_file,
    file_for_year_range,
    search_dataset,
)

WINDOWS = {
    "baseline": {"climate_scenario": "historical", "year_range": "2011_2014"},
    "future": {"climate_scenario": "ssp370", "year_range": "2051_2060"},
}


def fetch_climate_variable(variable: str, window: str, cache_dir: Path) -> Path:
    """Download one climate variable's decadal file for a given window.

    Args: variable — "tas" or "pr". window — "baseline" or "future" (see WINDOWS).
        cache_dir — destination directory for the cached file.
    Returns: path to the cached, checksum-verified file.
    """
    spec = WINDOWS[window]
    dataset = search_dataset(
        product="InputData",
        climate_scenario=spec["climate_scenario"],
        climate_variable=variable,
    )
    file_entry = file_for_year_range(dataset, spec["year_range"])
    return download_file(file_entry, cache_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable", required=True, choices=["tas", "pr"])
    parser.add_argument("--window", required=True, choices=list(WINDOWS))
    parser.add_argument("--cache-dir", required=True, type=Path)
    args = parser.parse_args()
    print(f"Cached: {fetch_climate_variable(args.variable, args.window, args.cache_dir)}")


if __name__ == "__main__":
    main()
