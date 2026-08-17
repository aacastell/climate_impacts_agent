import argparse
from pathlib import Path

from climate_agent.data.download.client import download_file, only_file, search_dataset

WINDOWS = {"baseline": "historical", "future": "ssp370"}


def fetch_biome(window: str, cache_dir: Path) -> Path:
    """Download the total NPP (net primary productivity) dataset (CLASSIC) for a given window.

    Was originally scoped to the pft/c4grass land-cover fraction, but that variable turned out
    to be static under the 2015soc scenario (confirmed against real data: ~0 variance globally
    across the full 1850-2100 span) — no climate signal at all. NPP is a genuinely
    climate-responsive CLASSIC output instead.

    Args: window — "baseline" or "future" (see WINDOWS). cache_dir — destination directory.
    Returns: path to the cached, checksum-verified file.
    """
    dataset = search_dataset(
        product="OutputData",
        sector="biomes",
        model="classic",
        variable="npp",
        pft="total",
        soc_scenario="2015soc",
        climate_scenario=WINDOWS[window],
    )
    return download_file(only_file(dataset), cache_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", required=True, choices=list(WINDOWS))
    parser.add_argument("--cache-dir", required=True, type=Path)
    args = parser.parse_args()
    print(f"Cached: {fetch_biome(args.window, args.cache_dir)}")


if __name__ == "__main__":
    main()
