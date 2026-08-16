import argparse
from pathlib import Path

from climate_agent.data.download.client import download_file, only_file, search_dataset

WINDOWS = {"baseline": "historical", "future": "ssp370"}


def fetch_biome(window: str, cache_dir: Path) -> Path:
    """Download the c4grass plant-functional-type fraction dataset (CLASSIC) for a given window.

    Args: window — "baseline" or "future" (see WINDOWS). cache_dir — destination directory.
    Returns: path to the cached, checksum-verified file.
    """
    dataset = search_dataset(
        product="OutputData",
        sector="biomes",
        model="classic",
        variable="pft",
        pft="c4grass",
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
