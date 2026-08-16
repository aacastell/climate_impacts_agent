import argparse
from pathlib import Path

from climate_agent.data.download.client import download_file, only_file, search_dataset

WINDOWS = {"baseline": "historical", "future": "ssp370"}
SENS_SCENARIO = {"baseline": "default", "future": "2015co2"}


def fetch_agriculture(window: str, cache_dir: Path) -> Path:
    """Download the maize yield dataset (LPJmL, rainfed) for a given window.

    Args: window — "baseline" or "future" (see WINDOWS). cache_dir — destination directory.
    Returns: path to the cached, checksum-verified file.
    """
    dataset = search_dataset(
        product="OutputData",
        sector="agriculture",
        model="lpjml",
        crop="mai",
        variable="yield",
        irrigation="noirr",
        soc_scenario="2015soc",
        sens_scenario=SENS_SCENARIO[window],
        climate_scenario=WINDOWS[window],
    )
    return download_file(only_file(dataset), cache_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", required=True, choices=list(WINDOWS))
    parser.add_argument("--cache-dir", required=True, type=Path)
    args = parser.parse_args()
    print(f"Cached: {fetch_agriculture(args.window, args.cache_dir)}")


if __name__ == "__main__":
    main()
