import hashlib
import tempfile
from pathlib import Path

import httpx

ISIMIP_API_BASE = "https://data.isimip.org/api/v1"
DEFAULT_SIMULATION_ROUND = "ISIMIP3b"
DEFAULT_CLIMATE_FORCING = "gfdl-esm4"


def search_dataset(**specifiers) -> dict:
    """Search ISIMIP for a single dataset matching the given specifiers, fixed to our project's model.

    Args: specifiers — ISIMIP API query params (e.g. climate_scenario="ssp370", sector="agriculture").
    Returns: the first matching dataset's full detail, including its files list.
    """
    params = {
        "simulation_round": DEFAULT_SIMULATION_ROUND,
        "climate_forcing": DEFAULT_CLIMATE_FORCING,
        "page_size": 1,
        **specifiers,
    }
    search = httpx.get(f"{ISIMIP_API_BASE}/datasets/", params=params, timeout=30.0)
    search.raise_for_status()
    results = search.json()["results"]
    if not results:
        raise ValueError(f"No ISIMIP dataset found for {specifiers}")

    detail = httpx.get(f"{ISIMIP_API_BASE}/datasets/{results[0]['id']}/", timeout=30.0)
    detail.raise_for_status()
    return detail.json()


def file_for_year_range(dataset: dict, year_range: str) -> dict:
    """Pick the file within a dataset matching a decadal year-range suffix (e.g. "2051_2060")."""
    for file_entry in dataset["files"]:
        if year_range in file_entry["name"]:
            return file_entry
    raise ValueError(f"No file matching {year_range} in dataset {dataset['id']}")


def only_file(dataset: dict) -> dict:
    """A dataset's single file, for sectors whose output isn't decade-chunked (agriculture, biome)."""
    files = dataset["files"]
    if len(files) != 1:
        raise ValueError(f"Expected exactly one file in dataset {dataset['id']}, found {len(files)}")
    return files[0]


def download_file(file_entry: dict, cache_dir: Path) -> Path:
    """Download a file entry into cache_dir, atomically, verifying its checksum. Skips if already cached and valid.

    Args: file_entry — dict with name, file_url, checksum, checksum_type. cache_dir — destination directory.
    Returns: path to the cached, verified file.
    """
    final_path = cache_dir / file_entry["name"]
    if final_path.exists() and _checksum_matches(final_path, file_entry):
        return final_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with httpx.stream("GET", file_entry["file_url"], timeout=None, follow_redirects=True) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                tmp.write(chunk)

    if not _checksum_matches(tmp_path, file_entry):
        tmp_path.unlink()
        raise ValueError(f"Checksum mismatch downloading {file_entry['name']}")

    tmp_path.rename(final_path)
    return final_path


def _checksum_matches(path: Path, file_entry: dict) -> bool:
    """Verify a file's checksum against ISIMIP's recorded value (algorithm given per-file, e.g. sha512)."""
    hasher = hashlib.new(file_entry["checksum_type"])
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == file_entry["checksum"]
