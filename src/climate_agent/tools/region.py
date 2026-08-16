from climate_agent.schemas import BBox

GEOCODE_LOOKUP = {
    "tolima, colombia": {"min_lat": 3.3, "min_lon": -76.0, "max_lat": 5.4, "max_lon": -74.3},
    "colombia": {"min_lat": -4.2, "min_lon": -79.0, "max_lat": 12.5, "max_lon": -66.9},
    "mexico": {"min_lat": 14.5, "min_lon": -118.4, "max_lat": 32.7, "max_lon": -86.7},
    "united states": {"min_lat": 24.5, "min_lon": -124.8, "max_lat": 49.4, "max_lon": -66.9},
    "china": {"min_lat": 18.2, "min_lon": 73.5, "max_lat": 53.6, "max_lon": 134.8},
    "brazil": {"min_lat": -33.7, "min_lon": -73.9, "max_lat": 5.3, "max_lon": -34.8},
}


def geocode(region_text: str) -> BBox | None:
    """Resolve free-text region name to a bounding box. Stub: fixed lookup table only.

    Args: region_text — place name as typed/extracted (case-insensitive).
    Returns: BBox if found in the lookup table, else None.
    """
    bbox = GEOCODE_LOOKUP.get(region_text.strip().lower())
    return BBox(**bbox) if bbox else None
