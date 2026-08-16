import httpx 

from climate_agent.schemas import BBox

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "climate-impacts-agent-demo/0.1 (portfolio project)"

def geocode(region_text: str) -> BBox | None:
    """Resolve free-text region name to a bounding box via OpenStreetMap Nominatim
    
    Args: region_text - place name as typed/extracted.
    Returns: BBox if a match is found, else None.
    """
    response = httpx.get(
        NOMINATIM_URL,
        params={"q": region_text, "format": "json", "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=10.0,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        return None

    min_lat, max_lat, min_lon, max_lon = (float(v) for v in results[0]["boundingbox"])
    return BBox(min_lat=min_lat,
                min_lon=min_lon,
                max_lat=max_lat,
                max_lon=max_lon)
