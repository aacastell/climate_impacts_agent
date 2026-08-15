from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


class BBox(BaseModel):
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float


class GridCell(BaseModel):
    lat: float
    lon: float
    value: float


class QueryResponse(BaseModel):
    region: str
    gwl: str
    sector: str
    bbox: BBox
    sector_impact: str
    narration: str
    impact_grid: list[GridCell]
    driver_grids: dict[str, list[GridCell]]
    drivers: dict[str, str]
    driver_context: dict[str, str]
