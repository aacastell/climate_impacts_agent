import operator
from typing import Annotated, Literal, TypedDict

from climate_agent.schemas import GridCell


class ToolCall(TypedDict):
    tool: str
    args: dict
    result: object

class AgentState(TypedDict):
    query: str

    tool_calls: Annotated[list[ToolCall], operator.add]
    attempts: int
    status: Literal[
        "resolving", 
        "ready_to_narrate", 
        "verifying", 
        "done", 
        "degraded"]

    region: str | None
    bbox: str | None
    gwl: str | None
    sector: str | None

    impact_grid: list[GridCell] | None
    driver_grids: dict[str, list[GridCell]] | None
    drivers: dict[str, str] | None
    driver_context: dict[str, str] | None
    sector_impact: str | None

    narration: str | None
    narration_attempts: int
    grounded: bool
    assumptions: Annotated[list[str], operator.add]

    

