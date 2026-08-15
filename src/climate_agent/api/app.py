import os

from dotenv import load_dotenv
from fastapi import FastAPI

from climate_agent.schemas import QueryRequest, QueryResponse

load_dotenv()

if os.environ.get("ORCHESTRATION_BACKEND") == "graph":
    from climate_agent.orchestration.graph import query as run_query
else:
    from climate_agent.orchestration.mock import query as run_query

app = FastAPI(title="Climate Impacts Agent API")


@app.get("/")
def root():
    return {"service": "climate-impacts-agent-api"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    return run_query(request.query)
