import os

import httpx
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")


class QueryError(Exception):
    pass


def fetch_query(query: str) -> dict:
    try:
        response = httpx.post(
            f"{API_BASE_URL}/query",
            json={"query": query},
            timeout=10.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise QueryError(str(e)) from e
    return response.json()
