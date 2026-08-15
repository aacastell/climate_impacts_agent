import httpx
import pytest

from climate_agent.frontend.api_client import QueryError, fetch_query


def _fake_request() -> httpx.Request:
    return httpx.Request("POST", "http://testserver/query")


def test_fetch_query_raises_query_error_on_http_status_error(monkeypatch):
    response = httpx.Response(500, request=_fake_request())
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: response)

    with pytest.raises(QueryError):
        fetch_query("How would 2°C of warming affect agriculture in Colombia?")


def test_fetch_query_raises_query_error_on_connection_failure(monkeypatch):
    def fake_post(*args, **kwargs):
        raise httpx.ConnectError("connection refused", request=_fake_request())

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(QueryError):
        fetch_query("How would 2°C of warming affect agriculture in Colombia?")


def test_fetch_query_accepts_and_forwards_a_string_query(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["json"] = json
        return httpx.Response(200, json={"ok": True}, request=_fake_request())

    monkeypatch.setattr(httpx, "post", fake_post)

    query_text = "How would 2°C of warming affect agriculture in Tolima, Colombia?"
    fetch_query(query_text)

    assert isinstance(captured["json"]["query"], str)
    assert captured["json"]["query"] == query_text
