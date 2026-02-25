"""Regression tests for economic provider payload parsing."""

from __future__ import annotations

import httpx
import pytest

from loom.research.providers.economic import (
    EconomicProviderError,
    _request_json,
    economic_get_observations,
    economic_search,
)


def _client_for_body(
    body: str | bytes,
    *,
    status_code: int = 200,
    content_type: str = "application/json",
) -> httpx.AsyncClient:
    payload = body.encode("utf-8") if isinstance(body, str) else body

    async def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status_code,
            headers={"content-type": content_type},
            content=payload,
            request=request,
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(_handler))


@pytest.mark.parametrize("body", ["null", "'null'", "None", '"null"', "(null)"])
async def test_request_json_coerces_nullish_payloads_to_empty_mapping(body: str):
    async with _client_for_body(body, content_type="text/plain") as client:
        payload = await _request_json(client, "https://example.test/data")
    assert payload == {}


async def test_request_json_handles_xssi_prefixed_json():
    raw = ")]}',\n" + '{"data":{"flows":3}}'
    async with _client_for_body(raw, content_type="application/json") as client:
        payload = await _request_json(client, "https://example.test/data")
    assert payload == {"data": {"flows": 3}}


async def test_request_json_raises_for_non_json_payload():
    async with _client_for_body("<html>nope</html>", content_type="text/html") as client:
        with pytest.raises(EconomicProviderError, match="Invalid JSON payload"):
            await _request_json(client, "https://example.test/data")


async def test_fred_search_supports_direct_series_id_lookup():
    async with _client_for_body("{}") as client:
        hits = await economic_search(
            provider="fred",
            query="unrate",
            max_results=5,
            client=client,
        )
    assert len(hits) == 1
    assert hits[0]["provider"] == "fred"
    assert hits[0]["series_id"] == "UNRATE"


async def test_fred_search_parses_series_links_from_search_page():
    html_body = """
<html>
  <body>
    <a href="/series/UNRATE">Unemployment Rate</a>
    <a href="/series/CPIAUCSL">Consumer Price Index for All Urban Consumers</a>
  </body>
</html>
"""

    async def _handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/searchresults"
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/html"},
            content=html_body.encode("utf-8"),
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        hits = await economic_search(
            provider="fred",
            query="unemployment rate",
            max_results=5,
            client=client,
        )
    assert len(hits) == 2
    assert hits[0]["provider"] == "fred"
    assert hits[0]["series_id"] == "UNRATE"
    assert hits[1]["series_id"] == "CPIAUCSL"


async def test_fred_observations_parse_fredgraph_csv_payload():
    csv_body = (
        "DATE,UNRATE\n"
        "2023-01-01,3.4\n"
        "2023-02-01,3.6\n"
        "2023-03-01,.\n"
    )
    async with _client_for_body(csv_body, content_type="text/csv") as client:
        payload = await economic_get_observations(
            provider="fred",
            series_id="UNRATE",
            max_observations=10,
            client=client,
        )
    assert payload["provider"] == "fred"
    assert payload["series_id"] == "UNRATE"
    assert payload["title"] == "UNRATE"
    assert payload["coverage"]["count"] == 2
    assert payload["observations"][0]["period"] == "2023-01-01"
    assert payload["observations"][0]["value"] == 3.4
