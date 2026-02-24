"""Regression tests for economic provider payload parsing."""

from __future__ import annotations

import httpx
import pytest

from loom.research.providers.economic import EconomicProviderError, _request_json


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
