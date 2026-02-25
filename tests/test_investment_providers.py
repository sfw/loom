"""Tests for investment provider adapters and finance helpers."""

from __future__ import annotations

import httpx

from loom.research.providers.markets import fetch_stooq_daily_prices
from loom.research.providers.sec_finance import extract_ttm_value, fetch_sec_ticker_map


async def test_fetch_stooq_daily_prices_parses_csv():
    csv_body = (
        "Date,Open,High,Low,Close,Volume\n"
        "2025-01-01,100,101,99,100.5,1000\n"
        "2025-01-02,100.5,102,100,101.5,1100\n"
    )

    async def _handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/q/d/l/"
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/csv"},
            content=csv_body.encode("utf-8"),
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        payload = await fetch_stooq_daily_prices(symbol="AAPL", client=client)
    assert payload["provider"] == "stooq"
    assert payload["symbol"] == "AAPL"
    assert len(payload["rows"]) == 2
    assert payload["rows"][-1]["close"] == 101.5


async def test_fetch_sec_ticker_map_parses_payload():
    body = b'{"0":{"cik_str":320193,"ticker":"AAPL","title":"Apple Inc."}}'

    async def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            content=body,
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        mapping = await fetch_sec_ticker_map(client=client)
    assert mapping["AAPL"]["cik"] == "0000320193"


def test_extract_ttm_value_uses_latest_four_quarters():
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"val": 10, "end": "2024-03-31", "form": "10-Q"},
                            {"val": 20, "end": "2024-06-30", "form": "10-Q"},
                            {"val": 30, "end": "2024-09-30", "form": "10-Q"},
                            {"val": 40, "end": "2024-12-31", "form": "10-Q"},
                            {"val": 999, "end": "2023-12-31", "form": "10-K"},
                        ]
                    }
                }
            }
        }
    }
    ttm = extract_ttm_value(facts, tag="Revenues")
    assert ttm is not None
    assert ttm["value"] == 100
