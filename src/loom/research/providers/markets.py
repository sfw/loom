"""Keyless market-data provider adapters."""

from __future__ import annotations

import csv
import io
import os
from datetime import date
from typing import Any

import httpx

from loom.research.finance import safe_float

DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_USER_AGENT = "Loom/1.0 (+https://github.com/sfw/loom)"
SUPPORTED_MARKET_PROVIDERS = frozenset({"stooq"})


class MarketDataProviderError(RuntimeError):
    """Raised when market-data provider output is invalid."""


def _headers() -> dict[str, str]:
    return {
        "User-Agent": os.environ.get("LOOM_WEB_USER_AGENT", "").strip() or DEFAULT_USER_AGENT,
        "Accept": "text/csv,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }


def normalize_stooq_symbol(symbol: str) -> str:
    token = symbol.strip().lower()
    if not token:
        raise MarketDataProviderError("symbol is required")
    if "." not in token:
        # Default to US listings for common equity/ETF tickers.
        token = f"{token}.us"
    return token


def _parse_iso_day(value: str) -> date | None:
    text = value.strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _in_range(day: date, start: str, end: str) -> bool:
    start_day = _parse_iso_day(start) if start else None
    end_day = _parse_iso_day(end) if end else None
    if start_day and day < start_day:
        return False
    if end_day and day > end_day:
        return False
    return True


def _parse_stooq_csv(csv_text: str, *, start: str = "", end: str = "") -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
    rows: list[dict[str, Any]] = []
    for row in reader:
        if not isinstance(row, dict):
            continue
        day = _parse_iso_day(str(row.get("Date", "")).strip())
        if day is None:
            continue
        if not _in_range(day, start, end):
            continue
        open_v = safe_float(row.get("Open"))
        high_v = safe_float(row.get("High"))
        low_v = safe_float(row.get("Low"))
        close_v = safe_float(row.get("Close"))
        volume_v = safe_float(row.get("Volume"))
        if close_v is None:
            continue
        rows.append(
            {
                "date": day.isoformat(),
                "open": open_v,
                "high": high_v,
                "low": low_v,
                "close": close_v,
                "volume": volume_v if volume_v is not None else 0.0,
            }
        )
    rows.sort(key=lambda item: item["date"])
    return rows


async def fetch_stooq_daily_prices(
    *,
    symbol: str,
    start: str = "",
    end: str = "",
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    normalized = normalize_stooq_symbol(symbol)
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(DEFAULT_TIMEOUT_SECONDS),
            follow_redirects=True,
            headers=_headers(),
        )
    try:
        response = await client.get(
            "https://stooq.com/q/d/l/",
            params={"s": normalized, "i": "d"},
        )
        response.raise_for_status()
        rows = _parse_stooq_csv(response.text, start=start, end=end)
    finally:
        if owns_client:
            await client.aclose()

    if not rows:
        raise MarketDataProviderError(f"No market data returned for symbol '{symbol}'")

    return {
        "provider": "stooq",
        "symbol": symbol.upper(),
        "normalized_symbol": normalized,
        "rows": rows,
        "source_url": f"https://stooq.com/q/d/l/?s={normalized}&i=d",
        "as_of": rows[-1]["date"],
    }


__all__ = [
    "MarketDataProviderError",
    "SUPPORTED_MARKET_PROVIDERS",
    "fetch_stooq_daily_prices",
    "normalize_stooq_symbol",
]
