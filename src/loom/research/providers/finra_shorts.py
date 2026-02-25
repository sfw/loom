"""FINRA short-interest and short-sale-volume adapters."""

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
FINRA_SHORT_INTEREST_BASE = "https://cdn.finra.org/equity/otcmarket/biweekly/"
FINRA_DAILY_SHORT_VOLUME_BASE = "https://cdn.finra.org/equity/regsho/daily/"


class FinraShortDataError(RuntimeError):
    """Raised when FINRA short-data payloads are invalid."""


def _headers() -> dict[str, str]:
    return {
        "User-Agent": os.environ.get("LOOM_WEB_USER_AGENT", "").strip() or DEFAULT_USER_AGENT,
        "Accept": "text/csv,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }


def _norm_key(value: object) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _parse_day(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) == 8 and text.isdigit():
        try:
            return date.fromisoformat(f"{text[:4]}-{text[4:6]}-{text[6:8]}")
        except ValueError:
            return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _guess_dialect(text: str) -> csv.Dialect:
    sample = text[:4096]
    try:
        return csv.Sniffer().sniff(sample)
    except csv.Error:
        class _Fallback(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            doublequote = True
            skipinitialspace = True
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL

        return _Fallback()


def _pick_value(row: dict[str, Any], aliases: set[str]) -> str:
    for key, value in row.items():
        if _norm_key(key) in aliases:
            return str(value or "").strip()
    return ""


def _pick_float(row: dict[str, Any], aliases: set[str]) -> float | None:
    return safe_float(_pick_value(row, aliases))


def parse_finra_short_interest_csv(text: str, *, source: str = "") -> list[dict[str, Any]]:
    payload = text.lstrip("\ufeff").strip()
    if not payload:
        raise FinraShortDataError("FINRA short-interest payload is empty")

    reader = csv.DictReader(io.StringIO(payload), dialect=_guess_dialect(payload))

    symbol_keys = {"symbol", "symbolcode", "ticker"}
    date_keys = {"settlementdate", "date", "asofdate", "reportdate"}
    short_interest_keys = {
        "currentshortpositionquantity",
        "shortinterest",
        "shortinterestquantity",
        "shortposition",
    }
    avg_daily_volume_keys = {
        "averagedailyvolumequantity",
        "averagedailyvolume",
        "avgdailyvolume",
    }
    days_to_cover_keys = {"daystocoverquantity", "daystocover", "shortinterestdaystocover"}

    rows: list[dict[str, Any]] = []
    for raw in reader:
        if not isinstance(raw, dict):
            continue
        symbol = _pick_value(raw, symbol_keys).upper()
        if not symbol:
            continue
        day = _parse_day(_pick_value(raw, date_keys))
        if day is None:
            continue

        short_interest = _pick_float(raw, short_interest_keys)
        if short_interest is None:
            continue

        rows.append(
            {
                "date": day.isoformat(),
                "symbol": symbol,
                "short_interest": short_interest,
                "average_daily_volume": _pick_float(raw, avg_daily_volume_keys),
                "days_to_cover": _pick_float(raw, days_to_cover_keys),
                "source": source,
            }
        )

    rows.sort(key=lambda row: (str(row.get("date", "")), str(row.get("symbol", ""))))
    if not rows:
        raise FinraShortDataError("No parseable short-interest rows found")
    return rows


def parse_finra_daily_short_volume(text: str, *, source: str = "") -> list[dict[str, Any]]:
    payload = text.lstrip("\ufeff").strip()
    if not payload:
        raise FinraShortDataError("FINRA daily short-volume payload is empty")

    delimiter = "|" if "|" in payload.splitlines()[0] else ","
    reader = csv.DictReader(io.StringIO(payload), delimiter=delimiter)

    symbol_keys = {"symbol", "ticker"}
    date_keys = {"date", "tradedate", "tradingdate"}
    short_volume_keys = {"shortvolume", "shortvol"}
    short_exempt_keys = {"shortexemptvolume", "shortexemptvol"}
    total_volume_keys = {"totalvolume", "volume"}
    market_keys = {"market", "marketcenter", "marketcenterid"}

    rows: list[dict[str, Any]] = []
    for raw in reader:
        if not isinstance(raw, dict):
            continue
        symbol = _pick_value(raw, symbol_keys).upper()
        if not symbol:
            continue
        day = _parse_day(_pick_value(raw, date_keys))
        if day is None:
            continue

        short_volume = _pick_float(raw, short_volume_keys)
        total_volume = _pick_float(raw, total_volume_keys)
        if short_volume is None or total_volume is None:
            continue

        rows.append(
            {
                "date": day.isoformat(),
                "symbol": symbol,
                "short_volume": short_volume,
                "short_exempt_volume": _pick_float(raw, short_exempt_keys),
                "total_volume": total_volume,
                "short_volume_ratio": (short_volume / total_volume) if total_volume > 0 else None,
                "market": _pick_value(raw, market_keys),
                "source": source,
            }
        )

    rows.sort(key=lambda row: (str(row.get("date", "")), str(row.get("symbol", ""))))
    if not rows:
        raise FinraShortDataError("No parseable daily short-volume rows found")
    return rows


async def _fetch_text(
    *,
    url: str,
    client: httpx.AsyncClient | None = None,
) -> tuple[str, str]:
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(DEFAULT_TIMEOUT_SECONDS),
            follow_redirects=True,
            headers=_headers(),
        )
    try:
        response = await client.get(url)
        response.raise_for_status()
    finally:
        if owns_client:
            await client.aclose()
    return response.text, str(response.url)


def _normalize_date_token(value: object) -> str:
    text = "".join(ch for ch in str(value or "") if ch.isdigit())
    if len(text) != 8:
        raise FinraShortDataError("date must be YYYYMMDD or YYYY-MM-DD")
    return text


async def fetch_finra_short_interest_csv(
    *,
    date_token: str,
    source_url: str = "",
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    url = source_url.strip()
    if not url:
        token = _normalize_date_token(date_token)
        url = f"{FINRA_SHORT_INTEREST_BASE}shrt{token}.csv"

    text, resolved = await _fetch_text(url=url, client=client)
    rows = parse_finra_short_interest_csv(text, source=resolved)
    return {
        "provider": "finra",
        "dataset": "short_interest",
        "rows": rows,
        "source_url": resolved,
        "as_of": rows[-1].get("date"),
    }


async def fetch_finra_daily_short_volume(
    *,
    date_token: str,
    market: str = "CNMS",
    source_url: str = "",
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    url = source_url.strip()
    if not url:
        token = _normalize_date_token(date_token)
        market_token = str(market or "CNMS").strip().upper()
        url = f"{FINRA_DAILY_SHORT_VOLUME_BASE}{market_token}shvol{token}.txt"

    text, resolved = await _fetch_text(url=url, client=client)
    rows = parse_finra_daily_short_volume(text, source=resolved)
    return {
        "provider": "finra",
        "dataset": "daily_short_volume",
        "rows": rows,
        "source_url": resolved,
        "as_of": rows[-1].get("date"),
    }


__all__ = [
    "FINRA_DAILY_SHORT_VOLUME_BASE",
    "FINRA_SHORT_INTEREST_BASE",
    "FinraShortDataError",
    "fetch_finra_daily_short_volume",
    "fetch_finra_short_interest_csv",
    "parse_finra_daily_short_volume",
    "parse_finra_short_interest_csv",
]
