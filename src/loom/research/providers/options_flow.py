"""Keyless options-flow provider adapters and parsers."""

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

# Public CSV used on Cboe's market statistics page.
CBOE_TOTAL_PUT_CALL_URL = "https://cdn.cboe.com/data/us/options/market_statistics/daily/totalpc.csv"
CBOE_ALLOWED_HOSTS = frozenset({"cdn.cboe.com", "www.cboe.com", "cboe.com"})


class OptionsFlowProviderError(RuntimeError):
    """Raised when options flow source payloads are invalid."""


def _headers() -> dict[str, str]:
    return {
        "User-Agent": os.environ.get("LOOM_WEB_USER_AGENT", "").strip() or DEFAULT_USER_AGENT,
        "Accept": "text/csv,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }


def _norm_key(value: object) -> str:
    out = "".join(ch for ch in str(value or "").lower() if ch.isalnum())
    return out


def _parse_iso_day(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        # Handles clean ISO date input and strings like YYYY-MM-DDTHH:MM:SS.
        return date.fromisoformat(text[:10])
    except ValueError:
        pass
    if len(text) == 8 and text.isdigit():
        try:
            return date.fromisoformat(f"{text[:4]}-{text[4:6]}-{text[6:8]}")
        except ValueError:
            return None
    if "/" in text:
        parts = text.split("/")
        if len(parts) == 3:
            mm, dd, yyyy = parts
            if len(yyyy) == 4 and mm.isdigit() and dd.isdigit():
                try:
                    return date.fromisoformat(
                        f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}",
                    )
                except ValueError:
                    return None
    return None


def _in_range(day: date, start_date: str, end_date: str) -> bool:
    start = _parse_iso_day(start_date) if start_date else None
    end = _parse_iso_day(end_date) if end_date else None
    if start and day < start:
        return False
    if end and day > end:
        return False
    return True


def _pick_value(row: dict[str, Any], names: set[str]) -> str:
    for key, value in row.items():
        if _norm_key(key) in names:
            return str(value or "").strip()
    return ""


def _pick_float(row: dict[str, Any], names: set[str]) -> float | None:
    value = _pick_value(row, names)
    return safe_float(value)


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


def parse_options_flow_csv(
    csv_text: str,
    *,
    source: str = "",
) -> list[dict[str, Any]]:
    """Parse heterogeneous options-flow CSV files into a normalized schema."""
    cleaned = csv_text.lstrip("\ufeff").strip()
    if not cleaned:
        raise OptionsFlowProviderError("Options flow CSV payload is empty")

    dialect = _guess_dialect(cleaned)
    reader = csv.DictReader(io.StringIO(cleaned), dialect=dialect)

    date_keys = {"date", "tradedate", "asofdate", "asof", "pricedate"}
    symbol_keys = {"symbol", "ticker", "underlying", "underlyingsymbol", "root"}
    put_keys = {
        "putvolume",
        "putsvolume",
        "putcontracts",
        "putcontractvolume",
        "put",
        "putvol",
        "totalputvolume",
    }
    call_keys = {
        "callvolume",
        "callsvolume",
        "callcontracts",
        "callcontractvolume",
        "call",
        "callvol",
        "totalcallvolume",
    }
    total_keys = {"totalvolume", "volume", "totalcontracts", "contracts"}
    ratio_keys = {"putcallratio", "pcratio", "pc", "ratio", "totalpcratio"}
    open_interest_keys = {
        "openinterest",
        "totalopeninterest",
        "oi",
        "openinterestcontracts",
    }

    rows: list[dict[str, Any]] = []
    for raw in reader:
        if not isinstance(raw, dict):
            continue
        day = _parse_iso_day(_pick_value(raw, date_keys))
        if day is None:
            continue

        symbol = _pick_value(raw, symbol_keys).upper()
        put_volume = _pick_float(raw, put_keys)
        call_volume = _pick_float(raw, call_keys)
        total_volume = _pick_float(raw, total_keys)
        ratio = _pick_float(raw, ratio_keys)
        if ratio is None and put_volume is not None and call_volume not in (None, 0):
            ratio = put_volume / call_volume
        if total_volume is None and (put_volume is not None or call_volume is not None):
            total_volume = float(put_volume or 0.0) + float(call_volume or 0.0)

        if ratio is None and put_volume is None and call_volume is None and total_volume is None:
            continue

        rows.append(
            {
                "date": day.isoformat(),
                "symbol": symbol,
                "put_volume": put_volume,
                "call_volume": call_volume,
                "total_volume": total_volume,
                "put_call_ratio": ratio,
                "open_interest": _pick_float(raw, open_interest_keys),
                "source": source,
            }
        )

    rows.sort(key=lambda row: (str(row.get("date", "")), str(row.get("symbol", ""))))
    if not rows:
        raise OptionsFlowProviderError("No parseable options flow rows found")
    return rows


def filter_options_rows(
    rows: list[dict[str, Any]],
    *,
    symbol: str = "",
    start_date: str = "",
    end_date: str = "",
    max_rows: int = 1000,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    needle = symbol.strip().upper()
    max_rows = max(1, min(100_000, int(max_rows)))

    for row in rows:
        day = _parse_iso_day(row.get("date"))
        if day is None:
            continue
        if not _in_range(day, start_date, end_date):
            continue
        row_symbol = str(row.get("symbol", "")).strip().upper()
        if needle and row_symbol and row_symbol != needle:
            continue
        if needle and not row_symbol:
            # Aggregate rows without symbols are not valid for symbol-level ops.
            continue
        filtered.append(row)

    if len(filtered) > max_rows:
        filtered = filtered[-max_rows:]
    return filtered


def summarize_options_flow(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "as_of": None,
            "avg_put_call_ratio": None,
            "total_put_volume": 0.0,
            "total_call_volume": 0.0,
            "total_volume": 0.0,
        }

    put_values = [
        float(v)
        for v in (safe_float(r.get("put_volume")) for r in rows)
        if v is not None
    ]
    call_values = [
        float(v) for v in (safe_float(r.get("call_volume")) for r in rows) if v is not None
    ]
    ratio_values = [
        float(v) for v in (safe_float(r.get("put_call_ratio")) for r in rows) if v is not None
    ]
    total_values = [
        float(v) for v in (safe_float(r.get("total_volume")) for r in rows) if v is not None
    ]

    avg_ratio = (sum(ratio_values) / len(ratio_values)) if ratio_values else None
    return {
        "row_count": len(rows),
        "as_of": rows[-1].get("date"),
        "avg_put_call_ratio": avg_ratio,
        "latest_put_call_ratio": ratio_values[-1] if ratio_values else None,
        "total_put_volume": sum(put_values),
        "total_call_volume": sum(call_values),
        "total_volume": sum(total_values),
    }


async def fetch_options_flow_csv(
    *,
    source_url: str,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch and parse options flow CSV from a URL."""
    url = source_url.strip()
    if not url:
        raise OptionsFlowProviderError("source_url is required")

    parsed = httpx.URL(url)
    if not parsed.scheme.startswith("http"):
        raise OptionsFlowProviderError("source_url must be http(s)")

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

    rows = parse_options_flow_csv(response.text, source=url)
    return rows, str(response.url)


async def fetch_cboe_put_call_history(
    *,
    start_date: str = "",
    end_date: str = "",
    max_rows: int = 2_000,
    source_url: str = CBOE_TOTAL_PUT_CALL_URL,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Fetch Cboe aggregate put/call history from public CSV endpoints."""
    rows, resolved_url = await fetch_options_flow_csv(source_url=source_url, client=client)
    filtered = filter_options_rows(
        rows,
        start_date=start_date,
        end_date=end_date,
        max_rows=max_rows,
    )
    if not filtered:
        raise OptionsFlowProviderError("No Cboe options rows matched filters")
    return {
        "provider": "cboe",
        "rows": filtered,
        "as_of": filtered[-1].get("date"),
        "source_url": resolved_url,
    }


__all__ = [
    "CBOE_ALLOWED_HOSTS",
    "CBOE_TOTAL_PUT_CALL_URL",
    "OptionsFlowProviderError",
    "fetch_cboe_put_call_history",
    "fetch_options_flow_csv",
    "filter_options_rows",
    "parse_options_flow_csv",
    "summarize_options_flow",
]
