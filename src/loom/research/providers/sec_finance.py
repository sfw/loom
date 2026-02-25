"""SEC company mapping and fundamentals adapters (keyless)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import date
from typing import Any

import httpx

from loom.research.finance import safe_float

DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_USER_AGENT = "Loom/1.0 (+https://github.com/sfw/loom)"
SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"


class SecDataError(RuntimeError):
    """Raised when SEC data payloads are invalid."""


def _headers() -> dict[str, str]:
    return {
        "User-Agent": os.environ.get("LOOM_WEB_USER_AGENT", "").strip() or DEFAULT_USER_AGENT,
        "Accept": "application/json,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }


def _norm_text(value: object, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def _normalize_cik(value: object) -> str:
    text = "".join(ch for ch in str(value or "") if ch.isdigit())
    if not text:
        return ""
    return text.zfill(10)


def _parse_day(value: object) -> date | None:
    text = _norm_text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


async def fetch_sec_ticker_map(
    *,
    client: httpx.AsyncClient | None = None,
) -> dict[str, dict[str, Any]]:
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(DEFAULT_TIMEOUT_SECONDS),
            follow_redirects=True,
            headers=_headers(),
        )
    try:
        response = await client.get(SEC_TICKER_URL)
        response.raise_for_status()
        payload = response.json()
    finally:
        if owns_client:
            await client.aclose()

    out: dict[str, dict[str, Any]] = {}
    if isinstance(payload, list):
        iterator = enumerate(payload)
    elif isinstance(payload, dict):
        iterator = payload.items()
    else:
        raise SecDataError("Unexpected SEC ticker payload format")

    for _idx, item in iterator:
        if not isinstance(item, dict):
            continue
        ticker = _norm_text(item.get("ticker")).upper()
        cik = _normalize_cik(item.get("cik_str") or item.get("cik"))
        title = _norm_text(item.get("title") or item.get("name"))
        if not ticker or not cik:
            continue
        out[ticker] = {
            "ticker": ticker,
            "cik": cik,
            "name": title,
            "source": SEC_TICKER_URL,
        }
    if not out:
        raise SecDataError("SEC ticker mapping returned no records")
    return out


async def resolve_ticker_to_cik(
    *,
    ticker: str,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    mapping = await fetch_sec_ticker_map(client=client)
    key = ticker.strip().upper()
    if key not in mapping:
        raise SecDataError(f"Ticker '{ticker}' not found in SEC mapping")
    return mapping[key]


async def fetch_company_facts(
    *,
    cik: str,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    cik10 = _normalize_cik(cik)
    if not cik10:
        raise SecDataError("CIK is required")

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(DEFAULT_TIMEOUT_SECONDS),
            follow_redirects=True,
            headers=_headers(),
        )
    try:
        response = await client.get(SEC_COMPANYFACTS_URL.format(cik10=cik10))
        response.raise_for_status()
        payload = response.json()
    finally:
        if owns_client:
            await client.aclose()

    if not isinstance(payload, dict):
        raise SecDataError("Unexpected SEC companyfacts format")
    return payload


def _concept_points(
    facts_payload: Mapping[str, Any],
    *,
    tag: str,
    taxonomy: str = "us-gaap",
) -> list[dict[str, Any]]:
    facts = facts_payload.get("facts", {}) if isinstance(facts_payload, Mapping) else {}
    tx = facts.get(taxonomy, {}) if isinstance(facts, Mapping) else {}
    concept = tx.get(tag, {}) if isinstance(tx, Mapping) else {}
    units = concept.get("units", {}) if isinstance(concept, Mapping) else {}
    if not isinstance(units, Mapping):
        return []

    rows: list[dict[str, Any]] = []
    for unit_name, entries in units.items():
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, Mapping):
                continue
            val = safe_float(item.get("val"))
            end_day = _parse_day(item.get("end"))
            if val is None or end_day is None:
                continue
            rows.append(
                {
                    "tag": tag,
                    "taxonomy": taxonomy,
                    "unit": str(unit_name),
                    "value": val,
                    "end": end_day.isoformat(),
                    "form": _norm_text(item.get("form")),
                    "fy": _norm_text(item.get("fy")),
                    "fp": _norm_text(item.get("fp")),
                    "accn": _norm_text(item.get("accn")),
                    "filed": _norm_text(item.get("filed")),
                }
            )
    rows.sort(key=lambda row: (row["end"], row.get("filed", "")), reverse=True)
    return rows


def extract_latest_value(
    facts_payload: Mapping[str, Any],
    *,
    tag: str,
    taxonomy: str = "us-gaap",
) -> dict[str, Any] | None:
    rows = _concept_points(facts_payload, tag=tag, taxonomy=taxonomy)
    if not rows:
        return None
    return rows[0]


def extract_ttm_value(
    facts_payload: Mapping[str, Any],
    *,
    tag: str,
    taxonomy: str = "us-gaap",
) -> dict[str, Any] | None:
    rows = _concept_points(facts_payload, tag=tag, taxonomy=taxonomy)
    if not rows:
        return None

    # Prefer latest 4 quarterly values when available.
    quarterly = [row for row in rows if row.get("form") == "10-Q"]
    if len(quarterly) >= 4:
        top4 = quarterly[:4]
        return {
            "tag": tag,
            "taxonomy": taxonomy,
            "unit": top4[0]["unit"],
            "value": sum(float(row["value"]) for row in top4),
            "end": top4[0]["end"],
            "basis": "sum_latest_4_quarters",
            "points": top4,
        }

    # Fallback to latest annual/any reported value.
    latest = rows[0]
    return {
        "tag": tag,
        "taxonomy": taxonomy,
        "unit": latest["unit"],
        "value": float(latest["value"]),
        "end": latest["end"],
        "basis": "latest_reported_value",
        "points": [latest],
    }


__all__ = [
    "SEC_COMPANYFACTS_URL",
    "SEC_TICKER_URL",
    "SecDataError",
    "extract_latest_value",
    "extract_ttm_value",
    "fetch_company_facts",
    "fetch_sec_ticker_map",
    "resolve_ticker_to_cik",
]
