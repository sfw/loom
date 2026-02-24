"""Currency/FX helpers built on keyless public reference-rate datasets."""

from __future__ import annotations

import csv
import io
import os
import time
from datetime import date, datetime
from typing import Any

import httpx

ECB_CSV_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.csv"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_CACHE_TTL_SECONDS = 6 * 60 * 60
DEFAULT_USER_AGENT = "Loom/1.0 (+https://github.com/sfw/loom)"

_ECB_CACHE: dict[str, Any] = {"fetched_at": 0.0, "rates": {}}


def _headers() -> dict[str, str]:
    return {
        "User-Agent": os.environ.get("LOOM_WEB_USER_AGENT", "").strip() or DEFAULT_USER_AGENT,
        "Accept": "text/csv,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_iso_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _normalize_currency(value: object) -> str:
    text = str(value or "").strip().upper()
    return text


async def fetch_ecb_reference_rates(
    *,
    force_refresh: bool = False,
    client: httpx.AsyncClient | None = None,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
) -> dict[str, dict[str, float]]:
    now = time.time()
    cached_rates = _ECB_CACHE.get("rates")
    fetched_at = float(_ECB_CACHE.get("fetched_at", 0.0) or 0.0)
    if (
        not force_refresh
        and isinstance(cached_rates, dict)
        and cached_rates
        and now - fetched_at < max(60, cache_ttl_seconds)
    ):
        return cached_rates

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(DEFAULT_TIMEOUT_SECONDS),
            follow_redirects=True,
            headers=_headers(),
        )
    try:
        response = await client.get(ECB_CSV_URL)
        response.raise_for_status()
        text = response.text or ""
    finally:
        if owns_client:
            await client.aclose()

    rates = _parse_ecb_csv(text)
    _ECB_CACHE["fetched_at"] = now
    _ECB_CACHE["rates"] = rates
    return rates


def _parse_ecb_csv(csv_text: str) -> dict[str, dict[str, float]]:
    reader = csv.DictReader(io.StringIO(csv_text))
    out: dict[str, dict[str, float]] = {}
    for row in reader:
        if not isinstance(row, dict):
            continue
        day = _parse_iso_date(row.get("Date"))
        if day is None:
            continue
        day_key = day.isoformat()
        day_rates: dict[str, float] = {"EUR": 1.0}
        for code, raw in row.items():
            if code == "Date":
                continue
            curr = _normalize_currency(code)
            val = _safe_float(raw)
            if not curr or val is None or val <= 0:
                continue
            day_rates[curr] = val
        out[day_key] = day_rates
    return out


def _nearest_available_day(
    rates_by_day: dict[str, dict[str, float]],
    *,
    target_day: date,
    max_lookback_days: int = 31,
) -> str | None:
    # ECB reference rates are business-day-only. Search backward for nearest
    # available date with data.
    day = target_day
    for _ in range(max(0, max_lookback_days) + 1):
        key = day.isoformat()
        if key in rates_by_day:
            return key
        day = day.fromordinal(day.toordinal() - 1)
    return None


def _cross_rate(
    *,
    from_currency: str,
    to_currency: str,
    from_per_eur: float,
    to_per_eur: float,
) -> float:
    # ECB data encodes: 1 EUR = X CCY.
    # from->to = (to_per_eur / from_per_eur).
    if from_currency == to_currency:
        return 1.0
    return to_per_eur / from_per_eur


async def convert_via_ecb_reference_rates(
    *,
    amount: float,
    from_currency: str,
    to_currency: str,
    from_date: date,
    to_date: date | None = None,
    client: httpx.AsyncClient | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    from_currency = _normalize_currency(from_currency)
    to_currency = _normalize_currency(to_currency)
    if not from_currency or not to_currency:
        raise ValueError("from_currency and to_currency are required")
    if amount is None:
        raise ValueError("amount is required")
    if to_date is None:
        to_date = from_date

    rates = await fetch_ecb_reference_rates(force_refresh=force_refresh, client=client)
    from_day_key = _nearest_available_day(rates, target_day=from_date)
    to_day_key = _nearest_available_day(rates, target_day=to_date)
    if from_day_key is None or to_day_key is None:
        raise ValueError("No ECB rates available for requested dates")

    from_day_rates = rates.get(from_day_key, {})
    to_day_rates = rates.get(to_day_key, {})
    from_rate_in = from_day_rates.get(from_currency)
    usd_rate_in = from_day_rates.get("USD")
    to_rate_out = to_day_rates.get(to_currency)
    usd_rate_out = to_day_rates.get("USD")

    warnings: list[str] = []
    if from_day_key != from_date.isoformat():
        warnings.append(
            "from_date has no direct ECB quote; using nearest prior business day "
            f"{from_day_key}."
        )
    if to_day_key != to_date.isoformat():
        warnings.append(
            "to_date has no direct ECB quote; using nearest prior business day "
            f"{to_day_key}."
        )

    if from_rate_in is None:
        raise ValueError(f"ECB does not provide {from_currency} quote on {from_day_key}")
    if to_rate_out is None:
        raise ValueError(f"ECB does not provide {to_currency} quote on {to_day_key}")

    fx_rate = _cross_rate(
        from_currency=from_currency,
        to_currency=to_currency,
        from_per_eur=from_rate_in,
        to_per_eur=to_rate_out,
    )
    converted = float(amount) * fx_rate

    # Also return USD cross snapshot when available for downstream inflation
    # normalization paths.
    usd_cross = None
    if usd_rate_in is not None and usd_rate_out is not None:
        usd_cross = {
            "from_day_usd_per_eur": usd_rate_in,
            "to_day_usd_per_eur": usd_rate_out,
            "from_to_usd_rate": _cross_rate(
                from_currency=from_currency,
                to_currency="USD",
                from_per_eur=from_rate_in,
                to_per_eur=usd_rate_in,
            ),
            "usd_to_target_rate": _cross_rate(
                from_currency="USD",
                to_currency=to_currency,
                from_per_eur=usd_rate_out,
                to_per_eur=to_rate_out,
            ),
        }

    return {
        "amount": float(amount),
        "from_currency": from_currency,
        "to_currency": to_currency,
        "from_date_effective": from_day_key,
        "to_date_effective": to_day_key,
        "from_per_eur": from_rate_in,
        "to_per_eur": to_rate_out,
        "fx_rate": fx_rate,
        "converted_amount": converted,
        "warnings": warnings,
        "source": "ecb_reference_rates",
        "source_url": ECB_CSV_URL,
        "usd_cross": usd_cross,
    }
