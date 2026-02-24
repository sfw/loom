"""Keyless economic-data provider adapters with normalized output."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import httpx

from loom.research.text import coerce_int, parse_year

DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_USER_AGENT = "Loom/1.0 (+https://github.com/sfw/loom)"
MAX_RESULTS = 100
MAX_OBSERVATIONS = 2_000

SUPPORTED_ECONOMIC_PROVIDERS = frozenset(
    {"world_bank", "oecd", "eurostat", "dbnomics", "bls"}
)
_SERIES_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{4,}$")
_NULLISH_PAYLOAD_TEXT = frozenset(
    {
        "",
        "null",
        "'null'",
        '"null"',
        "none",
        "'none'",
        '"none"',
        "(null)",
    }
)
_JSON_XSSI_PREFIXES = (")]}',", "for(;;);")


class EconomicProviderError(RuntimeError):
    """Raised when an economic provider returns invalid/unsupported output."""


def _headers() -> dict[str, str]:
    return {
        "User-Agent": os.environ.get("LOOM_WEB_USER_AGENT", "").strip() or DEFAULT_USER_AGENT,
        "Accept": "application/json,text/csv,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }


def _clamp_int(value: object, default: int, lo: int, hi: int) -> int:
    parsed = coerce_int(value, default=default)
    if parsed is None:
        parsed = default
    return max(lo, min(hi, parsed))


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            return float(text)
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_text(value: object, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def _make_coverage(observations: list[dict[str, Any]]) -> dict[str, Any]:
    if not observations:
        return {"count": 0, "start": None, "end": None}
    periods = [str(item.get("period", "")).strip() for item in observations]
    periods = [p for p in periods if p]
    periods.sort()
    return {
        "count": len(observations),
        "start": periods[0] if periods else None,
        "end": periods[-1] if periods else None,
    }


def _dedupe_hits(rows: list[dict[str, Any]], max_results: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        series_id = _norm_text(row.get("series_id")).lower()
        key = f"{_norm_text(row.get('provider')).lower()}::{series_id}"
        if not series_id or key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= max_results:
            break
    return out


async def _request_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: Mapping[str, object] | None = None,
) -> Any:
    response = await client.get(url, params=params)
    response.raise_for_status()
    return _decode_json_response(response, url=url)


def _normalize_json_payload(payload: Any) -> Any:
    if payload is None:
        return {}
    if isinstance(payload, str):
        text = payload.strip()
        if text.lower() in _NULLISH_PAYLOAD_TEXT:
            return {}
    return payload


def _decode_json_response(response: httpx.Response, *, url: str) -> Any:
    if not response.content:
        return {}
    try:
        return _normalize_json_payload(response.json())
    except json.JSONDecodeError:
        text = response.text.lstrip("\ufeff").strip()
        if text.lower() in _NULLISH_PAYLOAD_TEXT:
            return {}
        for prefix in _JSON_XSSI_PREFIXES:
            if text.startswith(prefix):
                _, _, remainder = text.partition("\n")
                text = (remainder or text[len(prefix) :]).strip()
                break
        if text.lower() in _NULLISH_PAYLOAD_TEXT:
            return {}
        try:
            return _normalize_json_payload(json.loads(text))
        except json.JSONDecodeError as e:
            raise EconomicProviderError(f"Invalid JSON payload from {url}") from e


def _normalize_period(text: object) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    # Keep original if it already looks structured.
    if re.match(r"^\d{4}(-\d{2}(-\d{2})?)?$", raw):
        return raw
    year = parse_year(raw)
    if year is not None:
        return str(year)
    return raw


def _filter_observations(
    observations: list[dict[str, Any]],
    *,
    start_period: str,
    end_period: str,
    max_observations: int,
) -> list[dict[str, Any]]:
    start_year = parse_year(start_period)
    end_year = parse_year(end_period)
    out: list[dict[str, Any]] = []
    for row in observations:
        period = _normalize_period(row.get("period"))
        value = _safe_float(row.get("value"))
        if not period or value is None:
            continue
        year = parse_year(period)
        if start_year is not None and year is not None and year < start_year:
            continue
        if end_year is not None and year is not None and year > end_year:
            continue
        out.append(
            {
                "period": period,
                "value": value,
                "dimension": row.get("dimension") if row.get("dimension") else None,
                "note": row.get("note") if row.get("note") else None,
            }
        )
    out.sort(key=lambda item: str(item.get("period", "")))
    if len(out) > max_observations:
        out = out[-max_observations:]
    return out


async def economic_search(
    *,
    provider: str,
    query: str,
    max_results: int,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    provider = provider.strip().lower()
    if provider not in SUPPORTED_ECONOMIC_PROVIDERS:
        raise EconomicProviderError(f"Unsupported provider '{provider}'")

    max_results = _clamp_int(max_results, default=15, lo=1, hi=MAX_RESULTS)
    query = query.strip()
    if not query:
        raise EconomicProviderError("query is required for search")

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(DEFAULT_TIMEOUT_SECONDS),
            follow_redirects=True,
            headers=_headers(),
        )

    try:
        if provider == "world_bank":
            hits = await _search_world_bank(client, query=query, max_results=max_results)
        elif provider == "dbnomics":
            hits = await _search_dbnomics(client, query=query, max_results=max_results)
        elif provider == "oecd":
            hits = await _search_oecd(client, query=query, max_results=max_results)
        elif provider == "eurostat":
            hits = await _search_eurostat(client, query=query, max_results=max_results)
        else:
            hits = _search_bls(query=query, max_results=max_results)
        return _dedupe_hits(hits, max_results=max_results)
    finally:
        if owns_client:
            await client.aclose()


async def economic_get_observations(
    *,
    provider: str,
    series_id: str,
    start_period: str = "",
    end_period: str = "",
    max_observations: int = 240,
    filters: Mapping[str, object] | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    provider = provider.strip().lower()
    if provider not in SUPPORTED_ECONOMIC_PROVIDERS:
        raise EconomicProviderError(f"Unsupported provider '{provider}'")

    series_id = series_id.strip()
    if not series_id:
        raise EconomicProviderError("series_id is required")

    max_observations = _clamp_int(
        max_observations,
        default=240,
        lo=1,
        hi=MAX_OBSERVATIONS,
    )
    filters = dict(filters or {})

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(DEFAULT_TIMEOUT_SECONDS),
            follow_redirects=True,
            headers=_headers(),
        )

    try:
        if provider == "world_bank":
            payload = await _obs_world_bank(
                client,
                series_id=series_id,
                start_period=start_period,
                end_period=end_period,
                max_observations=max_observations,
                filters=filters,
            )
        elif provider == "dbnomics":
            payload = await _obs_dbnomics(
                client,
                series_id=series_id,
                start_period=start_period,
                end_period=end_period,
                max_observations=max_observations,
            )
        elif provider == "oecd":
            payload = await _obs_oecd(
                client,
                series_id=series_id,
                start_period=start_period,
                end_period=end_period,
                max_observations=max_observations,
            )
        elif provider == "eurostat":
            payload = await _obs_eurostat(
                client,
                series_id=series_id,
                start_period=start_period,
                end_period=end_period,
                max_observations=max_observations,
                filters=filters,
            )
        else:
            payload = await _obs_bls(
                client,
                series_id=series_id,
                start_period=start_period,
                end_period=end_period,
                max_observations=max_observations,
            )

        observations = _filter_observations(
            list(payload.get("observations", [])),
            start_period=start_period,
            end_period=end_period,
            max_observations=max_observations,
        )
        payload["observations"] = observations
        payload["coverage"] = _make_coverage(observations)
        return payload
    finally:
        if owns_client:
            await client.aclose()


async def economic_get_series(
    *,
    provider: str,
    series_id: str,
    start_period: str = "",
    end_period: str = "",
    max_observations: int = 25,
    filters: Mapping[str, object] | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    payload = await economic_get_observations(
        provider=provider,
        series_id=series_id,
        start_period=start_period,
        end_period=end_period,
        max_observations=max_observations,
        filters=filters,
        client=client,
    )
    return {
        "provider": payload.get("provider"),
        "series_id": payload.get("series_id"),
        "title": payload.get("title"),
        "description": payload.get("description"),
        "dataset": payload.get("dataset"),
        "unit": payload.get("unit"),
        "frequency": payload.get("frequency"),
        "source_url": payload.get("source_url"),
        "coverage": payload.get("coverage"),
        "sample_observations": payload.get("observations", []),
    }


async def _search_world_bank(
    client: httpx.AsyncClient,
    *,
    query: str,
    max_results: int,
) -> list[dict[str, Any]]:
    payload = await _request_json(
        client,
        "https://api.worldbank.org/v2/indicator",
        params={"format": "json", "per_page": 500, "page": 1},
    )
    rows: list[dict[str, Any]] = []
    items = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    q = query.lower()
    for item in items:
        if not isinstance(item, dict):
            continue
        series_id = _norm_text(item.get("id"))
        title = _norm_text(item.get("name"))
        description = _norm_text(item.get("sourceNote"))
        text_blob = " ".join([series_id, title, description]).lower()
        if q not in text_blob:
            continue
        rows.append(
            {
                "provider": "world_bank",
                "series_id": series_id,
                "title": title,
                "description": description,
                "frequency": _norm_text(item.get("periodicity")),
                "unit": _norm_text(item.get("unit")),
                "dataset": _norm_text(item.get("source", {}).get("value")),
                "source_url": f"https://api.worldbank.org/v2/indicator/{series_id}",
            }
        )
        if len(rows) >= max_results:
            break
    return rows


async def _obs_world_bank(
    client: httpx.AsyncClient,
    *,
    series_id: str,
    start_period: str,
    end_period: str,
    max_observations: int,
    filters: Mapping[str, object],
) -> dict[str, Any]:
    country = _norm_text(filters.get("country"), fallback="all")
    date_param = ""
    sy = parse_year(start_period)
    ey = parse_year(end_period)
    if sy is not None and ey is not None and sy <= ey:
        date_param = f"{sy}:{ey}"
    elif sy is not None:
        date_param = f"{sy}:{datetime.now().year}"

    params: dict[str, object] = {"format": "json", "per_page": 20_000}
    if date_param:
        params["date"] = date_param

    payload = await _request_json(
        client,
        f"https://api.worldbank.org/v2/country/{country}/indicator/{series_id}",
        params=params,
    )
    items = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    observations: list[dict[str, Any]] = []
    title = ""
    description = ""
    dataset = ""
    unit = ""
    for item in items:
        if not isinstance(item, dict):
            continue
        if not title:
            title = _norm_text(item.get("indicator", {}).get("value"), fallback=series_id)
            dataset = _norm_text(item.get("obs_status"))
        value = _safe_float(item.get("value"))
        period = _norm_text(item.get("date"))
        country_name = _norm_text(item.get("country", {}).get("value"))
        if value is None or not period:
            continue
        observations.append(
            {
                "period": period,
                "value": value,
                "dimension": {"country": country_name} if country_name else None,
                "note": _norm_text(item.get("decimal"), fallback=""),
            }
        )
        if len(observations) >= max_observations:
            break
    return {
        "provider": "world_bank",
        "series_id": series_id,
        "title": title or series_id,
        "description": description,
        "dataset": dataset,
        "unit": unit,
        "frequency": "annual",
        "source_url": (
            f"https://api.worldbank.org/v2/country/{country}/indicator/{series_id}?format=json"
        ),
        "observations": observations,
    }


async def _search_dbnomics(
    client: httpx.AsyncClient,
    *,
    query: str,
    max_results: int,
) -> list[dict[str, Any]]:
    payload = await _request_json(
        client,
        "https://api.db.nomics.world/v22/search",
        params={"q": query, "limit": max_results},
    )
    docs = []
    if isinstance(payload, dict):
        docs = payload.get("docs", [])
        if not docs and isinstance(payload.get("datasets"), dict):
            docs = payload.get("datasets", {}).get("docs", [])

    rows: list[dict[str, Any]] = []
    for item in docs if isinstance(docs, list) else []:
        if not isinstance(item, dict):
            continue
        provider_code = _norm_text(item.get("provider_code"))
        dataset_code = _norm_text(item.get("dataset_code"))
        series_code = _norm_text(item.get("series_code"))
        series_id = "/".join(
            [part for part in [provider_code, dataset_code, series_code] if part]
        )
        if not series_id:
            continue
        rows.append(
            {
                "provider": "dbnomics",
                "series_id": series_id,
                "title": _norm_text(item.get("name"), fallback=series_id),
                "description": _norm_text(item.get("description")),
                "frequency": _norm_text(item.get("frequency")),
                "unit": _norm_text(item.get("unit")),
                "dataset": _norm_text(item.get("dataset_name"), fallback=dataset_code),
                "source_url": f"https://db.nomics.world/{series_id}",
            }
        )
    return rows


async def _obs_dbnomics(
    client: httpx.AsyncClient,
    *,
    series_id: str,
    start_period: str,
    end_period: str,
    max_observations: int,
) -> dict[str, Any]:
    parts = [p for p in series_id.split("/") if p]
    if len(parts) < 3:
        raise EconomicProviderError(
            "DBnomics series_id must look like PROVIDER/DATASET/SERIES"
        )
    provider_code, dataset_code, series_code = parts[0], parts[1], "/".join(parts[2:])
    payload = await _request_json(
        client,
        f"https://api.db.nomics.world/v22/series/{provider_code}/{dataset_code}/{series_code}",
    )
    series = {}
    if isinstance(payload, dict):
        if isinstance(payload.get("series"), dict):
            series = payload["series"]
        elif isinstance(payload.get("series"), list) and payload.get("series"):
            first = payload["series"][0]
            if isinstance(first, dict):
                series = first

    periods = list(series.get("period", []) or [])
    values = list(series.get("value", []) or [])
    observations: list[dict[str, Any]] = []
    for idx, period in enumerate(periods):
        value = _safe_float(values[idx] if idx < len(values) else None)
        if value is None:
            continue
        observations.append({"period": _normalize_period(period), "value": value})
    observations = _filter_observations(
        observations,
        start_period=start_period,
        end_period=end_period,
        max_observations=max_observations,
    )
    return {
        "provider": "dbnomics",
        "series_id": series_id,
        "title": _norm_text(series.get("name"), fallback=series_id),
        "description": _norm_text(series.get("description")),
        "dataset": _norm_text(series.get("dataset_name"), fallback=dataset_code),
        "unit": _norm_text(series.get("unit")),
        "frequency": _norm_text(series.get("frequency")),
        "source_url": f"https://db.nomics.world/{series_id}",
        "observations": observations,
    }


async def _search_oecd(
    client: httpx.AsyncClient,
    *,
    query: str,
    max_results: int,
) -> list[dict[str, Any]]:
    payload = await _request_json(
        client,
        "https://sdmx.oecd.org/public/rest/dataflow/all/latest",
        params={"format": "sdmx-json"},
    )
    q = query.lower()
    rows: list[dict[str, Any]] = []
    flows = payload.get("data", {}).get("dataflows", []) if isinstance(payload, dict) else []
    for flow in flows if isinstance(flows, list) else []:
        if not isinstance(flow, dict):
            continue
        flow_id = _norm_text(flow.get("id"))
        name = _norm_text(flow.get("name"), fallback=flow_id)
        description = _norm_text(flow.get("description"))
        text_blob = " ".join([flow_id, name, description]).lower()
        if q not in text_blob:
            continue
        rows.append(
            {
                "provider": "oecd",
                "series_id": flow_id,
                "title": name,
                "description": description,
                "frequency": "",
                "unit": "",
                "dataset": flow_id,
                "source_url": f"https://sdmx.oecd.org/public/rest/data/{flow_id}",
            }
        )
        if len(rows) >= max_results:
            break
    return rows


def _parse_sdmx_values(payload: dict[str, Any]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    datasets = payload.get("data", {}).get("dataSets", []) if isinstance(payload, dict) else []
    structures = payload.get("data", {}).get("structure", {}) if isinstance(payload, dict) else {}
    dimensions = structures.get("dimensions", {}).get("observation", [])
    time_index: dict[str, str] = {}
    if isinstance(dimensions, list):
        for dim in dimensions:
            if not isinstance(dim, dict):
                continue
            if _norm_text(dim.get("id")).upper() in {"TIME_PERIOD", "TIME"}:
                values = dim.get("values", [])
                if isinstance(values, list):
                    for idx, row in enumerate(values):
                        if not isinstance(row, dict):
                            continue
                        time_index[str(idx)] = _norm_text(row.get("id"))

    if not datasets or not isinstance(datasets, list):
        return observations
    first = datasets[0] if datasets else {}
    obs_map = first.get("observations", {}) if isinstance(first, dict) else {}
    if not isinstance(obs_map, dict):
        return observations

    for key, raw_val in obs_map.items():
        value: float | None = None
        note = ""
        if isinstance(raw_val, list) and raw_val:
            value = _safe_float(raw_val[0])
            if len(raw_val) > 1:
                note = _norm_text(raw_val[1])
        else:
            value = _safe_float(raw_val)
        if value is None:
            continue
        parts = str(key).split(":")
        period = time_index.get(parts[-1], parts[-1])
        observations.append({"period": _normalize_period(period), "value": value, "note": note})
    return observations


async def _obs_oecd(
    client: httpx.AsyncClient,
    *,
    series_id: str,
    start_period: str,
    end_period: str,
    max_observations: int,
) -> dict[str, Any]:
    # OECD SDMX expects a dataset and key. We support passing either:
    # - "<DATAFLOW>/<KEY>" or
    # - "<DATAFLOW>" (uses "all" key)
    if "/" in series_id:
        dataflow, key = series_id.split("/", 1)
    else:
        dataflow, key = series_id, "all"
    payload = await _request_json(
        client,
        f"https://sdmx.oecd.org/public/rest/data/{dataflow}/{key}",
        params={"format": "sdmx-json"},
    )
    observations = _parse_sdmx_values(payload if isinstance(payload, dict) else {})
    observations = _filter_observations(
        observations,
        start_period=start_period,
        end_period=end_period,
        max_observations=max_observations,
    )
    return {
        "provider": "oecd",
        "series_id": series_id,
        "title": series_id,
        "description": "",
        "dataset": dataflow,
        "unit": "",
        "frequency": "",
        "source_url": f"https://sdmx.oecd.org/public/rest/data/{dataflow}/{key}",
        "observations": observations,
    }


async def _search_eurostat(
    client: httpx.AsyncClient,
    *,
    query: str,
    max_results: int,
) -> list[dict[str, Any]]:
    payload = await _request_json(
        client,
        "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/dataflow/ESTAT/all/latest",
        params={"format": "SDMX-JSON"},
    )
    q = query.lower()
    rows: list[dict[str, Any]] = []
    flows = payload.get("structure", {}).get("dataflows", {}) if isinstance(payload, dict) else {}
    if isinstance(flows, dict):
        iterator = flows.items()
    else:
        iterator = []
    for flow_id, body in iterator:
        title = flow_id
        description = ""
        if isinstance(body, dict):
            title = _norm_text(body.get("name"), fallback=flow_id)
            description = _norm_text(body.get("description"))
        text_blob = " ".join([flow_id, title, description]).lower()
        if q not in text_blob:
            continue
        rows.append(
            {
                "provider": "eurostat",
                "series_id": flow_id,
                "title": title,
                "description": description,
                "frequency": "",
                "unit": "",
                "dataset": flow_id,
                "source_url": (
                    "https://ec.europa.eu/eurostat/api/dissemination/"
                    f"statistics/1.0/data/{flow_id}"
                ),
            }
        )
        if len(rows) >= max_results:
            break
    return rows


def _flatten_eurostat_values(payload: dict[str, Any]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    values = payload.get("value", {}) if isinstance(payload, dict) else {}
    if not isinstance(values, dict):
        return observations
    dimension = payload.get("dimension", {}) if isinstance(payload, dict) else {}
    time_map: dict[str, str] = {}
    if isinstance(dimension, dict):
        time_dim = dimension.get("time") or dimension.get("time_period")
        if isinstance(time_dim, dict):
            category = time_dim.get("category", {})
            if isinstance(category, dict):
                index = category.get("index", {})
                if isinstance(index, dict):
                    for period, idx in index.items():
                        time_map[str(idx)] = str(period)
    for idx, raw in values.items():
        val = _safe_float(raw)
        if val is None:
            continue
        parts = str(idx).split(":")
        period = time_map.get(parts[-1], parts[-1])
        observations.append({"period": _normalize_period(period), "value": val})
    return observations


async def _obs_eurostat(
    client: httpx.AsyncClient,
    *,
    series_id: str,
    start_period: str,
    end_period: str,
    max_observations: int,
    filters: Mapping[str, object],
) -> dict[str, Any]:
    params: dict[str, object] = {"format": "JSON"}
    if start_period:
        params["sinceTimePeriod"] = start_period
    if end_period:
        params["untilTimePeriod"] = end_period
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float)):
            params[str(key)] = value
    payload = await _request_json(
        client,
        f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{series_id}",
        params=params,
    )
    observations = _flatten_eurostat_values(payload if isinstance(payload, dict) else {})
    observations = _filter_observations(
        observations,
        start_period=start_period,
        end_period=end_period,
        max_observations=max_observations,
    )
    return {
        "provider": "eurostat",
        "series_id": series_id,
        "title": series_id,
        "description": "",
        "dataset": series_id,
        "unit": "",
        "frequency": "",
        "source_url": (
            "https://ec.europa.eu/eurostat/api/dissemination/"
            f"statistics/1.0/data/{series_id}"
        ),
        "observations": observations,
    }


def _search_bls(*, query: str, max_results: int) -> list[dict[str, Any]]:
    # BLS public API does not expose a complete unauthenticated free-text
    # search endpoint. We still support direct series-id lookup patterns.
    if not _SERIES_ID_RE.match(query):
        return []
    series_id = query.upper()
    return [
        {
            "provider": "bls",
            "series_id": series_id,
            "title": f"BLS series {series_id}",
            "description": "Direct series-id lookup.",
            "frequency": "",
            "unit": "",
            "dataset": "BLS Public API",
            "source_url": f"https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id}",
        }
    ][:max_results]


async def _obs_bls(
    client: httpx.AsyncClient,
    *,
    series_id: str,
    start_period: str,
    end_period: str,
    max_observations: int,
) -> dict[str, Any]:
    params: dict[str, object] = {}
    sy = parse_year(start_period)
    ey = parse_year(end_period)
    if sy is not None:
        params["startyear"] = sy
    if ey is not None:
        params["endyear"] = ey

    payload = await _request_json(
        client,
        f"https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id}",
        params=params if params else None,
    )
    status = _norm_text(payload.get("status")) if isinstance(payload, dict) else ""
    if status and status.upper() != "REQUEST_SUCCEEDED":
        raise EconomicProviderError(f"BLS request failed: {status}")

    series = []
    if isinstance(payload, dict):
        series = payload.get("Results", {}).get("series", [])
    first = series[0] if isinstance(series, list) and series else {}
    data_rows = first.get("data", []) if isinstance(first, dict) else []
    observations: list[dict[str, Any]] = []
    for item in data_rows if isinstance(data_rows, list) else []:
        if not isinstance(item, dict):
            continue
        year = _norm_text(item.get("year"))
        period = _norm_text(item.get("period"))
        value = _safe_float(item.get("value"))
        if value is None or not year:
            continue
        period_norm = year
        if period and period.upper().startswith("M") and len(period) == 3:
            period_norm = f"{year}-{period[1:]}"
        observations.append(
            {
                "period": period_norm,
                "value": value,
                "note": "; ".join(str(x) for x in item.get("footnotes", []) if x),
            }
        )
    observations = _filter_observations(
        observations,
        start_period=start_period,
        end_period=end_period,
        max_observations=max_observations,
    )
    return {
        "provider": "bls",
        "series_id": series_id,
        "title": _norm_text(first.get("seriesID"), fallback=series_id),
        "description": "",
        "dataset": "BLS Public API",
        "unit": "",
        "frequency": "",
        "source_url": f"https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id}",
        "observations": observations,
    }
