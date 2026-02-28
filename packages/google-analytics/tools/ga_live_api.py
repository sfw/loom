"""Google Analytics live API retrieval tool.

Supports GA4 Data API operations for fetching metadata and report rows
from a specific property using run-scoped Loom auth profiles.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from loom.auth.secrets import SecretResolver
from loom.tools.registry import Tool, ToolContext, ToolResult

_GA_DATA_API_BASE = "https://analyticsdata.googleapis.com/v1beta"
_DEFAULT_DATE_RANGES = [{"startDate": "28daysAgo", "endDate": "yesterday"}]


@dataclass(frozen=True)
class _ResolvedCredentialBundle:
    access_token: str = ""
    api_key: str = ""
    property_id: str = ""
    profile_id: str = ""


class GALiveApiTool(Tool):
    """Fetch GA4 metadata and report data via Google Analytics Data API."""

    @property
    def name(self) -> str:
        return "ga_live_api"

    @property
    def description(self) -> str:
        return (
            "Retrieve live GA4 property data via Analytics Data API. Operations: "
            "run_report, run_realtime_report, get_metadata. Requires "
            "google_analytics auth profile (or explicit access_token/api_key)."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["run_report", "run_realtime_report", "get_metadata"],
                    "description": "GA API operation.",
                },
                "property_id": {
                    "type": "string",
                    "description": (
                        "GA4 property id (digits only). Optional if present in auth "
                        "profile env as GA_PROPERTY_ID / GOOGLE_ANALYTICS_PROPERTY_ID."
                    ),
                },
                "dimensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Dimension names for report operations.",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metric names for report operations.",
                },
                "date_ranges": {
                    "type": "array",
                    "description": (
                        "Date ranges for run_report. Each item accepts "
                        "{startDate,endDate} or {start_date,end_date}."
                    ),
                    "items": {"type": "object"},
                },
                "start_date": {
                    "type": "string",
                    "description": "Shortcut start date for run_report.",
                },
                "end_date": {
                    "type": "string",
                    "description": "Shortcut end date for run_report.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum rows to return (default 1000, max 10000).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Result offset for pagination.",
                },
                "order_bys": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Optional GA API orderBys payload.",
                },
                "dimension_filter": {
                    "type": "object",
                    "description": "Optional GA API dimensionFilter payload.",
                },
                "metric_filter": {
                    "type": "object",
                    "description": "Optional GA API metricFilter payload.",
                },
                "keep_empty_rows": {
                    "type": "boolean",
                    "description": "Whether to retain rows with all metrics zero.",
                },
                "return_property_quota": {
                    "type": "boolean",
                    "description": "Whether to include quota details in API response.",
                },
                "currency_code": {
                    "type": "string",
                    "description": "Optional report currency code (for monetization metrics).",
                },
                "auth_provider": {
                    "type": "string",
                    "description": "Auth provider selector (default google_analytics).",
                },
                "access_token": {
                    "type": "string",
                    "description": "Optional OAuth access token override.",
                },
                "api_key": {
                    "type": "string",
                    "description": "Optional Google API key override.",
                },
                "output_path": {
                    "type": "string",
                    "description": (
                        "Optional output file path. For report operations, .csv writes "
                        "tabular rows; otherwise writes JSON."
                    ),
                },
            },
            "required": ["operation"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 75

    @property
    def auth_requirements(self) -> list[dict[str, Any]]:
        return [
            {
                "provider": "google_analytics",
                "source": "api",
                "modes": ["api_key", "oauth2_pkce", "oauth2_device", "env_passthrough"],
            }
        ]

    @property
    def is_mutating(self) -> bool:
        return True

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        operation = str(args.get("operation", "")).strip().lower()
        if operation not in {"run_report", "run_realtime_report", "get_metadata"}:
            return ToolResult.fail(
                "operation must be run_report/run_realtime_report/get_metadata"
            )

        try:
            creds = _resolve_credentials(args, ctx)
        except ValueError as e:
            return ToolResult.fail(str(e))

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if creds.access_token:
            headers["Authorization"] = f"Bearer {creds.access_token}"
        params: dict[str, str] = {}
        if creds.api_key:
            params["key"] = creds.api_key

        if not headers.get("Authorization") and not params.get("key"):
            return ToolResult.fail(
                "No GA credential resolved. Configure google_analytics auth profile "
                "with a token_ref/secret_ref/env token or pass access_token/api_key."
            )

        try:
            if operation == "get_metadata":
                endpoint = f"/properties/{creds.property_id}/metadata"
                response_payload = await _request_json(
                    method="GET",
                    endpoint=endpoint,
                    headers=headers,
                    params=params,
                    payload=None,
                )
                output_lines = [
                    f"Fetched metadata for property {creds.property_id}.",
                    (
                        f"Dimensions: {len(response_payload.get('dimensions', []))}, "
                        f"metrics: {len(response_payload.get('metrics', []))}."
                    ),
                ]
                table_rows: list[dict[str, Any]] = []
            else:
                endpoint_suffix = (
                    ":runReport"
                    if operation == "run_report"
                    else ":runRealtimeReport"
                )
                endpoint = f"/properties/{creds.property_id}{endpoint_suffix}"
                payload = _build_report_payload(args, realtime=(operation == "run_realtime_report"))
                response_payload = await _request_json(
                    method="POST",
                    endpoint=endpoint,
                    headers=headers,
                    params=params,
                    payload=payload,
                )
                table_rows = _normalize_report_rows(response_payload)
                row_count = len(table_rows)
                output_lines = [
                    f"Fetched {row_count} row(s) from property {creds.property_id}.",
                    f"Operation: {operation}.",
                ]

            files_changed: list[str] = []
            output_path = str(args.get("output_path", "")).strip()
            if output_path:
                if ctx.workspace is None:
                    return ToolResult.fail("No workspace set for output_path")
                target = self._resolve_path(output_path, ctx.workspace)
                _write_output_file(
                    target,
                    payload=response_payload,
                    table_rows=table_rows,
                    operation=operation,
                    changelog=ctx.changelog,
                    subtask_id=ctx.subtask_id,
                )
                rel = str(target.relative_to(ctx.workspace))
                files_changed.append(rel)
                output_lines.append(f"Output written: {rel}")

            result_data = {
                "operation": operation,
                "property_id": creds.property_id,
                "profile_id": creds.profile_id,
                "row_count": len(table_rows),
                "rows": table_rows,
                "response": response_payload,
            }
            return ToolResult.ok(
                "\n".join(output_lines),
                data=result_data,
                files_changed=files_changed,
            )
        except ValueError as e:
            return ToolResult.fail(str(e))
        except Exception as e:
            return ToolResult.fail(f"GA API request failed: {e}")


async def _request_json(
    *,
    method: str,
    endpoint: str,
    headers: dict[str, str],
    params: dict[str, str],
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    url = f"{_GA_DATA_API_BASE}{endpoint}"
    timeout = httpx.Timeout(45.0, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=payload,
        )

    if response.status_code >= 400:
        detail = response.text.strip()[:800]
        if response.status_code in {401, 403}:
            raise ValueError(
                "Google Analytics authentication failed. Check token/API key scope and "
                f"property access. Response {response.status_code}: {detail}"
            )
        raise ValueError(
            f"GA API error {response.status_code}: {detail}"
        )

    try:
        body = response.json()
    except Exception as e:
        raise ValueError(f"GA API returned non-JSON response: {e}") from e

    if not isinstance(body, dict):
        raise ValueError("GA API response must be a JSON object")
    return body


def _build_report_payload(args: dict[str, Any], *, realtime: bool) -> dict[str, Any]:
    metrics = _coerce_name_list(args.get("metrics"))
    if not metrics:
        raise ValueError("metrics is required for run_report/run_realtime_report")

    dimensions = _coerce_name_list(args.get("dimensions"))
    payload: dict[str, Any] = {
        "metrics": [{"name": name} for name in metrics],
    }
    if dimensions:
        payload["dimensions"] = [{"name": name} for name in dimensions]

    if not realtime:
        payload["dateRanges"] = _coerce_date_ranges(args)

    limit = _coerce_int(args.get("limit"), default=1000, minimum=1, maximum=10_000)
    if limit is not None:
        payload["limit"] = str(limit)

    offset = _coerce_int(args.get("offset"), default=0, minimum=0, maximum=1_000_000)
    if offset:
        payload["offset"] = str(offset)

    order_bys = args.get("order_bys")
    if isinstance(order_bys, list) and order_bys:
        payload["orderBys"] = [item for item in order_bys if isinstance(item, dict)]

    dim_filter = args.get("dimension_filter")
    if isinstance(dim_filter, dict) and dim_filter:
        payload["dimensionFilter"] = dim_filter

    metric_filter = args.get("metric_filter")
    if isinstance(metric_filter, dict) and metric_filter:
        payload["metricFilter"] = metric_filter

    keep_empty_rows = args.get("keep_empty_rows")
    if isinstance(keep_empty_rows, bool):
        payload["keepEmptyRows"] = keep_empty_rows

    return_property_quota = args.get("return_property_quota")
    if isinstance(return_property_quota, bool):
        payload["returnPropertyQuota"] = return_property_quota

    currency_code = str(args.get("currency_code", "")).strip()
    if currency_code:
        payload["currencyCode"] = currency_code

    return payload


def _coerce_name_list(raw: object) -> list[str]:
    if isinstance(raw, str):
        raw = [item.strip() for item in raw.split(",")]
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = str(item or "").strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _coerce_date_ranges(args: dict[str, Any]) -> list[dict[str, str]]:
    raw = args.get("date_ranges")
    if isinstance(raw, list) and raw:
        out: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            start = str(item.get("startDate", item.get("start_date", ""))).strip()
            end = str(item.get("endDate", item.get("end_date", ""))).strip()
            if not start or not end:
                continue
            out.append({"startDate": start, "endDate": end})
        if out:
            return out

    start = str(args.get("start_date", "")).strip()
    end = str(args.get("end_date", "")).strip()
    if start and end:
        return [{"startDate": start, "endDate": end}]

    return list(_DEFAULT_DATE_RANGES)


def _coerce_int(
    raw: object,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int | None:
    if raw is None or raw == "":
        return default
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _normalize_report_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    dims = [
        str(item.get("name", "")).strip()
        for item in payload.get("dimensionHeaders", [])
        if isinstance(item, dict)
    ]
    metrics = [
        str(item.get("name", "")).strip()
        for item in payload.get("metricHeaders", [])
        if isinstance(item, dict)
    ]
    headers = [name for name in [*dims, *metrics] if name]

    rows: list[dict[str, Any]] = []
    for row in payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        dim_values = row.get("dimensionValues", [])
        metric_values = row.get("metricValues", [])

        line: dict[str, Any] = {}
        for index, header in enumerate(dims):
            value = ""
            if isinstance(dim_values, list) and index < len(dim_values):
                item = dim_values[index]
                if isinstance(item, dict):
                    value = str(item.get("value", ""))
            line[header] = value

        for index, header in enumerate(metrics):
            value = ""
            if isinstance(metric_values, list) and index < len(metric_values):
                item = metric_values[index]
                if isinstance(item, dict):
                    value = str(item.get("value", ""))
            line[header] = value

        if headers and not line:
            continue
        rows.append(line)
    return rows


def _write_output_file(
    target: Path,
    *,
    payload: dict[str, Any],
    table_rows: list[dict[str, Any]],
    operation: str,
    changelog: Any | None,
    subtask_id: str,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if changelog is not None:
        changelog.record_before_write(str(target), subtask_id=subtask_id)

    write_csv = (
        operation in {"run_report", "run_realtime_report"}
        and target.suffix.lower() == ".csv"
    )
    if write_csv:
        fieldnames: list[str] = []
        for row in table_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
        for row in table_rows:
            writer.writerow(row)
        target.write_text(buffer.getvalue(), encoding="utf-8")
        return

    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_credentials(args: dict[str, Any], ctx: ToolContext) -> _ResolvedCredentialBundle:
    auth_provider = str(args.get("auth_provider", "google_analytics")).strip() or "google_analytics"
    access_token = str(args.get("access_token", "")).strip()
    api_key = str(args.get("api_key", "")).strip()
    property_id = str(args.get("property_id", "")).strip()
    profile_id = ""

    profile = None
    auth_context = ctx.auth_context
    if auth_context is not None and hasattr(auth_context, "profile_for_provider"):
        profile = auth_context.profile_for_provider(auth_provider)

    env_values: dict[str, str] = {}
    if profile is not None:
        profile_id = str(getattr(profile, "profile_id", "") or "")
        env_values = _resolve_profile_env_values(profile, auth_context)

        property_id = property_id or (
            env_values.get("GA_PROPERTY_ID")
            or env_values.get("GOOGLE_ANALYTICS_PROPERTY_ID")
            or env_values.get("GOOGLE_PROPERTY_ID")
            or ""
        )

        if not access_token:
            access_token = (
                env_values.get("GA_ACCESS_TOKEN")
                or env_values.get("GOOGLE_ACCESS_TOKEN")
                or env_values.get("ACCESS_TOKEN")
                or env_values.get("GA_TOKEN")
                or ""
            )
        if not api_key:
            api_key = (
                env_values.get("GA_API_KEY")
                or env_values.get("GOOGLE_API_KEY")
                or env_values.get("API_KEY")
                or ""
            )

        mode = str(getattr(profile, "mode", "") or "").strip().lower()
        secret_candidates: list[str] = []
        token_ref = str(getattr(profile, "token_ref", "") or "").strip()
        secret_ref = str(getattr(profile, "secret_ref", "") or "").strip()
        if token_ref:
            secret_candidates.append(token_ref)
        if secret_ref:
            secret_candidates.append(secret_ref)

        for ref in secret_candidates:
            resolved_secret = _resolve_secret_ref(ref, auth_context)
            token_candidate, key_candidate, property_candidate = _extract_secret_fields(
                resolved_secret,
                default_mode=mode,
            )
            if not access_token and token_candidate:
                access_token = token_candidate
            if not api_key and key_candidate:
                api_key = key_candidate
            if not property_id and property_candidate:
                property_id = property_candidate

    if not property_id:
        raise ValueError(
            "property_id is required. Provide property_id argument or configure "
            "GA_PROPERTY_ID/GOOGLE_ANALYTICS_PROPERTY_ID in auth profile env."
        )

    if not access_token and not api_key:
        raise ValueError(
            "No credential resolved for google_analytics. Configure access token "
            "(token_ref/secret_ref/env) or pass access_token/api_key directly."
        )

    return _ResolvedCredentialBundle(
        access_token=access_token,
        api_key=api_key,
        property_id=property_id,
        profile_id=profile_id,
    )


def _resolve_profile_env_values(profile: Any, auth_context: Any | None) -> dict[str, str]:
    env = getattr(profile, "env", {})
    if not isinstance(env, dict):
        return {}

    resolver = getattr(auth_context, "secret_resolver", None)
    fallback_resolver = SecretResolver()

    resolved: dict[str, str] = {}
    for key, raw_value in env.items():
        name = str(key or "").strip()
        if not name:
            continue
        value = str(raw_value or "").strip()
        if not value:
            continue
        try:
            if resolver is not None and hasattr(resolver, "resolve_maybe"):
                final_value = str(resolver.resolve_maybe(value)).strip()
            else:
                final_value = str(fallback_resolver.resolve_maybe(value)).strip()
        except Exception:
            final_value = value
        if final_value:
            resolved[name] = final_value
    return resolved


def _resolve_secret_ref(secret_ref: str, auth_context: Any | None) -> str:
    clean = str(secret_ref or "").strip()
    if not clean:
        return ""

    if auth_context is not None and hasattr(auth_context, "resolve_secret_ref"):
        return str(auth_context.resolve_secret_ref(clean)).strip()

    resolver = SecretResolver()
    return str(resolver.resolve(clean)).strip()


def _extract_secret_fields(
    raw_secret: str,
    *,
    default_mode: str,
) -> tuple[str, str, str]:
    clean = str(raw_secret or "").strip()
    if not clean:
        return "", "", ""

    try:
        payload = json.loads(clean)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        access_token = str(
            payload.get("access_token")
            or payload.get("token")
            or payload.get("bearer_token")
            or ""
        ).strip()
        api_key = str(payload.get("api_key") or payload.get("key") or "").strip()
        property_id = str(
            payload.get("property_id")
            or payload.get("ga_property_id")
            or ""
        ).strip()
        return access_token, api_key, property_id

    if default_mode == "api_key":
        return "", clean, ""

    # GA Data API typically expects OAuth Bearer token.
    return clean, "", ""
