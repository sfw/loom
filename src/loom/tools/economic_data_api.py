"""Unified keyless economic-data API tool."""

from __future__ import annotations

import json
from typing import Any

import httpx

from loom.research.providers import (
    SUPPORTED_ECONOMIC_PROVIDERS,
    EconomicProviderError,
    economic_get_observations,
    economic_get_series,
    economic_search,
)
from loom.research.text import coerce_int
from loom.tools.registry import Tool, ToolContext, ToolResult

_OPERATIONS = {"search", "get_series", "get_observations"}


class EconomicDataApiTool(Tool):
    """Query keyless public macroeconomic data providers via one contract."""

    @property
    def name(self) -> str:
        return "economic_data_api"

    @property
    def description(self) -> str:
        return (
            "Query keyless public economic datasets (World Bank, OECD, Eurostat, "
            "DBnomics, BLS, FRED) with normalized outputs."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["search", "get_series", "get_observations"],
                    "description": "Requested economic data operation.",
                },
                "provider": {
                    "type": "string",
                    "description": (
                        "Provider name: world_bank, oecd, eurostat, dbnomics, bls, fred."
                    ),
                },
                "providers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional provider list for search/fallback.",
                },
                "query": {
                    "type": "string",
                    "description": "Search text for operation=search.",
                },
                "series_id": {
                    "type": "string",
                    "description": "Series identifier for series/observation ops.",
                },
                "start_period": {
                    "type": "string",
                    "description": "Optional lower period bound (for example 2000).",
                },
                "end_period": {
                    "type": "string",
                    "description": "Optional upper period bound (for example 2024).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Search result cap (default 15).",
                },
                "max_observations": {
                    "type": "integer",
                    "description": "Observation cap (default 240).",
                },
                "filters": {
                    "type": "object",
                    "description": "Provider-specific filter map.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional path to write a markdown report.",
                },
            },
            "required": ["operation"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 45

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        operation = str(args.get("operation", "")).strip().lower()
        if operation not in _OPERATIONS:
            return ToolResult.fail("operation must be search/get_series/get_observations")

        providers = _normalize_providers(args.get("providers"), args.get("provider"))
        if providers is None:
            return ToolResult.fail(
                "provider/providers must contain only "
                "world_bank/oecd/eurostat/dbnomics/bls/fred"
            )
        if not providers:
            providers = ["world_bank"]

        filters = args.get("filters", {})
        if filters is None:
            filters = {}
        if not isinstance(filters, dict):
            return ToolResult.fail("filters must be an object")

        start_period = str(args.get("start_period", "")).strip()
        end_period = str(args.get("end_period", "")).strip()

        max_results = _clamp_int(args.get("max_results"), default=15, lo=1, hi=100)
        max_observations = _clamp_int(
            args.get("max_observations"),
            default=240,
            lo=1,
            hi=2_000,
        )

        async with httpx.AsyncClient() as client:
            if operation == "search":
                query = str(args.get("query", "")).strip()
                if not query:
                    return ToolResult.fail("query is required for operation=search")
                return await self._execute_search(
                    query=query,
                    providers=providers,
                    max_results=max_results,
                    output_path=str(args.get("output_path", "")).strip(),
                    ctx=ctx,
                    client=client,
                )

            series_id = str(args.get("series_id", "")).strip()
            if not series_id:
                return ToolResult.fail("series_id is required for series/observation operations")
            return await self._execute_series_operation(
                operation=operation,
                series_id=series_id,
                providers=providers,
                start_period=start_period,
                end_period=end_period,
                max_observations=max_observations,
                filters=filters,
                output_path=str(args.get("output_path", "")).strip(),
                ctx=ctx,
                client=client,
            )

    async def _execute_search(
        self,
        *,
        query: str,
        providers: list[str],
        max_results: int,
        output_path: str,
        ctx: ToolContext,
        client: httpx.AsyncClient,
    ) -> ToolResult:
        rows: list[dict[str, Any]] = []
        provider_errors: dict[str, str] = {}
        per_provider_limit = max(1, min(max_results, 30))

        for provider in providers:
            try:
                found = await economic_search(
                    provider=provider,
                    query=query,
                    max_results=per_provider_limit,
                    client=client,
                )
            except Exception as e:
                provider_errors[provider] = f"{type(e).__name__}: {e}"
                continue
            rows.extend(found)

        rows = _dedupe_rows(rows, cap=max_results)
        lines = [f"Economic search '{query}' -> {len(rows)} result(s)."]
        for idx, row in enumerate(rows, start=1):
            title = str(row.get("title", "")).strip() or str(row.get("series_id", ""))
            lines.append(f"{idx}. {title}")
            lines.append(
                "   "
                + " | ".join(
                    part
                    for part in [
                        str(row.get("provider", "")),
                        str(row.get("series_id", "")),
                        str(row.get("dataset", "")),
                    ]
                    if part
                )
            )
            src = str(row.get("source_url", "")).strip()
            if src:
                lines.append(f"   Source: {src}")

        payload = {
            "operation": "search",
            "query": query,
            "providers": providers,
            "count": len(rows),
            "results": rows,
            "provider_errors": provider_errors,
            "keyless": True,
        }
        files_changed = _write_optional_report(
            tool=self,
            payload=payload,
            output_path=output_path,
            ctx=ctx,
            default_name="economic-data-search.md",
        )
        if files_changed:
            lines.append("Report written: " + ", ".join(files_changed))
        if provider_errors:
            lines.append(
                "Provider warnings: "
                + "; ".join(f"{name} ({msg})" for name, msg in sorted(provider_errors.items()))
            )
        return ToolResult.ok(
            "\n".join(lines),
            data=payload,
            files_changed=files_changed,
        )

    async def _execute_series_operation(
        self,
        *,
        operation: str,
        series_id: str,
        providers: list[str],
        start_period: str,
        end_period: str,
        max_observations: int,
        filters: dict[str, Any],
        output_path: str,
        ctx: ToolContext,
        client: httpx.AsyncClient,
    ) -> ToolResult:
        provider_errors: dict[str, str] = {}
        selected_payload: dict[str, Any] | None = None
        selected_provider = ""

        for provider in providers:
            try:
                if operation == "get_series":
                    payload = await economic_get_series(
                        provider=provider,
                        series_id=series_id,
                        start_period=start_period,
                        end_period=end_period,
                        max_observations=max_observations,
                        filters=filters,
                        client=client,
                    )
                else:
                    payload = await economic_get_observations(
                        provider=provider,
                        series_id=series_id,
                        start_period=start_period,
                        end_period=end_period,
                        max_observations=max_observations,
                        filters=filters,
                        client=client,
                    )
            except Exception as e:
                provider_errors[provider] = f"{type(e).__name__}: {e}"
                continue
            selected_payload = payload
            selected_provider = provider
            break

        if selected_payload is None:
            message = "All requested providers failed."
            if provider_errors:
                message += " " + "; ".join(
                    f"{name} ({msg})" for name, msg in sorted(provider_errors.items())
                )
            return ToolResult.fail(message)

        selected_payload["operation"] = operation
        selected_payload["provider_attempts"] = list(providers)
        selected_payload["provider_errors"] = provider_errors
        selected_payload["keyless"] = True

        lines = [
            f"{operation} succeeded via {selected_provider}",
            f"Series: {selected_payload.get('series_id', series_id)}",
            f"Title: {selected_payload.get('title', '')}",
        ]
        coverage = selected_payload.get("coverage") or {}
        if isinstance(coverage, dict):
            lines.append(
                "Coverage: "
                + ", ".join(
                    [
                        f"count={coverage.get('count', 0)}",
                        f"start={coverage.get('start', '')}",
                        f"end={coverage.get('end', '')}",
                    ]
                )
            )

        files_changed = _write_optional_report(
            tool=self,
            payload=selected_payload,
            output_path=output_path,
            ctx=ctx,
            default_name=f"economic-data-{operation}.md",
        )
        if files_changed:
            lines.append("Report written: " + ", ".join(files_changed))
        if provider_errors:
            lines.append(
                "Fallback notes: "
                + "; ".join(f"{name} ({msg})" for name, msg in sorted(provider_errors.items()))
            )

        return ToolResult.ok(
            "\n".join(lines),
            data=selected_payload,
            files_changed=files_changed,
        )


def _normalize_providers(raw: object, single: object) -> list[str] | None:
    values: list[str] = []
    if isinstance(raw, list):
        values.extend(str(item or "").strip().lower() for item in raw)
    elif raw is not None and raw != "":
        values.append(str(raw or "").strip().lower())

    single_text = str(single or "").strip().lower()
    if single_text:
        values.append(single_text)

    out: list[str] = []
    for provider in values:
        if not provider:
            continue
        if provider not in SUPPORTED_ECONOMIC_PROVIDERS:
            return None
        if provider not in out:
            out.append(provider)
    return out


def _clamp_int(value: object, *, default: int, lo: int, hi: int) -> int:
    parsed = coerce_int(value, default=default)
    if parsed is None:
        parsed = default
    return max(lo, min(hi, parsed))


def _dedupe_rows(rows: list[dict[str, Any]], *, cap: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = f"{row.get('provider','')}::{row.get('series_id','')}".lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= cap:
            break
    return out


def _write_optional_report(
    *,
    tool: Tool,
    payload: dict[str, Any],
    output_path: str,
    ctx: ToolContext,
    default_name: str,
) -> list[str]:
    if ctx.workspace is None:
        return []

    target = output_path or default_name
    path = tool._resolve_path(target, ctx.workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    if ctx.changelog is not None:
        ctx.changelog.record_before_write(str(path), subtask_id=ctx.subtask_id)

    lines = [
        "# Economic Data Report",
        "",
        f"- **Operation**: {payload.get('operation', '')}",
        f"- **Provider**: {payload.get('provider', payload.get('providers', ''))}",
        f"- **Series**: {payload.get('series_id', '')}",
        f"- **Count**: {payload.get('count', payload.get('coverage', {}).get('count', 0))}",
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return [str(path.relative_to(ctx.workspace))]


__all__ = ["EconomicDataApiTool", "EconomicProviderError"]
