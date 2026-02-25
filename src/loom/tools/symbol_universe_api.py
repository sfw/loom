"""Symbol-universe and ticker mapping tool."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import httpx

from loom.research.providers import (
    SecDataError,
    fetch_sec_ticker_map,
    resolve_ticker_to_cik,
)
from loom.research.text import coerce_int
from loom.tools.registry import Tool, ToolContext, ToolResult

_OPERATIONS = {"resolve_symbol", "map_ticker_cik", "list_symbols"}


class SymbolUniverseApiTool(Tool):
    """Resolve and enumerate investable symbols with SEC mapping support."""

    @property
    def name(self) -> str:
        return "symbol_universe_api"

    @property
    def description(self) -> str:
        return (
            "Resolve symbols, map ticker->CIK, and list universe entries "
            "from SEC mapping or local CSV."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["resolve_symbol", "map_ticker_cik", "list_symbols"],
                },
                "symbol": {
                    "type": "string",
                    "description": "Single symbol/ticker.",
                },
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ticker list for batch mapping.",
                },
                "query": {
                    "type": "string",
                    "description": "Optional substring filter for list_symbols.",
                },
                "source_path": {
                    "type": "string",
                    "description": "Optional local CSV source for list_symbols.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max list entries (default 100).",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional markdown report path.",
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
            return ToolResult.fail("operation must be resolve_symbol/map_ticker_cik/list_symbols")

        limit = _clamp_int(args.get("limit"), default=100, lo=1, hi=10_000)
        query = str(args.get("query", "")).strip().lower()

        if operation == "list_symbols" and str(args.get("source_path", "")).strip():
            if ctx.workspace is None:
                return ToolResult.fail("No workspace set for source_path")
            source_path = self._resolve_path(str(args.get("source_path")), ctx.workspace)
            entries = _load_local_csv(source_path)
            if query:
                entries = [
                    row
                    for row in entries
                    if query in row.get("ticker", "").lower()
                    or query in row.get("name", "").lower()
                ]
            entries = entries[:limit]
            payload = {
                "operation": "list_symbols",
                "source": str(source_path),
                "count": len(entries),
                "entries": entries,
                "keyless": True,
            }
            return _finalize(self, ctx, args=args, payload=payload, headline="Listed symbols")

        async with httpx.AsyncClient() as client:
            if operation == "resolve_symbol":
                symbol = str(args.get("symbol", "")).strip().upper()
                if not symbol:
                    return ToolResult.fail("symbol is required for resolve_symbol")
                try:
                    resolved = await resolve_ticker_to_cik(ticker=symbol, client=client)
                except Exception as e:
                    return ToolResult.fail(f"Resolve failed: {e}")
                payload = {
                    "operation": operation,
                    "symbol": symbol,
                    "resolved": resolved,
                    "keyless": True,
                }
                return _finalize(
                    self,
                    ctx,
                    args=args,
                    payload=payload,
                    headline=f"Resolved {symbol}",
                )

            if operation == "map_ticker_cik":
                tickers = _collect_tickers(args)
                if not tickers:
                    return ToolResult.fail("tickers (or symbol) required for map_ticker_cik")
                mapping = await fetch_sec_ticker_map(client=client)
                rows: list[dict[str, Any]] = []
                missing: list[str] = []
                for ticker in tickers:
                    item = mapping.get(ticker.upper())
                    if item is None:
                        missing.append(ticker.upper())
                        continue
                    rows.append(item)
                payload = {
                    "operation": operation,
                    "count": len(rows),
                    "mapped": rows,
                    "missing": missing,
                    "keyless": True,
                }
                return _finalize(
                    self,
                    ctx,
                    args=args,
                    payload=payload,
                    headline=f"Mapped {len(rows)} ticker(s)",
                )

            try:
                mapping = await fetch_sec_ticker_map(client=client)
            except Exception as e:
                return ToolResult.fail(f"list_symbols failed: {e}")

        entries = sorted(mapping.values(), key=lambda row: row.get("ticker", ""))
        if query:
            entries = [
                row
                for row in entries
                if query in row.get("ticker", "").lower() or query in row.get("name", "").lower()
            ]
        entries = entries[:limit]
        payload = {
            "operation": "list_symbols",
            "source": "sec_company_tickers",
            "count": len(entries),
            "entries": entries,
            "keyless": True,
        }
        return _finalize(self, ctx, args=args, payload=payload, headline="Listed symbols")


def _clamp_int(value: object, *, default: int, lo: int, hi: int) -> int:
    parsed = coerce_int(value, default=default)
    if parsed is None:
        parsed = default
    return max(lo, min(hi, parsed))


def _collect_tickers(args: dict[str, Any]) -> list[str]:
    out: list[str] = []
    single = str(args.get("symbol", "")).strip()
    if single:
        out.append(single)
    raw = args.get("tickers", [])
    if isinstance(raw, list):
        for item in raw:
            text = str(item or "").strip()
            if text:
                out.append(text)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in out:
        key = item.upper()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _load_local_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SecDataError(f"source_path not found: {path}")
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker", "")).strip().upper()
            name = str(row.get("name", "")).strip()
            if not ticker:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "cik": str(row.get("cik", "")).strip(),
                    "name": name,
                    "source": str(path),
                }
            )
    return rows


def _finalize(
    tool: Tool,
    ctx: ToolContext,
    *,
    args: dict[str, Any],
    payload: dict[str, Any],
    headline: str,
) -> ToolResult:
    lines = [headline]
    files_changed: list[str] = []
    output_path = str(args.get("output_path", "")).strip()
    if output_path and ctx.workspace is not None:
        path = tool._resolve_path(output_path, ctx.workspace)
        path.parent.mkdir(parents=True, exist_ok=True)
        if ctx.changelog is not None:
            ctx.changelog.record_before_write(str(path), subtask_id=ctx.subtask_id)
        path.write_text(_render_report(payload), encoding="utf-8")
        rel = str(path.relative_to(ctx.workspace))
        files_changed.append(rel)
        lines.append(f"Report written: {rel}")
    return ToolResult.ok("\n".join(lines), data=payload, files_changed=files_changed)


def _render_report(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Symbol Universe API Report",
            "",
            f"- **Operation**: {payload.get('operation', '')}",
            f"- **Count**: {payload.get('count', 0)}",
            "",
            "```json",
            json.dumps(payload, indent=2),
            "```",
            "",
        ]
    )


__all__ = ["SymbolUniverseApiTool", "SecDataError"]
