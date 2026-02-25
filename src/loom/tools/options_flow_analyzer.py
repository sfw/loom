"""Options flow analysis tool (keyless-first)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from loom.research.options_signals import detect_unusual_options_flow, score_options_flow
from loom.research.providers import (
    CBOE_TOTAL_PUT_CALL_URL,
    OptionsFlowProviderError,
    fetch_cboe_put_call_history,
    fetch_options_flow_csv,
    filter_options_rows,
    parse_options_flow_csv,
    summarize_options_flow,
)
from loom.research.text import coerce_int
from loom.tools.registry import Tool, ToolContext, ToolResult

_OPERATIONS = {
    "get_put_call_history",
    "get_symbol_flow",
    "score_flow",
    "detect_unusual_flow",
}


class OptionsFlowAnalyzerTool(Tool):
    """Analyze put/call history and options-flow anomalies."""

    @property
    def name(self) -> str:
        return "options_flow_analyzer"

    @property
    def description(self) -> str:
        return (
            "Analyze keyless options flow data (Cboe aggregate history or provided "
            "symbol-level CSV), score sentiment, and detect unusual activity."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "get_put_call_history",
                        "get_symbol_flow",
                        "score_flow",
                        "detect_unusual_flow",
                    ],
                },
                "symbol": {
                    "type": "string",
                    "description": "Optional symbol filter for symbol-level operations.",
                },
                "start_date": {
                    "type": "string",
                    "description": "Optional lower date bound YYYY-MM-DD.",
                },
                "end_date": {
                    "type": "string",
                    "description": "Optional upper date bound YYYY-MM-DD.",
                },
                "lookback": {
                    "type": "integer",
                    "description": "Lookback window for scoring/anomaly detection (default 20).",
                },
                "z_threshold": {
                    "type": "number",
                    "description": "Z-score threshold for unusual flow detection (default 2.0).",
                },
                "source_path": {
                    "type": "string",
                    "description": "Optional local CSV path for symbol-level flow.",
                },
                "source_url": {
                    "type": "string",
                    "description": "Optional CSV URL override.",
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Max rows retained in payload (default 500).",
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
        return 60

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        operation = str(args.get("operation", "")).strip().lower()
        if operation not in _OPERATIONS:
            return ToolResult.fail(
                "operation must be get_put_call_history/get_symbol_flow/"
                "score_flow/detect_unusual_flow"
            )

        symbol = str(args.get("symbol", "")).strip().upper()
        start_date = str(args.get("start_date", "")).strip()
        end_date = str(args.get("end_date", "")).strip()
        lookback = _clamp_int(args.get("lookback"), default=20, lo=5, hi=252)
        max_rows = _clamp_int(args.get("max_rows"), default=500, lo=10, hi=10_000)
        z_threshold = _to_float(args.get("z_threshold"), default=2.0)

        source_path = str(args.get("source_path", "")).strip()
        source_url = str(args.get("source_url", "")).strip()

        try:
            rows, sources, warnings = await _load_flow_rows(
                tool=self,
                ctx=ctx,
                operation=operation,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                max_rows=max_rows,
                source_path=source_path,
                source_url=source_url,
            )
        except Exception as e:
            return ToolResult.fail(str(e))

        payload: dict[str, Any]
        if operation == "get_put_call_history":
            summary = summarize_options_flow(rows)
            payload = {
                "operation": operation,
                "symbol": symbol or None,
                "summary": summary,
                "rows": rows,
                "count": len(rows),
                "as_of": summary.get("as_of"),
                "sources": sources,
                "warnings": warnings,
                "confidence": 0.75 if len(rows) >= 20 else 0.5,
                "keyless": True,
            }
        elif operation == "get_symbol_flow":
            summary = summarize_options_flow(rows)
            payload = {
                "operation": operation,
                "symbol": symbol,
                "summary": summary,
                "rows": rows,
                "count": len(rows),
                "as_of": summary.get("as_of"),
                "sources": sources,
                "warnings": warnings,
                "confidence": 0.72 if len(rows) >= 20 else 0.45,
                "keyless": True,
            }
        elif operation == "score_flow":
            score = score_options_flow(rows, lookback=lookback)
            payload = {
                "operation": operation,
                "symbol": symbol or None,
                "score": score,
                "row_count": len(rows),
                "as_of": rows[-1].get("date") if rows else None,
                "sources": sources,
                "warnings": warnings + list(score.get("warnings", [])),
                "confidence": score.get("confidence", 0.5),
                "keyless": True,
            }
        else:
            events = detect_unusual_options_flow(
                rows,
                lookback=lookback,
                z_threshold=z_threshold,
            )
            payload = {
                "operation": operation,
                "symbol": symbol or None,
                "event_count": len(events),
                "events": events,
                "as_of": rows[-1].get("date") if rows else None,
                "sources": sources,
                "warnings": warnings,
                "confidence": 0.78 if events else 0.62,
                "keyless": True,
            }

        return _finalize(self, ctx, args=args, payload=payload)


async def _load_flow_rows(
    *,
    tool: Tool,
    ctx: ToolContext,
    operation: str,
    symbol: str,
    start_date: str,
    end_date: str,
    max_rows: int,
    source_path: str,
    source_url: str,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    warnings: list[str] = []
    sources: list[str] = []

    if source_path:
        if ctx.workspace is None:
            raise ValueError("No workspace set for source_path")
        path = tool._resolve_read_path(source_path, ctx.workspace, ctx.read_roots)
        text = path.read_text(encoding="utf-8", errors="ignore")
        rows = parse_options_flow_csv(text, source=str(path))
        rows = filter_options_rows(
            rows,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            max_rows=max_rows,
        )
        if not rows:
            raise OptionsFlowProviderError("No options rows matched filters")
        sources.append(str(path))
        return rows, sources, warnings

    async with httpx.AsyncClient() as client:
        if source_url:
            rows, resolved = await fetch_options_flow_csv(source_url=source_url, client=client)
            rows = filter_options_rows(
                rows,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                max_rows=max_rows,
            )
            if not rows:
                raise OptionsFlowProviderError("No options rows matched filters")
            sources.append(resolved)
            return rows, sources, warnings

        if operation == "get_symbol_flow" and symbol:
            raise OptionsFlowProviderError(
                "symbol-level flow requires source_path/source_url in keyless mode"
            )

        payload = await fetch_cboe_put_call_history(
            start_date=start_date,
            end_date=end_date,
            max_rows=max_rows,
            source_url=CBOE_TOTAL_PUT_CALL_URL,
            client=client,
        )
        rows = payload.get("rows", [])
        if symbol:
            # Cboe aggregate feed has no symbols; keep behavior explicit.
            warnings.append("Cboe aggregate feed has no per-symbol data; symbol filter ignored.")
        sources.append(str(payload.get("source_url", CBOE_TOTAL_PUT_CALL_URL)))
        return rows, sources, warnings


def _clamp_int(value: object, *, default: int, lo: int, hi: int) -> int:
    parsed = coerce_int(value, default=default)
    if parsed is None:
        parsed = default
    return max(lo, min(hi, parsed))


def _to_float(value: object, *, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _finalize(
    tool: Tool,
    ctx: ToolContext,
    *,
    args: dict[str, Any],
    payload: dict[str, Any],
) -> ToolResult:
    files_changed: list[str] = []
    lines = [f"Computed {payload.get('operation', '')}."]
    output_path = str(args.get("output_path", "")).strip()
    if output_path:
        if ctx.workspace is None:
            return ToolResult.fail("No workspace set for output_path")
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
            "# Options Flow Analyzer Report",
            "",
            f"- **Operation**: {payload.get('operation', '')}",
            "",
            "```json",
            json.dumps(payload, indent=2),
            "```",
            "",
        ]
    )


__all__ = ["OptionsFlowAnalyzerTool"]
