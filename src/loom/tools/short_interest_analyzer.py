"""Short-interest and short-sale-volume analysis tool."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import httpx

from loom.research.finance import safe_float
from loom.research.providers import (
    FinraShortDataError,
    fetch_finra_daily_short_volume,
    fetch_finra_short_interest_csv,
    parse_finra_daily_short_volume,
    parse_finra_short_interest_csv,
)
from loom.research.short_signals import compute_short_pressure, detect_squeeze_setup
from loom.research.text import coerce_int
from loom.tools.registry import Tool, ToolContext, ToolResult

_OPERATIONS = {
    "get_short_interest",
    "get_daily_short_volume",
    "compute_short_pressure",
    "detect_squeeze_setup",
}


class ShortInterestAnalyzerTool(Tool):
    """Analyze short-interest positioning and short-sale flow."""

    @property
    def name(self) -> str:
        return "short_interest_analyzer"

    @property
    def description(self) -> str:
        return (
            "Analyze FINRA short-interest and daily short-sale-volume data, compute "
            "short-pressure scores, and detect squeeze setups."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "get_short_interest",
                        "get_daily_short_volume",
                        "compute_short_pressure",
                        "detect_squeeze_setup",
                    ],
                },
                "symbol": {
                    "type": "string",
                    "description": "Optional symbol filter.",
                },
                "date": {
                    "type": "string",
                    "description": "Date token YYYYMMDD or YYYY-MM-DD for FINRA files.",
                },
                "market": {
                    "type": "string",
                    "description": "FINRA market code for daily short volume (default CNMS).",
                },
                "source_path": {
                    "type": "string",
                    "description": "Optional local path for the primary dataset.",
                },
                "source_url": {
                    "type": "string",
                    "description": "Optional source URL override for primary dataset.",
                },
                "daily_source_path": {
                    "type": "string",
                    "description": "Optional local path for daily short volume dataset.",
                },
                "daily_source_url": {
                    "type": "string",
                    "description": "Optional URL for daily short volume dataset.",
                },
                "short_interest_shares": {
                    "type": "number",
                    "description": "Optional inline short interest shares.",
                },
                "float_shares": {
                    "type": "number",
                    "description": "Optional float shares for short-interest ratio.",
                },
                "average_daily_volume": {
                    "type": "number",
                    "description": "Optional ADV for days-to-cover.",
                },
                "short_volume": {
                    "type": "number",
                    "description": "Optional inline short volume.",
                },
                "total_volume": {
                    "type": "number",
                    "description": "Optional inline total volume.",
                },
                "short_pressure_score": {
                    "type": "number",
                    "description": "Optional precomputed short pressure score.",
                },
                "price_momentum_20d": {
                    "type": "number",
                    "description": "Optional 20-day price momentum for squeeze detection.",
                },
                "squeeze_threshold": {
                    "type": "number",
                    "description": "Short-pressure threshold for squeeze setup (default 70).",
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Maximum rows returned (default 500).",
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
        return 75

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        operation = str(args.get("operation", "")).strip().lower()
        if operation not in _OPERATIONS:
            return ToolResult.fail(
                "operation must be get_short_interest/get_daily_short_volume/"
                "compute_short_pressure/detect_squeeze_setup"
            )

        symbol = str(args.get("symbol", "")).strip().upper()
        max_rows = _clamp_int(args.get("max_rows"), default=500, lo=10, hi=20_000)

        try:
            if operation == "get_short_interest":
                payload = await _op_short_interest(
                    self,
                    args,
                    ctx,
                    symbol=symbol,
                    max_rows=max_rows,
                )
            elif operation == "get_daily_short_volume":
                payload = await _op_daily_short_volume(
                    self,
                    args,
                    ctx,
                    symbol=symbol,
                    max_rows=max_rows,
                )
            elif operation == "compute_short_pressure":
                payload = await _op_compute_pressure(self, args, ctx, symbol=symbol)
            else:
                payload = await _op_detect_squeeze(self, args, ctx, symbol=symbol)
        except Exception as e:
            return ToolResult.fail(str(e))

        return _finalize(self, ctx, args=args, payload=payload)


async def _op_short_interest(
    tool: Tool,
    args: dict[str, Any],
    ctx: ToolContext,
    *,
    symbol: str,
    max_rows: int,
) -> dict[str, Any]:
    rows, source = await _load_short_interest_rows(tool, args, ctx)
    if symbol:
        rows = [row for row in rows if str(row.get("symbol", "")).upper() == symbol]
    if len(rows) > max_rows:
        rows = rows[-max_rows:]
    if not rows:
        raise FinraShortDataError("No short-interest rows matched filters")

    return {
        "operation": "get_short_interest",
        "symbol": symbol or None,
        "count": len(rows),
        "rows": rows,
        "as_of": rows[-1].get("date"),
        "sources": [source],
        "warnings": [],
        "confidence": 0.78 if len(rows) >= 5 else 0.52,
        "keyless": True,
    }


async def _op_daily_short_volume(
    tool: Tool,
    args: dict[str, Any],
    ctx: ToolContext,
    *,
    symbol: str,
    max_rows: int,
) -> dict[str, Any]:
    rows, source = await _load_daily_short_rows(tool, args, ctx)
    if symbol:
        rows = [row for row in rows if str(row.get("symbol", "")).upper() == symbol]
    if len(rows) > max_rows:
        rows = rows[-max_rows:]
    if not rows:
        raise FinraShortDataError("No daily short-volume rows matched filters")

    warnings = [
        "Daily short sale volume is flow data and not equivalent to short interest position data."
    ]
    return {
        "operation": "get_daily_short_volume",
        "symbol": symbol or None,
        "count": len(rows),
        "rows": rows,
        "as_of": rows[-1].get("date"),
        "sources": [source],
        "warnings": warnings,
        "confidence": 0.76 if len(rows) >= 5 else 0.5,
        "keyless": True,
    }


async def _op_compute_pressure(
    tool: Tool,
    args: dict[str, Any],
    ctx: ToolContext,
    *,
    symbol: str,
) -> dict[str, Any]:
    warnings: list[str] = []
    source_list: list[str] = []

    short_interest_shares = safe_float(args.get("short_interest_shares"))
    average_daily_volume = safe_float(args.get("average_daily_volume"))
    short_volume = safe_float(args.get("short_volume"))
    total_volume = safe_float(args.get("total_volume"))

    if short_interest_shares is None:
        si_rows, si_source = await _load_short_interest_rows(tool, args, ctx)
        si_row = _latest_symbol_row(si_rows, symbol)
        if si_row is None:
            raise FinraShortDataError("Unable to derive short_interest_shares")
        short_interest_shares = safe_float(si_row.get("short_interest"))
        if average_daily_volume is None:
            average_daily_volume = safe_float(si_row.get("average_daily_volume"))
        source_list.append(si_source)

    if short_volume is None or total_volume is None:
        ds_rows, ds_source = await _load_daily_short_rows(tool, args, ctx)
        ds_row = _latest_symbol_row(ds_rows, symbol)
        if ds_row is not None:
            if short_volume is None:
                short_volume = safe_float(ds_row.get("short_volume"))
            if total_volume is None:
                total_volume = safe_float(ds_row.get("total_volume"))
            source_list.append(ds_source)

    pressure = compute_short_pressure(
        short_interest_shares=short_interest_shares,
        float_shares=args.get("float_shares"),
        average_daily_volume=average_daily_volume,
        short_volume=short_volume,
        total_volume=total_volume,
    )
    warnings.extend(pressure.get("warnings", []))

    return {
        "operation": "compute_short_pressure",
        "symbol": symbol or None,
        "metrics": pressure,
        "sources": source_list,
        "warnings": warnings,
        "as_of": _today_iso(),
        "confidence": pressure.get("confidence", 0.4),
        "keyless": True,
    }


async def _op_detect_squeeze(
    tool: Tool,
    args: dict[str, Any],
    ctx: ToolContext,
    *,
    symbol: str,
) -> dict[str, Any]:
    pressure_score = safe_float(args.get("short_pressure_score"))
    pressure_payload = None
    if pressure_score is None:
        pressure_payload = await _op_compute_pressure(tool, args, ctx, symbol=symbol)
        pressure_score = safe_float(
            pressure_payload.get("metrics", {}).get("short_pressure_score")
        )

    squeeze = detect_squeeze_setup(
        short_pressure_score=pressure_score,
        price_momentum_20d=args.get("price_momentum_20d"),
        threshold=safe_float(args.get("squeeze_threshold")) or 70.0,
    )

    warnings = list(squeeze.get("warnings", []))
    sources: list[str] = []
    if isinstance(pressure_payload, dict):
        warnings.extend(pressure_payload.get("warnings", []))
        sources = list(pressure_payload.get("sources", []))

    return {
        "operation": "detect_squeeze_setup",
        "symbol": symbol or None,
        "setup": squeeze,
        "sources": sources,
        "warnings": warnings,
        "as_of": _today_iso(),
        "confidence": squeeze.get("confidence", 0.4),
        "keyless": True,
    }


async def _load_short_interest_rows(
    tool: Tool,
    args: dict[str, Any],
    ctx: ToolContext,
) -> tuple[list[dict[str, Any]], str]:
    source_path = str(args.get("source_path", "")).strip()
    source_url = str(args.get("source_url", "")).strip()
    date_token = _date_token(args.get("date"))

    if source_path:
        if ctx.workspace is None:
            raise ValueError("No workspace set for source_path")
        path = tool._resolve_read_path(source_path, ctx.workspace, ctx.read_roots)
        text = path.read_text(encoding="utf-8", errors="ignore")
        return parse_finra_short_interest_csv(text, source=str(path)), str(path)

    async with httpx.AsyncClient() as client:
        payload = await fetch_finra_short_interest_csv(
            date_token=date_token,
            source_url=source_url,
            client=client,
        )
    return payload["rows"], str(payload.get("source_url", ""))


async def _load_daily_short_rows(
    tool: Tool,
    args: dict[str, Any],
    ctx: ToolContext,
) -> tuple[list[dict[str, Any]], str]:
    source_path = str(args.get("daily_source_path", "") or args.get("source_path", "")).strip()
    source_url = str(args.get("daily_source_url", "") or args.get("source_url", "")).strip()
    date_token = _date_token(args.get("date"))
    market = str(args.get("market", "CNMS")).strip().upper() or "CNMS"

    if source_path:
        if ctx.workspace is None:
            raise ValueError("No workspace set for daily source path")
        path = tool._resolve_read_path(source_path, ctx.workspace, ctx.read_roots)
        text = path.read_text(encoding="utf-8", errors="ignore")
        return parse_finra_daily_short_volume(text, source=str(path)), str(path)

    async with httpx.AsyncClient() as client:
        payload = await fetch_finra_daily_short_volume(
            date_token=date_token,
            market=market,
            source_url=source_url,
            client=client,
        )
    return payload["rows"], str(payload.get("source_url", ""))


def _latest_symbol_row(rows: list[dict[str, Any]], symbol: str) -> dict[str, Any] | None:
    if not rows:
        return None
    if symbol:
        filtered = [row for row in rows if str(row.get("symbol", "")).upper() == symbol]
        return filtered[-1] if filtered else None
    return rows[-1]


def _today_iso() -> str:
    return date.today().isoformat()


def _date_token(raw: object) -> str:
    text = "".join(ch for ch in str(raw or "") if ch.isdigit())
    if len(text) == 8:
        return text
    return date.today().strftime("%Y%m%d")


def _clamp_int(value: object, *, default: int, lo: int, hi: int) -> int:
    parsed = coerce_int(value, default=default)
    if parsed is None:
        parsed = default
    return max(lo, min(hi, parsed))


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
            "# Short Interest Analyzer Report",
            "",
            f"- **Operation**: {payload.get('operation', '')}",
            "",
            "```json",
            json.dumps(payload, indent=2),
            "```",
            "",
        ]
    )


__all__ = ["ShortInterestAnalyzerTool"]
