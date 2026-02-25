"""Keyless market-data API tool for price/return research."""

from __future__ import annotations

import json
from typing import Any

import httpx

from loom.research.finance import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    pct_returns,
    sharpe_ratio,
)
from loom.research.providers import (
    SUPPORTED_MARKET_PROVIDERS,
    MarketDataProviderError,
    fetch_stooq_daily_prices,
)
from loom.research.text import coerce_int
from loom.tools.registry import Tool, ToolContext, ToolResult

_OPERATIONS = {"get_prices", "get_returns", "get_actions"}


class MarketDataApiTool(Tool):
    """Fetch keyless daily market data and derived return metrics."""

    @property
    def name(self) -> str:
        return "market_data_api"

    @property
    def description(self) -> str:
        return "Retrieve keyless market prices and return analytics (provider support: stooq)."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["get_prices", "get_returns", "get_actions"],
                },
                "provider": {
                    "type": "string",
                    "description": "Market data provider (default stooq).",
                },
                "symbol": {
                    "type": "string",
                    "description": "Single ticker symbol.",
                },
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional multi-symbol fetch list.",
                },
                "start_date": {
                    "type": "string",
                    "description": "Optional start date YYYY-MM-DD.",
                },
                "end_date": {
                    "type": "string",
                    "description": "Optional end date YYYY-MM-DD.",
                },
                "risk_free_rate": {
                    "type": "number",
                    "description": "Optional annual risk-free rate for Sharpe.",
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Max rows per symbol in output data (default 300).",
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
            return ToolResult.fail("operation must be get_prices/get_returns/get_actions")

        provider = str(args.get("provider", "stooq")).strip().lower() or "stooq"
        if provider not in SUPPORTED_MARKET_PROVIDERS:
            return ToolResult.fail(f"Unsupported provider '{provider}'")

        symbols = _collect_symbols(args)
        if not symbols:
            return ToolResult.fail("Provide symbol or symbols")

        start_date = str(args.get("start_date", "")).strip()
        end_date = str(args.get("end_date", "")).strip()
        max_rows = _clamp_int(args.get("max_rows"), default=300, lo=10, hi=5_000)
        risk_free_rate = _to_float(args.get("risk_free_rate"), default=0.0)

        async with httpx.AsyncClient() as client:
            series_by_symbol: dict[str, dict[str, Any]] = {}
            provider_errors: dict[str, str] = {}
            for symbol in symbols:
                try:
                    payload = await fetch_stooq_daily_prices(
                        symbol=symbol,
                        start=start_date,
                        end=end_date,
                        client=client,
                    )
                except Exception as e:
                    provider_errors[symbol] = f"{type(e).__name__}: {e}"
                    continue
                rows = payload.get("rows", [])
                if isinstance(rows, list):
                    payload["rows"] = rows[-max_rows:]
                series_by_symbol[symbol.upper()] = payload

        if not series_by_symbol:
            message = "All symbols failed."
            if provider_errors:
                message += " " + "; ".join(f"{k} ({v})" for k, v in sorted(provider_errors.items()))
            return ToolResult.fail(message)

        if operation == "get_prices":
            payload = _build_prices_payload(
                series_by_symbol=series_by_symbol,
                provider=provider,
                provider_errors=provider_errors,
            )
            lines = [
                f"Fetched prices for {len(series_by_symbol)} symbol(s) via {provider}.",
            ]
        elif operation == "get_returns":
            payload = _build_returns_payload(
                series_by_symbol=series_by_symbol,
                provider=provider,
                provider_errors=provider_errors,
                risk_free_rate=risk_free_rate,
            )
            lines = [
                f"Computed returns for {len(series_by_symbol)} symbol(s) via {provider}.",
            ]
        else:
            payload = _build_actions_payload(
                series_by_symbol=series_by_symbol,
                provider=provider,
                provider_errors=provider_errors,
            )
            lines = [
                (
                    f"Inferred split-like actions for {len(series_by_symbol)} "
                    f"symbol(s) via {provider}."
                ),
                "Note: Action detection is heuristic based on large close-to-close jumps.",
            ]

        files_changed: list[str] = []
        output_path = str(args.get("output_path", "")).strip()
        if output_path and ctx.workspace is not None:
            path = self._resolve_path(output_path, ctx.workspace)
            path.parent.mkdir(parents=True, exist_ok=True)
            if ctx.changelog is not None:
                ctx.changelog.record_before_write(str(path), subtask_id=ctx.subtask_id)
            path.write_text(_render_report(payload), encoding="utf-8")
            rel = str(path.relative_to(ctx.workspace))
            files_changed.append(rel)
            lines.append(f"Report written: {rel}")

        return ToolResult.ok(
            "\n".join(lines),
            data=payload,
            files_changed=files_changed,
        )


def _collect_symbols(args: dict[str, Any]) -> list[str]:
    out: list[str] = []
    single = str(args.get("symbol", "")).strip()
    if single:
        out.append(single)
    raw = args.get("symbols", [])
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


def _build_prices_payload(
    *,
    series_by_symbol: dict[str, dict[str, Any]],
    provider: str,
    provider_errors: dict[str, str],
) -> dict[str, Any]:
    items: dict[str, Any] = {}
    for symbol, payload in series_by_symbol.items():
        rows = payload.get("rows", [])
        items[symbol] = {
            "count": len(rows) if isinstance(rows, list) else 0,
            "as_of": payload.get("as_of"),
            "source_url": payload.get("source_url"),
            "rows": rows if isinstance(rows, list) else [],
        }
    return {
        "operation": "get_prices",
        "provider": provider,
        "symbols": sorted(items.keys()),
        "series": items,
        "provider_errors": provider_errors,
        "keyless": True,
    }


def _build_returns_payload(
    *,
    series_by_symbol: dict[str, dict[str, Any]],
    provider: str,
    provider_errors: dict[str, str],
    risk_free_rate: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for symbol, payload in series_by_symbol.items():
        rows = payload.get("rows", [])
        closes = [
            float(row.get("close"))
            for row in rows
            if isinstance(row, dict) and row.get("close") is not None
        ]
        rets = pct_returns(closes)
        mdd, peak_idx, trough_idx = max_drawdown(closes)
        summary[symbol] = {
            "count": len(rets),
            "as_of": payload.get("as_of"),
            "cumulative_return": (closes[-1] / closes[0] - 1.0) if len(closes) >= 2 else 0.0,
            "annualized_return": annualized_return(rets),
            "annualized_volatility": annualized_volatility(rets),
            "sharpe": sharpe_ratio(rets, risk_free_rate=risk_free_rate),
            "max_drawdown": mdd,
            "drawdown_peak_index": peak_idx,
            "drawdown_trough_index": trough_idx,
            "returns": rets,
            "source_url": payload.get("source_url"),
        }
    return {
        "operation": "get_returns",
        "provider": provider,
        "symbols": sorted(summary.keys()),
        "risk_free_rate": risk_free_rate,
        "series": summary,
        "provider_errors": provider_errors,
        "keyless": True,
    }


def _build_actions_payload(
    *,
    series_by_symbol: dict[str, dict[str, Any]],
    provider: str,
    provider_errors: dict[str, str],
) -> dict[str, Any]:
    inferred: dict[str, list[dict[str, Any]]] = {}
    for symbol, payload in series_by_symbol.items():
        rows = payload.get("rows", [])
        actions: list[dict[str, Any]] = []
        prev_close: float | None = None
        for row in rows:
            if not isinstance(row, dict):
                continue
            close = row.get("close")
            if close is None:
                continue
            close_val = float(close)
            if prev_close is not None and prev_close > 0:
                ratio = close_val / prev_close
                # Flag large jumps that resemble split/reverse split events.
                if ratio >= 1.9 or ratio <= 0.55:
                    actions.append(
                        {
                            "date": row.get("date"),
                            "event": "possible_split_or_reverse_split",
                            "close_ratio": ratio,
                            "confidence": 0.45,
                        }
                    )
            prev_close = close_val
        inferred[symbol] = actions

    return {
        "operation": "get_actions",
        "provider": provider,
        "symbols": sorted(inferred.keys()),
        "inferred_actions": inferred,
        "provider_errors": provider_errors,
        "keyless": True,
        "warnings": [
            "Corporate actions are inferred heuristically from price jumps; verify with filings."
        ],
    }


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Market Data API Report",
        "",
        f"- **Operation**: {payload.get('operation', '')}",
        f"- **Provider**: {payload.get('provider', '')}",
        f"- **Symbols**: {', '.join(payload.get('symbols', []))}",
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


__all__ = ["MarketDataApiTool", "MarketDataProviderError"]
