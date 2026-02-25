"""Insider trading tracker based on keyless SEC filings."""

from __future__ import annotations

import json
from typing import Any

import httpx

from loom.research.insider_signals import (
    detect_cluster_buys,
)
from loom.research.insider_signals import (
    summarize_insider_activity as summarize_insider_signal,
)
from loom.research.providers import (
    SecDataError,
    SecInsiderDataError,
    extract_recent_form345_filings,
    fetch_sec_filing_transactions,
    fetch_sec_submissions,
    parse_form345_transactions,
    resolve_ticker_to_cik,
)
from loom.research.text import coerce_int
from loom.tools.registry import Tool, ToolContext, ToolResult

_OPERATIONS = {
    "get_recent_filings",
    "get_transactions",
    "summarize_insider_activity",
    "detect_cluster_buys",
}


class InsiderTradingTrackerTool(Tool):
    """Track and summarize insider filing activity (Forms 3/4/5)."""

    @property
    def name(self) -> str:
        return "insider_trading_tracker"

    @property
    def description(self) -> str:
        return (
            "Track insider transactions from SEC Forms 3/4/5, summarize net buy/sell "
            "activity, and detect cluster buys."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "get_recent_filings",
                        "get_transactions",
                        "summarize_insider_activity",
                        "detect_cluster_buys",
                    ],
                },
                "ticker": {
                    "type": "string",
                    "description": "Ticker symbol (for example AAPL).",
                },
                "cik": {
                    "type": "string",
                    "description": "Optional CIK (if provided, bypass ticker lookup).",
                },
                "max_filings": {
                    "type": "integer",
                    "description": "Maximum filings to read (default 20).",
                },
                "start_date": {
                    "type": "string",
                    "description": "Optional filing lower date bound YYYY-MM-DD.",
                },
                "filing_url": {
                    "type": "string",
                    "description": "Optional direct SEC filing URL to parse.",
                },
                "source_path": {
                    "type": "string",
                    "description": "Optional local Form 4 XML path.",
                },
                "window_days": {
                    "type": "integer",
                    "description": "Cluster window size in days (default 30).",
                },
                "min_insiders": {
                    "type": "integer",
                    "description": "Minimum distinct insiders for cluster detection (default 3).",
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
        return 90

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        operation = str(args.get("operation", "")).strip().lower()
        if operation not in _OPERATIONS:
            return ToolResult.fail(
                "operation must be get_recent_filings/get_transactions/"
                "summarize_insider_activity/detect_cluster_buys"
            )

        max_filings = _clamp_int(args.get("max_filings"), default=20, lo=1, hi=200)
        window_days = _clamp_int(args.get("window_days"), default=30, lo=3, hi=120)
        min_insiders = _clamp_int(args.get("min_insiders"), default=3, lo=2, hi=15)
        start_date = str(args.get("start_date", "")).strip()

        try:
            payload = await _execute_operation(
                tool=self,
                args=args,
                ctx=ctx,
                operation=operation,
                max_filings=max_filings,
                start_date=start_date,
                window_days=window_days,
                min_insiders=min_insiders,
            )
        except Exception as e:
            return ToolResult.fail(str(e))

        return _finalize(self, ctx, args=args, payload=payload)


async def _execute_operation(
    *,
    tool: Tool,
    args: dict[str, Any],
    ctx: ToolContext,
    operation: str,
    max_filings: int,
    start_date: str,
    window_days: int,
    min_insiders: int,
) -> dict[str, Any]:
    ticker = str(args.get("ticker", "")).strip().upper()
    cik = str(args.get("cik", "")).strip()
    source_path = str(args.get("source_path", "")).strip()
    filing_url = str(args.get("filing_url", "")).strip()

    if source_path:
        if ctx.workspace is None:
            raise ValueError("No workspace set for source_path")
        path = tool._resolve_read_path(source_path, ctx.workspace, ctx.read_roots)
        text = path.read_text(encoding="utf-8", errors="ignore")
        parsed = parse_form345_transactions(text)
        transactions = parsed.get("transactions", [])
        summary = summarize_insider_signal(transactions)
        payload = {
            "operation": operation,
            "identity": {
                "ticker": ticker or parsed.get("issuer", {}).get("symbol", ""),
                "cik": cik or parsed.get("issuer", {}).get("cik", ""),
                "issuer_name": parsed.get("issuer", {}).get("name", ""),
            },
            "source": str(path),
            "transactions": transactions,
            "summary": summary,
            "sources": [str(path)],
            "warnings": [],
            "as_of": parsed.get("period_of_report"),
            "keyless": True,
        }
        if operation == "get_transactions":
            return payload
        if operation == "summarize_insider_activity":
            return {
                **payload,
                "transactions": [],
                "confidence": summary.get("confidence", 0.3),
            }
        if operation == "detect_cluster_buys":
            clusters = detect_cluster_buys(
                transactions,
                window_days=window_days,
                min_insiders=min_insiders,
            )
            return {
                **payload,
                "clusters": clusters,
                "cluster_count": len(clusters),
                "transactions": [],
                "confidence": 0.72 if clusters else 0.55,
            }
        return {
            "operation": "get_recent_filings",
            "identity": payload["identity"],
            "filings": [],
            "count": 0,
            "sources": [str(path)],
            "warnings": ["source_path mode does not provide submissions listing."],
            "as_of": payload.get("as_of"),
            "confidence": 0.4,
            "keyless": True,
        }

    async with httpx.AsyncClient() as client:
        identity = await _resolve_identity(cik=cik, ticker=ticker, client=client)
        submissions = await fetch_sec_submissions(cik=identity["cik"], client=client)
        filings = extract_recent_form345_filings(
            submissions,
            max_filings=max_filings,
            start_date=start_date,
        )

        if operation == "get_recent_filings":
            return {
                "operation": operation,
                "identity": identity,
                "filings": filings,
                "count": len(filings),
                "sources": [f"https://data.sec.gov/submissions/CIK{identity['cik']}.json"],
                "warnings": [],
                "as_of": filings[0].get("filing_date") if filings else None,
                "confidence": 0.8 if filings else 0.4,
                "keyless": True,
            }

        transactions, warnings = await _collect_transactions(
            client=client,
            filings=filings,
            filing_url=filing_url,
        )
        summary = summarize_insider_signal(transactions)
        payload: dict[str, Any] = {
            "operation": operation,
            "identity": identity,
            "summary": summary,
            "sources": [
                f"https://data.sec.gov/submissions/CIK{identity['cik']}.json",
                "https://www.sec.gov/Archives/edgar/data/",
            ],
            "warnings": warnings + list(summary.get("warnings", [])),
            "as_of": filings[0].get("filing_date") if filings else None,
            "keyless": True,
        }

        if operation == "get_transactions":
            payload.update(
                {
                    "transactions": transactions,
                    "count": len(transactions),
                    "confidence": summary.get("confidence", 0.4),
                }
            )
            return payload

        if operation == "summarize_insider_activity":
            payload.update(
                {
                    "count": len(transactions),
                    "transactions": [],
                    "confidence": summary.get("confidence", 0.4),
                }
            )
            return payload

        clusters = detect_cluster_buys(
            transactions,
            window_days=window_days,
            min_insiders=min_insiders,
        )
        payload.update(
            {
                "clusters": clusters,
                "cluster_count": len(clusters),
                "count": len(transactions),
                "transactions": [],
                "confidence": 0.75 if clusters else 0.55,
            }
        )
        return payload


async def _resolve_identity(
    *,
    cik: str,
    ticker: str,
    client: httpx.AsyncClient,
) -> dict[str, str]:
    clean_cik = "".join(ch for ch in cik if ch.isdigit()).zfill(10) if cik else ""
    if clean_cik:
        return {"ticker": ticker, "cik": clean_cik, "issuer_name": ""}
    if not ticker:
        raise ValueError("Provide ticker or cik")
    try:
        resolved = await resolve_ticker_to_cik(ticker=ticker, client=client)
    except SecDataError as e:
        raise ValueError(f"ticker lookup failed: {e}") from e
    return {
        "ticker": str(resolved.get("ticker", ticker)).strip().upper(),
        "cik": str(resolved.get("cik", "")).strip(),
        "issuer_name": str(resolved.get("name", "")).strip(),
    }


async def _collect_transactions(
    *,
    client: httpx.AsyncClient,
    filings: list[dict[str, Any]],
    filing_url: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    transactions: list[dict[str, Any]] = []

    targets = filings
    if filing_url:
        targets = [{"filing_url": filing_url, "filing_date": "", "form": "4"}]

    for filing in targets:
        url = str(filing.get("filing_url", "")).strip()
        if not url:
            continue
        try:
            parsed = await fetch_sec_filing_transactions(filing_url=url, client=client)
        except (SecInsiderDataError, httpx.HTTPError) as e:
            warnings.append(f"Failed to parse filing {url}: {e}")
            continue

        for row in parsed.get("transactions", []):
            if not isinstance(row, dict):
                continue
            out = dict(row)
            out["filing_date"] = str(filing.get("filing_date", "")).strip()
            out["form"] = str(filing.get("form", "")).strip()
            out["filing_url"] = str(parsed.get("source_url", url))
            transactions.append(out)

    return transactions, warnings


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
            "# Insider Trading Tracker Report",
            "",
            f"- **Operation**: {payload.get('operation', '')}",
            "",
            "```json",
            json.dumps(payload, indent=2),
            "```",
            "",
        ]
    )


__all__ = ["InsiderTradingTrackerTool"]
