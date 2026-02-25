"""SEC fundamentals extraction tool (keyless)."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import httpx

from loom.research.providers import (
    SecDataError,
    extract_latest_value,
    extract_ttm_value,
    fetch_company_facts,
    resolve_ticker_to_cik,
)
from loom.tools.registry import Tool, ToolContext, ToolResult

_OPERATIONS = {"get_statement", "get_ttm_metrics", "get_quality_flags"}

_INCOME_TAGS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "eps_diluted": ["EarningsPerShareDiluted"],
}

_BALANCE_TAGS = {
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "long_term_debt": ["LongTermDebt"],
}

_CASHFLOW_TAGS = {
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpendituresIncurredButNotYetPaid",
    ],
}


class SecFundamentalsApiTool(Tool):
    """Read normalized statement metrics from SEC companyfacts."""

    @property
    def name(self) -> str:
        return "sec_fundamentals_api"

    @property
    def description(self) -> str:
        return (
            "Fetch SEC filing fundamentals: statement snapshots, TTM metrics, "
            "and quality flags (keyless EDGAR data)."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["get_statement", "get_ttm_metrics", "get_quality_flags"],
                },
                "ticker": {
                    "type": "string",
                    "description": "Ticker symbol (for example AAPL).",
                },
                "cik": {
                    "type": "string",
                    "description": "Optional CIK (if provided, bypass ticker lookup).",
                },
                "taxonomy": {
                    "type": "string",
                    "description": "Taxonomy key (default us-gaap).",
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
                "operation must be get_statement/get_ttm_metrics/get_quality_flags"
            )

        taxonomy = str(args.get("taxonomy", "us-gaap")).strip() or "us-gaap"
        cik = str(args.get("cik", "")).strip()
        ticker = str(args.get("ticker", "")).strip().upper()
        if not cik and not ticker:
            return ToolResult.fail("Provide ticker or cik")

        async with httpx.AsyncClient() as client:
            resolved_ticker = ticker
            if not cik:
                try:
                    resolved = await resolve_ticker_to_cik(ticker=ticker, client=client)
                except Exception as e:
                    return ToolResult.fail(f"ticker lookup failed: {e}")
                cik = str(resolved.get("cik", "")).strip()
                resolved_ticker = str(resolved.get("ticker", ticker)).strip().upper()

            if not cik:
                return ToolResult.fail("Unable to resolve cik")

            try:
                facts = await fetch_company_facts(cik=cik, client=client)
            except Exception as e:
                return ToolResult.fail(f"companyfacts fetch failed: {e}")

        identity = {
            "ticker": resolved_ticker or ticker,
            "cik": cik,
            "entity_name": str(facts.get("entityName", "")).strip(),
        }

        if operation == "get_statement":
            payload = _statement_payload(facts, taxonomy=taxonomy, identity=identity)
            headline = f"Fetched statement snapshot for {identity['ticker'] or identity['cik']}"
        elif operation == "get_ttm_metrics":
            payload = _ttm_payload(facts, taxonomy=taxonomy, identity=identity)
            headline = f"Computed TTM metrics for {identity['ticker'] or identity['cik']}"
        else:
            payload = _quality_payload(facts, taxonomy=taxonomy, identity=identity)
            headline = f"Computed data quality flags for {identity['ticker'] or identity['cik']}"

        files_changed: list[str] = []
        output_path = str(args.get("output_path", "")).strip()
        if output_path:
            if ctx.workspace is None:
                return ToolResult.fail("No workspace set for output_path")
            path = self._resolve_path(output_path, ctx.workspace)
            path.parent.mkdir(parents=True, exist_ok=True)
            if ctx.changelog is not None:
                ctx.changelog.record_before_write(str(path), subtask_id=ctx.subtask_id)
            path.write_text(_render_report(payload), encoding="utf-8")
            rel = str(path.relative_to(ctx.workspace))
            files_changed.append(rel)
            headline += f"\nReport written: {rel}"

        return ToolResult.ok(
            headline,
            data=payload,
            files_changed=files_changed,
        )


def _pick_latest(facts: dict[str, Any], tags: list[str], taxonomy: str) -> dict[str, Any] | None:
    for tag in tags:
        row = extract_latest_value(facts, tag=tag, taxonomy=taxonomy)
        if row is not None:
            return row
    return None


def _pick_ttm(facts: dict[str, Any], tags: list[str], taxonomy: str) -> dict[str, Any] | None:
    for tag in tags:
        row = extract_ttm_value(facts, tag=tag, taxonomy=taxonomy)
        if row is not None:
            return row
    return None


def _metric_to_value(metric: dict[str, Any] | None) -> float | None:
    if metric is None:
        return None
    try:
        return float(metric.get("value"))
    except (TypeError, ValueError):
        return None


def _statement_payload(
    facts: dict[str, Any],
    *,
    taxonomy: str,
    identity: dict[str, Any],
) -> dict[str, Any]:
    income = {key: _pick_latest(facts, tags, taxonomy) for key, tags in _INCOME_TAGS.items()}
    balance = {key: _pick_latest(facts, tags, taxonomy) for key, tags in _BALANCE_TAGS.items()}
    cashflow = {key: _pick_latest(facts, tags, taxonomy) for key, tags in _CASHFLOW_TAGS.items()}
    return {
        "operation": "get_statement",
        "identity": identity,
        "taxonomy": taxonomy,
        "statement": {
            "income_statement": income,
            "balance_sheet": balance,
            "cash_flow": cashflow,
        },
        "sources": [f"https://data.sec.gov/api/xbrl/companyfacts/CIK{identity['cik']}.json"],
        "keyless": True,
        "as_of": _latest_end([income, balance, cashflow]),
    }


def _ttm_payload(
    facts: dict[str, Any],
    *,
    taxonomy: str,
    identity: dict[str, Any],
) -> dict[str, Any]:
    revenue = _pick_ttm(facts, _INCOME_TAGS["revenue"], taxonomy)
    net_income = _pick_ttm(facts, _INCOME_TAGS["net_income"], taxonomy)
    ocf = _pick_ttm(facts, _CASHFLOW_TAGS["operating_cash_flow"], taxonomy)
    capex = _pick_ttm(facts, _CASHFLOW_TAGS["capex"], taxonomy)

    revenue_v = _metric_to_value(revenue)
    net_income_v = _metric_to_value(net_income)
    ocf_v = _metric_to_value(ocf)
    capex_v = _metric_to_value(capex)
    fcf_v = (ocf_v - abs(capex_v)) if ocf_v is not None and capex_v is not None else None
    margin_v = (net_income_v / revenue_v) if revenue_v and net_income_v is not None else None

    metrics = {
        "revenue_ttm": revenue_v,
        "net_income_ttm": net_income_v,
        "operating_cash_flow_ttm": ocf_v,
        "capex_ttm_abs": abs(capex_v) if capex_v is not None else None,
        "free_cash_flow_ttm": fcf_v,
        "net_margin_ttm": margin_v,
    }
    return {
        "operation": "get_ttm_metrics",
        "identity": identity,
        "taxonomy": taxonomy,
        "metrics": metrics,
        "components": {
            "revenue": revenue,
            "net_income": net_income,
            "operating_cash_flow": ocf,
            "capex": capex,
        },
        "sources": [f"https://data.sec.gov/api/xbrl/companyfacts/CIK{identity['cik']}.json"],
        "keyless": True,
        "as_of": _latest_end(
            [{"revenue": revenue, "net_income": net_income, "ocf": ocf, "capex": capex}]
        ),
    }


def _quality_payload(
    facts: dict[str, Any],
    *,
    taxonomy: str,
    identity: dict[str, Any],
) -> dict[str, Any]:
    statement = _statement_payload(facts, taxonomy=taxonomy, identity=identity)
    ttm = _ttm_payload(facts, taxonomy=taxonomy, identity=identity)

    missing: list[str] = []
    for section_name, section in statement["statement"].items():
        for key, value in section.items():
            if value is None:
                missing.append(f"{section_name}.{key}")

    stale = False
    as_of_text = statement.get("as_of")
    if isinstance(as_of_text, str) and as_of_text:
        try:
            age_days = (date.today() - date.fromisoformat(as_of_text)).days
            stale = age_days > 540
        except ValueError:
            stale = True
    else:
        stale = True

    equity_v = _metric_to_value(statement["statement"]["balance_sheet"].get("equity"))
    liabilities_v = _metric_to_value(statement["statement"]["balance_sheet"].get("liabilities"))
    assets_v = _metric_to_value(statement["statement"]["balance_sheet"].get("assets"))
    leverage = None
    if liabilities_v is not None and assets_v not in (None, 0):
        leverage = liabilities_v / assets_v

    quality_flags: list[str] = []
    if missing:
        quality_flags.append("missing_core_fields")
    if stale:
        quality_flags.append("stale_filing_data")
    if equity_v is not None and equity_v < 0:
        quality_flags.append("negative_equity")
    if leverage is not None and leverage > 0.85:
        quality_flags.append("high_liability_ratio")

    confidence = 1.0
    confidence -= min(0.45, 0.02 * len(missing))
    if stale:
        confidence -= 0.2
    confidence = max(0.05, confidence)

    return {
        "operation": "get_quality_flags",
        "identity": identity,
        "taxonomy": taxonomy,
        "quality_flags": quality_flags,
        "missing_fields": missing,
        "stale_data": stale,
        "leverage_ratio": leverage,
        "ttm_metrics": ttm.get("metrics", {}),
        "confidence": confidence,
        "sources": [f"https://data.sec.gov/api/xbrl/companyfacts/CIK{identity['cik']}.json"],
        "keyless": True,
        "as_of": statement.get("as_of"),
    }


def _latest_end(sections: list[dict[str, Any]]) -> str:
    latest = ""
    for section in sections:
        for item in section.values():
            if isinstance(item, dict):
                end_text = str(item.get("end", "")).strip()
                if end_text and end_text > latest:
                    latest = end_text
    return latest


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# SEC Fundamentals API Report",
        "",
        f"- **Operation**: {payload.get('operation', '')}",
        (
            f"- **Entity**: {payload.get('identity', {}).get('ticker', '')} / "
            f"{payload.get('identity', {}).get('cik', '')}"
        ),
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)


__all__ = ["SecFundamentalsApiTool", "SecDataError"]
