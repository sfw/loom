"""Earnings surprise prediction tool (keyless-first, model-based)."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import httpx

from loom.research.earnings_features import build_earnings_features
from loom.research.earnings_models import (
    backtest_earnings_model,
    compare_prediction_to_consensus,
    predict_earnings_surprise,
)
from loom.research.providers import (
    SecDataError,
    extract_latest_value,
    extract_ttm_value,
    fetch_company_facts,
    resolve_ticker_to_cik,
)
from loom.tools.registry import Tool, ToolContext, ToolResult

_OPERATIONS = {
    "build_features",
    "predict_surprise",
    "backtest_model",
    "compare_to_consensus",
}


class EarningsSurprisePredictorTool(Tool):
    """Predict and evaluate earnings surprises using keyless signals."""

    @property
    def name(self) -> str:
        return "earnings_surprise_predictor"

    @property
    def description(self) -> str:
        return (
            "Build earnings feature sets, predict surprise outcomes, backtest a deterministic "
            "model, and compare model output to user-provided consensus estimates."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "build_features",
                        "predict_surprise",
                        "backtest_model",
                        "compare_to_consensus",
                    ],
                },
                "ticker": {
                    "type": "string",
                    "description": "Optional ticker for SEC feature augmentation.",
                },
                "cik": {
                    "type": "string",
                    "description": "Optional CIK for SEC feature augmentation.",
                },
                "latest_eps": {"type": "number"},
                "prior_eps": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Historical EPS observations ordered oldest to newest.",
                },
                "revenue_ttm": {"type": "number"},
                "net_margin_ttm": {"type": "number"},
                "price_returns": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Recent periodic returns for momentum/volatility features.",
                },
                "sentiment_score": {"type": "number"},
                "options_flow_score": {"type": "number"},
                "short_pressure_score": {"type": "number"},
                "insider_score": {"type": "number"},
                "guidance_delta": {"type": "number"},
                "history": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Backtest rows with features and actual_surprise_pct.",
                },
                "consensus_eps": {
                    "type": "number",
                    "description": "Required for compare_to_consensus.",
                },
                "actual_eps": {
                    "type": "number",
                    "description": "Optional realized EPS for ex-post accuracy comparison.",
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
                "operation must be build_features/predict_surprise/backtest_model/"
                "compare_to_consensus"
            )

        try:
            payload = await _execute_operation(operation=operation, args=args)
        except Exception as e:
            return ToolResult.fail(str(e))

        return _finalize(self, ctx, args=args, payload=payload)


async def _execute_operation(*, operation: str, args: dict[str, Any]) -> dict[str, Any]:
    if operation == "backtest_model":
        history = args.get("history", [])
        if not isinstance(history, list):
            raise ValueError("history must be an array of objects")
        stats = backtest_earnings_model([row for row in history if isinstance(row, dict)])
        return {
            "operation": operation,
            "backtest": stats,
            "sources": [],
            "warnings": [],
            "as_of": _today_iso(),
            "confidence": stats.get("confidence", 0.3),
            "keyless": True,
        }

    features_payload, identity, sources, warnings = await _build_feature_payload(args)

    if operation == "build_features":
        return {
            "operation": operation,
            "identity": identity,
            "features": features_payload,
            "sources": sources,
            "warnings": warnings,
            "as_of": _today_iso(),
            "confidence": max(0.25, features_payload.get("feature_coverage", 0.0)),
            "keyless": True,
        }

    prediction = predict_earnings_surprise(features_payload)
    if operation == "predict_surprise":
        return {
            "operation": operation,
            "identity": identity,
            "features": features_payload,
            "prediction": prediction,
            "predicted_eps_range": prediction.get("predicted_eps_range"),
            "beat_prob": prediction.get("beat_prob"),
            "meet_prob": prediction.get("meet_prob"),
            "miss_prob": prediction.get("miss_prob"),
            "sources": sources,
            "warnings": warnings,
            "as_of": _today_iso(),
            "confidence": prediction.get("confidence", 0.4),
            "keyless": True,
            "mode": "model_only",
        }

    consensus = args.get("consensus_eps")
    compared = compare_prediction_to_consensus(
        prediction=prediction,
        consensus_eps=consensus,
        actual_eps=args.get("actual_eps"),
    )
    return {
        "operation": operation,
        "identity": identity,
        "features": features_payload,
        "prediction": prediction,
        "comparison": compared,
        "sources": sources,
        "warnings": warnings,
        "as_of": _today_iso(),
        "confidence": prediction.get("confidence", 0.4),
        "keyless": True,
        "mode": "consensus_compare",
    }


async def _build_feature_payload(
    args: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
    payload = {
        "latest_eps": args.get("latest_eps"),
        "prior_eps": args.get("prior_eps", []),
        "revenue_ttm": args.get("revenue_ttm"),
        "net_margin_ttm": args.get("net_margin_ttm"),
        "price_returns": args.get("price_returns", []),
        "sentiment_score": args.get("sentiment_score"),
        "options_flow_score": args.get("options_flow_score"),
        "short_pressure_score": args.get("short_pressure_score"),
        "insider_score": args.get("insider_score"),
        "guidance_delta": args.get("guidance_delta"),
    }

    sources: list[str] = []
    warnings: list[str] = []
    identity: dict[str, Any] = {
        "ticker": str(args.get("ticker", "")).strip().upper(),
        "cik": str(args.get("cik", "")).strip(),
    }

    ticker = identity["ticker"]
    cik = identity["cik"]
    if ticker or cik:
        async with httpx.AsyncClient() as client:
            sec_payload, sec_identity, sec_sources, sec_warnings = await _augment_from_sec(
                ticker=ticker,
                cik=cik,
                payload=payload,
                client=client,
            )
        payload = sec_payload
        identity.update(sec_identity)
        sources.extend(sec_sources)
        warnings.extend(sec_warnings)

    features = build_earnings_features(payload)
    warnings.extend(features.get("warnings", []))
    return features, identity, sources, warnings


async def _augment_from_sec(
    *,
    ticker: str,
    cik: str,
    payload: dict[str, Any],
    client: httpx.AsyncClient,
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
    out = dict(payload)
    warnings: list[str] = []
    identity: dict[str, Any] = {"ticker": ticker, "cik": cik, "issuer_name": ""}

    resolved_cik = "".join(ch for ch in cik if ch.isdigit()).zfill(10) if cik else ""
    if not resolved_cik and ticker:
        try:
            resolved = await resolve_ticker_to_cik(ticker=ticker, client=client)
            resolved_cik = str(resolved.get("cik", "")).strip()
            identity["ticker"] = str(resolved.get("ticker", ticker)).strip().upper()
            identity["issuer_name"] = str(resolved.get("name", "")).strip()
        except SecDataError as e:
            warnings.append(f"SEC ticker lookup failed: {e}")
            return out, identity, [], warnings

    if not resolved_cik:
        return out, identity, [], warnings

    try:
        facts = await fetch_company_facts(cik=resolved_cik, client=client)
    except Exception as e:
        warnings.append(f"SEC companyfacts fetch failed: {e}")
        return out, {**identity, "cik": resolved_cik}, [], warnings

    identity["cik"] = resolved_cik
    identity["issuer_name"] = str(facts.get("entityName", identity.get("issuer_name", ""))).strip()

    if out.get("revenue_ttm") is None:
        revenue = _first_ttm(
            facts,
            [
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet",
            ],
        )
        out["revenue_ttm"] = revenue

    if out.get("net_margin_ttm") is None:
        net_income = _first_ttm(facts, ["NetIncomeLoss"])
        revenue = out.get("revenue_ttm")
        if revenue not in (None, 0) and net_income is not None:
            out["net_margin_ttm"] = float(net_income) / float(revenue)

    if out.get("latest_eps") is None:
        eps = _first_latest(facts, ["EarningsPerShareDiluted", "EarningsPerShareBasic"])
        out["latest_eps"] = eps

    if not out.get("prior_eps"):
        # Fallback to latest EPS when history is not provided.
        if out.get("latest_eps") is not None:
            out["prior_eps"] = [float(out["latest_eps"])]

    sources = [f"https://data.sec.gov/api/xbrl/companyfacts/CIK{resolved_cik}.json"]
    return out, identity, sources, warnings


def _first_ttm(facts: dict[str, Any], tags: list[str]) -> float | None:
    for tag in tags:
        row = extract_ttm_value(facts, tag=tag)
        if row is None:
            continue
        try:
            return float(row.get("value"))
        except (TypeError, ValueError):
            continue
    return None


def _first_latest(facts: dict[str, Any], tags: list[str]) -> float | None:
    for tag in tags:
        row = extract_latest_value(facts, tag=tag)
        if row is None:
            continue
        try:
            return float(row.get("value"))
        except (TypeError, ValueError):
            continue
    return None


def _today_iso() -> str:
    return date.today().isoformat()


def _finalize(
    tool: Tool,
    ctx: ToolContext,
    *,
    args: dict[str, Any],
    payload: dict[str, Any],
) -> ToolResult:
    lines = [f"Computed {payload.get('operation', '')}."]
    files_changed: list[str] = []
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
            "# Earnings Surprise Predictor Report",
            "",
            f"- **Operation**: {payload.get('operation', '')}",
            "",
            "```json",
            json.dumps(payload, indent=2),
            "```",
            "",
        ]
    )


__all__ = ["EarningsSurprisePredictorTool"]
