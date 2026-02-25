"""Investor sentiment signal tool."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

import httpx

from loom.research.finance import clamp, safe_float
from loom.tools.registry import Tool, ToolContext, ToolResult

_OPERATIONS = {"score_sentiment", "get_put_call", "get_short_flow", "get_cot_positioning"}


class SentimentFeedsApiTool(Tool):
    """Aggregate and normalize investor sentiment inputs."""

    @property
    def name(self) -> str:
        return "sentiment_feeds_api"

    @property
    def description(self) -> str:
        return (
            "Compute investor sentiment scores and parse key sentiment proxies "
            "(put/call, short flow, COT positioning)."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "score_sentiment",
                        "get_put_call",
                        "get_short_flow",
                        "get_cot_positioning",
                    ],
                },
                "signals": {
                    "type": "array",
                    "description": (
                        "Signal records for score_sentiment: "
                        "{name,value,neutral,scale,direction,weight}."
                    ),
                    "items": {"type": "object"},
                },
                "put_volume": {"type": "number"},
                "call_volume": {"type": "number"},
                "short_volume": {"type": "number"},
                "total_volume": {"type": "number"},
                "long_contracts": {"type": "number"},
                "short_contracts": {"type": "number"},
                "source_path": {
                    "type": "string",
                    "description": "Optional CSV path for ratio calculations.",
                },
                "source_url": {
                    "type": "string",
                    "description": "Optional CSV URL for ratio calculations.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional markdown output path.",
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
            return ToolResult.fail(
                "operation must be score_sentiment/get_put_call/get_short_flow/get_cot_positioning"
            )

        try:
            if operation == "score_sentiment":
                payload = _score_sentiment(args)
            elif operation == "get_put_call":
                payload = await _put_call_payload(self, args, ctx)
            elif operation == "get_short_flow":
                payload = await _short_flow_payload(self, args, ctx)
            else:
                payload = await _cot_payload(self, args, ctx)
        except Exception as e:
            return ToolResult.fail(str(e))

        lines = [f"Computed sentiment payload for {operation}."]
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
            lines.append(f"Report written: {rel}")

        return ToolResult.ok("\n".join(lines), data=payload, files_changed=files_changed)


def _score_sentiment(args: dict[str, Any]) -> dict[str, Any]:
    raw = args.get("signals", [])
    if not isinstance(raw, list):
        raw = []

    rows: list[dict[str, Any]] = []
    weighted_total = 0.0
    weight_sum = 0.0
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "signal")).strip() or "signal"
        value = safe_float(item.get("value"))
        neutral = safe_float(item.get("neutral"))
        scale = safe_float(item.get("scale")) or 1.0
        direction = str(item.get("direction", "higher_is_bullish")).strip().lower()
        weight = safe_float(item.get("weight"))
        if weight is None or weight <= 0:
            weight = 1.0
        if value is None:
            continue
        if neutral is None:
            neutral = 0.0

        z = (value - neutral) / scale if scale != 0 else 0.0
        score = clamp(z, -3.0, 3.0) / 3.0
        if direction == "higher_is_bearish":
            score *= -1.0
        weighted_total += score * weight
        weight_sum += weight
        rows.append(
            {
                "name": name,
                "value": value,
                "neutral": neutral,
                "scale": scale,
                "direction": direction,
                "weight": weight,
                "normalized_score": score,
            }
        )

    aggregate = weighted_total / weight_sum if weight_sum > 0 else 0.0
    sentiment_label = _label_from_score(aggregate)
    confidence = 0.2 + min(0.75, 0.1 * len(rows))
    return {
        "operation": "score_sentiment",
        "signal_count": len(rows),
        "signals": rows,
        "aggregate_score": aggregate,
        "sentiment_label": sentiment_label,
        "confidence": confidence,
        "keyless": True,
    }


async def _put_call_payload(tool: Tool, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    put_v = safe_float(args.get("put_volume"))
    call_v = safe_float(args.get("call_volume"))
    source = "inline"
    if put_v is None or call_v is None:
        row, source = await _load_ratio_row(tool, args, ctx)
        put_v = safe_float(row.get("put_volume") or row.get("put"))
        call_v = safe_float(row.get("call_volume") or row.get("call"))
    if put_v is None or call_v in (None, 0):
        raise ValueError("put_volume and call_volume (or parseable source) are required")
    ratio = put_v / call_v
    return {
        "operation": "get_put_call",
        "put_volume": put_v,
        "call_volume": call_v,
        "put_call_ratio": ratio,
        "interpretation": "bearish" if ratio > 1.05 else ("bullish" if ratio < 0.85 else "neutral"),
        "source": source,
        "keyless": True,
    }


async def _short_flow_payload(tool: Tool, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    short_v = safe_float(args.get("short_volume"))
    total_v = safe_float(args.get("total_volume"))
    source = "inline"
    if short_v is None or total_v is None:
        row, source = await _load_ratio_row(tool, args, ctx)
        short_v = safe_float(row.get("short_volume") or row.get("short"))
        total_v = safe_float(row.get("total_volume") or row.get("total"))
    if short_v is None or total_v in (None, 0):
        raise ValueError("short_volume and total_volume (or parseable source) are required")
    ratio = short_v / total_v
    score = clamp((ratio - 0.45) / 0.25, -1.0, 1.0) * -1.0
    return {
        "operation": "get_short_flow",
        "short_volume": short_v,
        "total_volume": total_v,
        "short_volume_ratio": ratio,
        "sentiment_score": score,
        "interpretation": _label_from_score(score),
        "source": source,
        "keyless": True,
    }


async def _cot_payload(tool: Tool, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    long_v = safe_float(args.get("long_contracts"))
    short_v = safe_float(args.get("short_contracts"))
    source = "inline"
    if long_v is None or short_v is None:
        row, source = await _load_ratio_row(tool, args, ctx)
        long_v = safe_float(row.get("long_contracts") or row.get("long"))
        short_v = safe_float(row.get("short_contracts") or row.get("short"))
    if long_v is None or short_v is None:
        raise ValueError("long_contracts and short_contracts (or parseable source) are required")
    gross = long_v + short_v
    net = long_v - short_v
    net_ratio = (net / gross) if gross > 0 else 0.0
    return {
        "operation": "get_cot_positioning",
        "long_contracts": long_v,
        "short_contracts": short_v,
        "net_contracts": net,
        "net_ratio": net_ratio,
        "sentiment_label": _label_from_score(clamp(net_ratio, -1.0, 1.0)),
        "source": source,
        "keyless": True,
    }


async def _load_ratio_row(
    tool: Tool,
    args: dict[str, Any],
    ctx: ToolContext,
) -> tuple[dict[str, Any], str]:
    source_path = str(args.get("source_path", "")).strip()
    source_url = str(args.get("source_url", "")).strip()
    if source_path:
        if ctx.workspace is None:
            raise ValueError("No workspace set for source_path")
        path = tool._resolve_read_path(source_path, ctx.workspace, ctx.read_roots)
        text = path.read_text(encoding="utf-8", errors="ignore")
        row = _last_csv_row(text)
        return row, str(path)
    if source_url:
        async with httpx.AsyncClient() as client:
            response = await client.get(source_url)
            response.raise_for_status()
            row = _last_csv_row(response.text)
            return row, source_url
    raise ValueError("Provide inline fields or source_path/source_url")


def _last_csv_row(csv_text: str) -> dict[str, Any]:
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
    rows = [row for row in reader if isinstance(row, dict)]
    if not rows:
        raise ValueError("No CSV rows found")
    return rows[-1]


def _label_from_score(score: float) -> str:
    if score >= 0.35:
        return "bullish"
    if score <= -0.35:
        return "bearish"
    return "neutral"


def _render_report(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Sentiment Feeds API Report",
            "",
            f"- **Operation**: {payload.get('operation', '')}",
            "",
            "```json",
            json.dumps(payload, indent=2),
            "```",
            "",
        ]
    )


__all__ = ["SentimentFeedsApiTool"]
