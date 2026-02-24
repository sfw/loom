"""Historical currency normalizer (keyless FX + optional US inflation)."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from loom.research.inflation import (
    SERIES_PROVENANCE,
    SERIES_VERSION,
    calculate_inflation,
)
from loom.research.providers import convert_via_ecb_reference_rates
from loom.tools.registry import Tool, ToolContext, ToolResult

_MODES = {"fx_only", "fx_and_us_inflation"}


class HistoricalCurrencyNormalizerTool(Tool):
    """Normalize historical monetary values with keyless public data."""

    @property
    def name(self) -> str:
        return "historical_currency_normalizer"

    @property
    def description(self) -> str:
        return (
            "Normalize historical monetary amounts using keyless ECB FX rates and "
            "optional bundled US CPI inflation adjustment."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "Original nominal amount.",
                },
                "from_currency": {
                    "type": "string",
                    "description": "Source currency code (for example USD).",
                },
                "to_currency": {
                    "type": "string",
                    "description": "Target currency code (for example EUR).",
                },
                "from_date": {
                    "type": "string",
                    "description": "Source valuation date (YYYY or YYYY-MM-DD).",
                },
                "to_date": {
                    "type": "string",
                    "description": "Target valuation date (YYYY or YYYY-MM-DD).",
                },
                "mode": {
                    "type": "string",
                    "enum": ["fx_only", "fx_and_us_inflation"],
                    "description": "Normalization mode.",
                },
                "inflation_adjust": {
                    "type": "boolean",
                    "description": "Alias for mode=fx_and_us_inflation when true.",
                },
                "index": {
                    "type": "string",
                    "enum": ["cpi_u", "cpi_w"],
                    "description": "US inflation index when inflation mode is enabled.",
                },
                "region": {
                    "type": "string",
                    "description": "Inflation region (currently US only).",
                },
                "force_refresh_rates": {
                    "type": "boolean",
                    "description": "Force-refresh ECB reference-rate cache.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional markdown output path.",
                },
            },
            "required": ["amount", "from_currency", "to_currency", "from_date", "to_date"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 45

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        amount = _to_float(args.get("amount"))
        if amount is None:
            return ToolResult.fail("amount is required")

        from_currency = str(args.get("from_currency", "")).strip().upper()
        to_currency = str(args.get("to_currency", "")).strip().upper()
        if not from_currency or not to_currency:
            return ToolResult.fail("from_currency and to_currency are required")

        from_day = _parse_date(args.get("from_date"))
        to_day = _parse_date(args.get("to_date"))
        if from_day is None or to_day is None:
            return ToolResult.fail("from_date/to_date must be YYYY or YYYY-MM-DD")

        mode = _resolve_mode(args)
        if mode not in _MODES:
            return ToolResult.fail("mode must be fx_only or fx_and_us_inflation")

        index = str(args.get("index", "cpi_u")).strip().lower() or "cpi_u"
        if index not in {"cpi_u", "cpi_w"}:
            return ToolResult.fail("index must be cpi_u or cpi_w")

        region = str(args.get("region", "US")).strip().upper() or "US"
        force_refresh = bool(args.get("force_refresh_rates", False))

        warnings: list[str] = []
        fx_snapshot: dict[str, Any]
        try:
            fx_snapshot = await convert_via_ecb_reference_rates(
                amount=amount,
                from_currency=from_currency,
                to_currency=to_currency,
                from_date=from_day,
                to_date=to_day,
                force_refresh=force_refresh,
            )
        except Exception as e:
            return ToolResult.fail(f"FX conversion failed: {e}")

        warnings.extend(list(fx_snapshot.get("warnings", [])))

        partial_result = False
        inflation_payload: dict[str, Any] | None = None
        final_amount = float(fx_snapshot["converted_amount"])
        computation_path = "fx_only"

        if mode == "fx_and_us_inflation":
            if region != "US":
                partial_result = True
                warnings.append("Inflation adjustment currently supports region=US only.")
            else:
                computation_path = "fx_and_us_inflation"
                try:
                    to_usd = await convert_via_ecb_reference_rates(
                        amount=amount,
                        from_currency=from_currency,
                        to_currency="USD",
                        from_date=from_day,
                        to_date=from_day,
                        force_refresh=force_refresh,
                    )
                    inflated = calculate_inflation(
                        amount=float(to_usd["converted_amount"]),
                        from_year=from_day.year,
                        to_year=to_day.year,
                        index=index,
                    )
                    usd_to_target = await convert_via_ecb_reference_rates(
                        amount=float(inflated.adjusted_amount),
                        from_currency="USD",
                        to_currency=to_currency,
                        from_date=to_day,
                        to_date=to_day,
                        force_refresh=force_refresh,
                    )
                except Exception as e:
                    partial_result = True
                    warnings.append(f"Inflation path unavailable ({type(e).__name__}: {e})")
                else:
                    final_amount = float(usd_to_target["converted_amount"])
                    inflation_payload = {
                        "region": "US",
                        "index": index,
                        "series_version": SERIES_VERSION,
                        "series_provenance": SERIES_PROVENANCE,
                        "usd_amount_at_from_date": float(to_usd["converted_amount"]),
                        "usd_amount_inflation_adjusted": float(inflated.adjusted_amount),
                        "inflation_multiplier": float(inflated.multiplier),
                        "inflation_percent_change": float(inflated.percent_change),
                        "intermediate": {
                            "to_usd": to_usd,
                            "usd_to_target": usd_to_target,
                        },
                    }

        payload = {
            "amount": amount,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "from_date": from_day.isoformat(),
            "to_date": to_day.isoformat(),
            "mode": mode,
            "computation_path": computation_path,
            "fx_snapshot": fx_snapshot,
            "inflation": inflation_payload,
            "normalized_amount": final_amount,
            "partial_result": partial_result,
            "warnings": warnings,
            "keyless": True,
        }

        lines = [
            (
                f"{amount:,.4f} {from_currency} ({from_day.isoformat()}) -> "
                f"{final_amount:,.4f} {to_currency} ({to_day.isoformat()})"
            ),
            f"Mode: {mode}",
            f"FX source: {fx_snapshot.get('source', '')}",
        ]
        if inflation_payload:
            lines.append(
                "Inflation: "
                f"{inflation_payload.get('index')} "
                f"multiplier={inflation_payload.get('inflation_multiplier'):.6f}"
            )
        if warnings:
            lines.append("Warnings: " + "; ".join(warnings))

        files_changed: list[str] = []
        output_path = str(args.get("output_path", "")).strip()
        if output_path:
            files_changed = _write_report(
                tool=self,
                payload=payload,
                output_path=output_path,
                ctx=ctx,
            )
            if files_changed:
                lines.append("Report written: " + ", ".join(files_changed))

        return ToolResult.ok(
            "\n".join(lines),
            data=payload,
            files_changed=files_changed,
        )


def _resolve_mode(args: dict[str, Any]) -> str:
    mode = str(args.get("mode", "")).strip().lower()
    if mode:
        return mode
    if bool(args.get("inflation_adjust", False)):
        return "fx_and_us_inflation"
    return "fx_only"


def _to_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) == 4 and text.isdigit():
        return date(int(text), 1, 1)
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _write_report(
    *,
    tool: Tool,
    payload: dict[str, Any],
    output_path: str,
    ctx: ToolContext,
) -> list[str]:
    if ctx.workspace is None:
        return []
    path = tool._resolve_path(output_path, ctx.workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    if ctx.changelog is not None:
        ctx.changelog.record_before_write(str(path), subtask_id=ctx.subtask_id)
    lines = [
        "# Historical Currency Normalization",
        "",
        f"- **From**: {payload.get('amount')} {payload.get('from_currency')}",
        f"- **To**: {payload.get('to_currency')}",
        f"- **From Date**: {payload.get('from_date')}",
        f"- **To Date**: {payload.get('to_date')}",
        f"- **Mode**: {payload.get('mode')}",
        f"- **Normalized Amount**: {payload.get('normalized_amount')}",
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return [str(path.relative_to(ctx.workspace))]
