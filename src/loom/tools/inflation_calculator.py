"""Inflation calculator tool (offline, no API keys)."""

from __future__ import annotations

import json

from loom.research.inflation import (
    SERIES_PROVENANCE,
    SERIES_VERSION,
    available_years,
    calculate_inflation,
    cpi_series,
)
from loom.tools.registry import Tool, ToolContext, ToolResult


class InflationCalculatorTool(Tool):
    """Convert nominal amounts across years using bundled CPI data."""

    @property
    def name(self) -> str:
        return "inflation_calculator"

    @property
    def description(self) -> str:
        return (
            "Calculate inflation-adjusted USD values using bundled CPI "
            "series (keyless/offline)."
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
                "from_year": {
                    "type": "integer",
                    "description": "Source year.",
                },
                "to_year": {
                    "type": "integer",
                    "description": "Target year.",
                },
                "index": {
                    "type": "string",
                    "enum": ["cpi_u", "cpi_w"],
                    "description": "Inflation index. Defaults to cpi_u.",
                },
                "region": {
                    "type": "string",
                    "description": "Supported region is currently US.",
                },
                "include_series": {
                    "type": "boolean",
                    "description": "Include year-by-year index values between endpoints.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional markdown report output path.",
                },
            },
            "required": ["amount", "from_year", "to_year"],
        }

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        region = str(args.get("region", "US")).strip().upper() or "US"
        if region != "US":
            return ToolResult.fail("Only US region is supported in this pass")

        amount = _to_float(args.get("amount"))
        from_year = _to_int(args.get("from_year"))
        to_year = _to_int(args.get("to_year"))
        if amount is None or from_year is None or to_year is None:
            return ToolResult.fail("amount, from_year, and to_year are required")

        index = str(args.get("index", "cpi_u")).strip().lower() or "cpi_u"
        include_series = bool(args.get("include_series", False))

        try:
            result = calculate_inflation(
                amount=amount,
                from_year=from_year,
                to_year=to_year,
                index=index,
            )
        except ValueError as e:
            return ToolResult.fail(str(e))

        output_lines = [
            f"${amount:,.2f} in {from_year} -> ${result.adjusted_amount:,.2f} in {to_year}",
            (
                f"Multiplier: {result.multiplier:.6f} "
                f"({result.percent_change:+.2f}% cumulative change)"
            ),
            (
                f"Index: {index} ({from_year}: {result.from_index_value:.3f}, "
                f"{to_year}: {result.to_index_value:.3f})"
            ),
        ]
        if result.note:
            output_lines.append(f"Note: {result.note}")

        payload: dict = {
            "amount": amount,
            "from_year": from_year,
            "to_year": to_year,
            "adjusted_amount": result.adjusted_amount,
            "multiplier": result.multiplier,
            "percent_change": result.percent_change,
            "index": index,
            "from_index_value": result.from_index_value,
            "to_index_value": result.to_index_value,
            "region": region,
            "series_version": SERIES_VERSION,
            "series_provenance": SERIES_PROVENANCE,
            "note": result.note,
        }

        if include_series:
            series, _ = cpi_series(index)
            lo = min(from_year, to_year)
            hi = max(from_year, to_year)
            payload["series"] = [
                {
                    "year": year,
                    "index_value": series[year],
                    "multiplier_vs_from": series[year] / result.from_index_value,
                }
                for year in range(lo, hi + 1)
                if year in series
            ]

        files_changed: list[str] = []
        output_path_raw = str(args.get("output_path", "")).strip()
        if output_path_raw:
            if ctx.workspace is None:
                return ToolResult.fail("No workspace set for output_path")
            report_path = self._resolve_path(output_path_raw, ctx.workspace)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            if ctx.changelog is not None:
                ctx.changelog.record_before_write(str(report_path), subtask_id=ctx.subtask_id)
            report_path.write_text(
                _render_markdown_report(payload),
                encoding="utf-8",
            )
            rel = report_path.relative_to(ctx.workspace)
            files_changed.append(str(rel))
            output_lines.append(f"Report written: {rel}")

        return ToolResult.ok(
            "\n".join(output_lines),
            data=payload,
            files_changed=files_changed,
        )


def _to_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _render_markdown_report(payload: dict) -> str:
    amount = float(payload["amount"])
    adjusted = float(payload["adjusted_amount"])
    percent = float(payload["percent_change"])
    lines = [
        "# Inflation Conversion",
        "",
        f"- **Amount**: ${amount:,.2f}",
        f"- **From Year**: {payload['from_year']}",
        f"- **To Year**: {payload['to_year']}",
        f"- **Adjusted Amount**: ${adjusted:,.2f}",
        f"- **Cumulative Change**: {percent:+.2f}%",
        f"- **Index**: {payload['index']}",
        f"- **Series Version**: {payload['series_version']}",
        f"- **Provenance**: {payload['series_provenance']}",
    ]
    if payload.get("note"):
        lines.append(f"- **Note**: {payload['note']}")

    series = payload.get("series", [])
    if isinstance(series, list) and series:
        lines.extend([
            "",
            "## Index Series",
            "",
            "| Year | Index Value | Multiplier vs From |",
            "|---|---:|---:|",
        ])
        for row in series:
            year = int(row.get("year", 0))
            index_value = float(row.get("index_value", 0.0))
            mult = float(row.get("multiplier_vs_from", 0.0))
            lines.append(f"| {year} | {index_value:.3f} | {mult:.6f} |")

    lines.extend(["", "```json", json.dumps(payload, indent=2), "```", ""])
    return "\n".join(lines)


def supported_year_range() -> tuple[int, int]:
    """Expose supported year range for tests and diagnostics."""
    return available_years()
