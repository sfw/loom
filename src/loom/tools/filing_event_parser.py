"""Deterministic filing-event signal parser."""

from __future__ import annotations

import json
import re
from typing import Any

from loom.tools.registry import Tool, ToolContext, ToolResult

_OPERATIONS = {
    "extract_guidance_changes",
    "extract_buyback_dividend_changes",
    "extract_insider_activity",
}

_GUIDANCE_PATTERNS = [
    (
        re.compile(r"\b(raise|raised|increased|increasing)\b.{0,80}\b(guidance|outlook)\b", re.I),
        "guidance_up",
    ),
    (
        re.compile(r"\b(lower|lowered|reduced|reducing)\b.{0,80}\b(guidance|outlook)\b", re.I),
        "guidance_down",
    ),
    (re.compile(r"\bwithdrew\b.{0,80}\b(guidance|outlook)\b", re.I), "guidance_withdrawn"),
]

_CAPITAL_RETURN_PATTERNS = [
    (
        re.compile(
            r"\b(repurchase|buyback)\b.{0,80}\b(authori[sz]ed|program|increase|expanded)\b", re.I
        ),
        "buyback_authorization",
    ),
    (re.compile(r"\bdividend\b.{0,80}\b(increase|raised|boosted)\b", re.I), "dividend_increase"),
    (re.compile(r"\bdividend\b.{0,80}\b(cut|reduced|suspended)\b", re.I), "dividend_decrease"),
]

_INSIDER_PATTERNS = [
    (re.compile(r"\bform\s*4\b", re.I), "form4_reference"),
    (re.compile(r"\binsider(s)?\b.{0,80}\b(bought|purchase|acquired)\b", re.I), "insider_buying"),
    (re.compile(r"\binsider(s)?\b.{0,80}\b(sold|sale|disposed)\b", re.I), "insider_selling"),
]


class FilingEventParserTool(Tool):
    """Extract event signals from filing-like text blobs."""

    @property
    def name(self) -> str:
        return "filing_event_parser"

    @property
    def description(self) -> str:
        return (
            "Extract filing events (guidance, buyback/dividend, insider activity) "
            "from provided text/files."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "extract_guidance_changes",
                        "extract_buyback_dividend_changes",
                        "extract_insider_activity",
                    ],
                },
                "text": {
                    "type": "string",
                    "description": "Inline text to analyze.",
                },
                "path": {
                    "type": "string",
                    "description": "Optional single text/markdown file path.",
                },
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional file path list.",
                },
                "max_events": {
                    "type": "integer",
                    "description": "Max extracted events (default 50).",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional markdown output path.",
                },
            },
            "required": ["operation"],
        }

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        operation = str(args.get("operation", "")).strip().lower()
        if operation not in _OPERATIONS:
            return ToolResult.fail(
                "operation must be extract_guidance_changes/"
                "extract_buyback_dividend_changes/extract_insider_activity"
            )

        max_events = _clamp_int(args.get("max_events"), default=50, lo=1, hi=500)
        text = _collect_text(self, args, ctx)
        if not text.strip():
            return ToolResult.fail("Provide text and/or path(s) with parseable content")

        if operation == "extract_guidance_changes":
            rules = _GUIDANCE_PATTERNS
        elif operation == "extract_buyback_dividend_changes":
            rules = _CAPITAL_RETURN_PATTERNS
        else:
            rules = _INSIDER_PATTERNS

        sentences = _split_sentences(text)
        events: list[dict[str, Any]] = []
        for idx, sentence in enumerate(sentences):
            for pattern, event_type in rules:
                if not pattern.search(sentence):
                    continue
                events.append(
                    {
                        "event_type": event_type,
                        "sentence_index": idx,
                        "snippet": sentence[:360],
                        "confidence": _confidence_for_event(event_type, sentence),
                    }
                )
                if len(events) >= max_events:
                    break
            if len(events) >= max_events:
                break

        payload = {
            "operation": operation,
            "event_count": len(events),
            "events": events,
            "keyless": True,
            "confidence": _aggregate_confidence(events),
            "warnings": [] if events else ["No matching event signals found in provided text."],
        }

        lines = [f"Parsed {len(events)} event(s) for {operation}."]
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


def _clamp_int(value: object, *, default: int, lo: int, hi: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lo, min(hi, parsed))


def _collect_text(tool: Tool, args: dict[str, Any], ctx: ToolContext) -> str:
    chunks: list[str] = []
    inline = str(args.get("text", "")).strip()
    if inline:
        chunks.append(inline)

    paths: list[str] = []
    single = str(args.get("path", "")).strip()
    if single:
        paths.append(single)
    raw = args.get("paths", [])
    if isinstance(raw, list):
        for item in raw:
            text = str(item or "").strip()
            if text:
                paths.append(text)

    if paths and ctx.workspace is not None:
        for item in paths:
            path = tool._resolve_read_path(item, ctx.workspace, ctx.read_roots)
            if not path.exists() or not path.is_file():
                continue
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))

    return "\n\n".join(chunks)


def _split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for piece in re.split(r"(?<=[\.\?!])\s+", text):
        sentence = " ".join(piece.strip().split())
        if sentence:
            out.append(sentence)
    return out


def _confidence_for_event(event_type: str, sentence: str) -> float:
    score = 0.55
    if any(ch.isdigit() for ch in sentence):
        score += 0.1
    if "%" in sentence or "$" in sentence:
        score += 0.1
    if "board" in sentence.lower() or "authorized" in sentence.lower():
        score += 0.1
    if event_type in {"form4_reference", "insider_buying", "insider_selling"}:
        score += 0.05
    return min(0.95, score)


def _aggregate_confidence(events: list[dict[str, Any]]) -> float:
    if not events:
        return 0.25
    total = sum(float(event.get("confidence", 0.0)) for event in events)
    return max(0.0, min(1.0, total / len(events)))


def _render_report(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Filing Event Parser Report",
            "",
            f"- **Operation**: {payload.get('operation', '')}",
            f"- **Event Count**: {payload.get('event_count', 0)}",
            "",
            "```json",
            json.dumps(payload, indent=2),
            "```",
            "",
        ]
    )


__all__ = ["FilingEventParserTool"]
