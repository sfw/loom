"""Evidence ledger helpers for evidence-first verification.

Evidence records are stored outside task state YAML so prompt context stays
small while verification can still reason over persisted evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any

_EVIDENCE_TOOLS = frozenset({
    "web_fetch",
    "web_fetch_html",
    "web_search",
})
_SOURCE_TOOLS = frozenset({"web_fetch", "web_fetch_html"})

_DIMENSION_HINTS: tuple[tuple[str, str], ...] = (
    ("politic", "political"),
    ("regulat", "political"),
    ("policy", "political"),
    ("econom", "economic"),
    ("inflation", "economic"),
    ("cost", "economic"),
    ("social", "social"),
    ("demograph", "social"),
    ("customer", "social"),
    ("tech", "technological"),
    ("digital", "technological"),
    ("automation", "technological"),
    ("legal", "legal"),
    ("compliance", "legal"),
    ("environment", "environmental"),
    ("climate", "environmental"),
    ("emission", "environmental"),
)

_MARKET_KEYS = ("market", "geography", "region", "state", "province", "country")


def _normalize_text(value: object) -> str:
    text = str(value or "").strip()
    return re.sub(r"\s+", " ", text)


def _safe_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in mapping:
            text = _normalize_text(mapping.get(key))
            if text:
                return text
    return ""


def _infer_dimension(text: str) -> str:
    lowered = text.lower()
    for needle, label in _DIMENSION_HINTS:
        if needle in lowered:
            return label
    return ""


def _infer_market(args: dict[str, Any], context_text: str) -> str:
    market = _first_present(args, _MARKET_KEYS)
    if market:
        return market

    lowered = context_text.lower()
    for marker in (" in ", " for ", " across ", " within "):
        idx = lowered.find(marker)
        if idx == -1:
            continue
        candidate = _normalize_text(context_text[idx + len(marker) : idx + len(marker) + 48])
        candidate = re.split(r"[.,;:|()\[\]]", candidate)[0].strip()
        if len(candidate) >= 3:
            return candidate
    return ""


def _snippet(text: str, *, limit: int = 280) -> str:
    value = _normalize_text(text)
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def _score_quality(source_url: str, snippet_text: str, query_text: str) -> float:
    score = 0.35
    if source_url:
        score += 0.25
    if query_text:
        score += 0.15
    if len(snippet_text) >= 120:
        score += 0.2
    elif len(snippet_text) >= 60:
        score += 0.1
    return max(0.0, min(1.0, score))


def _first_url(text: str) -> str:
    match = re.search(r"https?://[^\s<>\]\"')]+", str(text or ""), flags=re.IGNORECASE)
    return _normalize_text(match.group(0)) if match else ""


def _evidence_kind(tool_name: str) -> str:
    if tool_name in _SOURCE_TOOLS:
        return "source"
    if tool_name == "web_search":
        return "discovery"
    return "context"


def _next_evidence_id(
    *,
    tool_name: str,
    source_url: str,
    query_text: str,
    snippet_text: str,
    subtask_id: str,
    existing_ids: set[str],
) -> str:
    seed = "|".join([
        tool_name,
        subtask_id,
        source_url,
        query_text,
        snippet_text[:140],
    ])
    digest = hashlib.sha1(seed.encode("utf-8", errors="replace")).hexdigest().upper()
    base = f"EV-{digest[:10]}"
    if base not in existing_ids:
        return base
    suffix = 2
    while True:
        candidate = f"{base}-{suffix}"
        if candidate not in existing_ids:
            return candidate
        suffix += 1


def _record_payload(call: Any) -> tuple[str, dict[str, Any], Any]:
    tool = str(getattr(call, "tool", "") or "")
    args = getattr(call, "args", {})
    if not isinstance(args, dict):
        args = {}
    result = getattr(call, "result", None)
    return tool, args, result


def extract_evidence_records(
    *,
    task_id: str,
    subtask_id: str,
    tool_calls: list[Any],
    existing_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract normalized evidence records from successful tool calls."""
    seen_ids = set(existing_ids or set())
    records: list[dict[str, Any]] = []

    for call in tool_calls:
        tool, args, result = _record_payload(call)
        if tool not in _EVIDENCE_TOOLS:
            continue
        if result is None or not getattr(result, "success", False):
            continue

        result_data = getattr(result, "data", None)
        if not isinstance(result_data, dict):
            result_data = {}

        source_url = _normalize_text(
            result_data.get("url")
            or args.get("url")
            or args.get("source_url")
            or ""
        )
        query_text = _normalize_text(
            args.get("query")
            or args.get("q")
            or args.get("search_query")
            or result_data.get("query")
            or ""
        )
        output_text = _normalize_text(getattr(result, "output", ""))
        if not source_url and tool == "web_search":
            source_url = _first_url(output_text)
        snippet_text = _snippet(output_text)
        context_text = _normalize_text(" ".join([
            query_text,
            source_url,
            _safe_json(args),
            snippet_text,
        ]))
        if not context_text:
            continue

        market = _infer_market(args, context_text)
        dimension = _infer_dimension(context_text)
        quality = _score_quality(source_url, snippet_text, query_text)
        evidence_id = _next_evidence_id(
            tool_name=tool,
            source_url=source_url,
            query_text=query_text,
            snippet_text=snippet_text,
            subtask_id=subtask_id,
            existing_ids=seen_ids,
        )
        seen_ids.add(evidence_id)
        created_at = _normalize_text(getattr(call, "timestamp", ""))
        if not created_at:
            created_at = datetime.now().isoformat()

        records.append({
            "evidence_id": evidence_id,
            "task_id": task_id,
            "subtask_id": subtask_id,
            "tool": tool,
            "evidence_kind": _evidence_kind(tool),
            "tool_call_id": _normalize_text(getattr(call, "call_id", "")),
            "source_url": source_url,
            "query": query_text,
            "market": market,
            "dimension": dimension,
            "snippet": snippet_text,
            "context_text": context_text[:1200],
            "quality": quality,
            "created_at": created_at,
        })

    return records


def merge_evidence_records(
    existing: list[dict[str, Any]] | None,
    new_records: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Merge evidence records by evidence_id, preserving first-seen order."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    for bucket in (existing or [], new_records or []):
        for item in bucket:
            if not isinstance(item, dict):
                continue
            evidence_id = _normalize_text(item.get("evidence_id"))
            if not evidence_id:
                fingerprint = hashlib.sha1(
                    _safe_json(item).encode("utf-8", errors="replace")
                ).hexdigest()
                evidence_id = f"EV-{fingerprint[:10].upper()}"
                item = dict(item)
                item["evidence_id"] = evidence_id
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            merged.append(dict(item))
    return merged


def summarize_evidence_records(
    records: list[dict[str, Any]] | None,
    *,
    max_entries: int = 12,
) -> str:
    """Build a compact ledger summary for prompts/logs."""
    if not records:
        return "No prior evidence records."

    lines: list[str] = []
    for item in (records or [])[:max_entries]:
        if not isinstance(item, dict):
            continue
        evidence_id = _normalize_text(item.get("evidence_id")) or "EV-UNKNOWN"
        market = _normalize_text(item.get("market")) or "unspecified"
        dimension = _normalize_text(item.get("dimension")) or "unspecified"
        source = _normalize_text(item.get("source_url")) or _normalize_text(item.get("query"))
        quality_raw = item.get("quality", 0.0)
        try:
            quality = float(quality_raw)
        except (TypeError, ValueError):
            quality = 0.0
        source_preview = source[:80] + "..." if len(source) > 83 else source
        lines.append(
            f"- {evidence_id} | market={market} | dimension={dimension} | "
            f"quality={quality:.2f} | source={source_preview or 'n/a'}"
        )

    if not lines:
        return "No prior evidence records."
    if len(records or []) > max_entries:
        lines.append(f"... ({len(records or []) - max_entries} more evidence records)")
    return "\n".join(lines)
