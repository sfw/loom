"""Tool catalog discovery for hybrid cowork tool exposure.

Provides a compact, filterable view of currently available tools so models can
discover long-tail tools without receiving full schemas on every turn.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from loom.tools.registry import Tool, ToolContext, ToolResult

DEFAULT_LIMIT = 20
MAX_LIMIT = 50
DEFAULT_MAX_PAYLOAD_BYTES = 12_000
MIN_MAX_PAYLOAD_BYTES = 1_024
MAX_MAX_PAYLOAD_BYTES = 65_536


def _clamp_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _required_args(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    required_raw = parameters.get("required", [])
    properties = parameters.get("properties", {})
    required_names: list[str] = []
    if isinstance(required_raw, list):
        for item in required_raw:
            text = str(item or "").strip()
            if text and text not in required_names:
                required_names.append(text)

    if not isinstance(properties, dict):
        properties = {}

    result: list[dict[str, Any]] = []
    for name in required_names:
        prop = properties.get(name, {})
        if not isinstance(prop, dict):
            prop = {}
        arg: dict[str, Any] = {
            "name": name,
            "type": str(prop.get("type", "any") or "any"),
        }
        enum = prop.get("enum", [])
        if isinstance(enum, list) and enum:
            arg["enum"] = [str(item) for item in enum[:20]]
        result.append(arg)
    return result


def _encoded_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8", errors="replace"))


class ListToolsTool(Tool):
    """Return a compact list of available tools with optional filtering."""

    name = "list_tools"
    description = (
        "List currently available tools in compact form. Supports filtering by "
        "query, category, mutating_only, and auth_required_only. Use this when "
        "you need a tool not present in the typed tool subset."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional substring filter against tool name/description.",
            },
            "category": {
                "type": "string",
                "description": "Optional category filter (for example core/coding/web/finance).",
            },
            "mutating_only": {
                "type": "boolean",
                "description": "If true, return only mutating tools.",
            },
            "auth_required_only": {
                "type": "boolean",
                "description": "If true, return only tools with auth requirements.",
            },
            "detail": {
                "type": "string",
                "enum": ["compact", "schema"],
                "description": (
                    "compact: name/summary/required args only. "
                    "schema: include full parameters schema."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Result page size (default 20, max 50).",
            },
            "offset": {
                "type": "integer",
                "description": "Result offset for pagination.",
            },
            "max_payload_bytes": {
                "type": "integer",
                "description": (
                    "Optional response byte budget. Clamped to a safe range."
                ),
            },
        },
    }

    def __init__(
        self,
        catalog_provider: Callable[[Any | None], list[dict[str, Any]]] | None = None,
    ) -> None:
        self._catalog_provider = catalog_provider

    def bind(
        self,
        catalog_provider: Callable[[Any | None], list[dict[str, Any]]],
    ) -> None:
        self._catalog_provider = catalog_provider

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if self._catalog_provider is None:
            return ToolResult.fail("list_tools is unavailable (catalog provider not bound).")

        detail = str(args.get("detail", "compact") or "compact").strip().lower()
        if detail not in {"compact", "schema"}:
            detail = "compact"

        query = str(args.get("query", "") or "").strip().lower()
        category = str(args.get("category", "") or "").strip().lower()
        mutating_only = _coerce_bool(args.get("mutating_only", False))
        auth_required_only = _coerce_bool(args.get("auth_required_only", False))
        limit = _clamp_int(args.get("limit"), default=DEFAULT_LIMIT, minimum=1, maximum=MAX_LIMIT)
        offset = _clamp_int(args.get("offset"), default=0, minimum=0, maximum=100_000)
        max_payload_bytes = _clamp_int(
            args.get("max_payload_bytes"),
            default=DEFAULT_MAX_PAYLOAD_BYTES,
            minimum=MIN_MAX_PAYLOAD_BYTES,
            maximum=MAX_MAX_PAYLOAD_BYTES,
        )

        try:
            rows = self._catalog_provider(ctx.auth_context)
        except Exception as e:
            return ToolResult.fail(f"list_tools provider error: {type(e).__name__}: {e}")

        if not isinstance(rows, list):
            return ToolResult.fail("list_tools provider returned invalid data.")

        filtered: list[dict[str, Any]] = []
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", "") or "").strip()
            if not name:
                continue
            summary = str(raw.get("summary", "") or "").strip()
            description = str(raw.get("description", "") or "").strip()
            row_category = str(raw.get("category", "") or "").strip().lower()
            mutates = bool(raw.get("mutates", False))
            auth_required = bool(raw.get("auth_required", False))
            parameters = raw.get("parameters", {})
            if not isinstance(parameters, dict):
                parameters = {}

            if query:
                haystack = f"{name}\n{summary}\n{description}".lower()
                if query not in haystack:
                    continue
            if category and row_category != category:
                continue
            if mutating_only and not mutates:
                continue
            if auth_required_only and not auth_required:
                continue

            filtered.append({
                "name": name,
                "summary": summary,
                "description": description,
                "category": row_category,
                "mutates": mutates,
                "auth_required": auth_required,
                "parameters": parameters,
            })

        filtered.sort(key=lambda item: str(item.get("name", "")))
        total_count = len(filtered)
        page_candidates = filtered[offset : offset + limit]

        response: dict[str, Any] = {
            "detail": detail,
            "total_count": total_count,
            "offset": offset,
            "limit": limit,
            "has_more": False,
            "truncated_by_size": False,
            "max_payload_bytes": max_payload_bytes,
            "tools": [],
        }

        for row in page_candidates:
            item = {
                "name": row["name"],
                "summary": row["summary"],
                "category": row["category"],
                "mutates": row["mutates"],
                "auth_required": row["auth_required"],
            }
            if detail == "schema":
                item["parameters"] = row["parameters"]
            else:
                item["required_args"] = _required_args(row["parameters"])

            candidate_tools = [*response["tools"], item]
            candidate_response = {**response, "tools": candidate_tools}
            if _encoded_size(candidate_response) > max_payload_bytes and response["tools"]:
                response["truncated_by_size"] = True
                response["has_more"] = True
                break
            if _encoded_size(candidate_response) > max_payload_bytes:
                # Guarantee at least one item with minimal fields.
                minimal_item = {
                    "name": row["name"],
                    "summary": row["summary"],
                    "category": row["category"],
                }
                response["tools"] = [minimal_item]
                response["truncated_by_size"] = True
                response["has_more"] = True
                break
            response["tools"] = candidate_tools

        consumed = len(response["tools"])
        if not response["has_more"] and (offset + consumed) < total_count:
            response["has_more"] = True
        if response["has_more"]:
            response["next_offset"] = offset + consumed
        else:
            response["next_offset"] = None

        rendered = json.dumps(response, ensure_ascii=False, indent=2)
        return ToolResult.ok(rendered, data=response)
