"""Slash `/tool` helper methods."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any


def tool_name_inventory(self) -> list[str]:
    """Return sorted available tool names."""
    try:
        names = self._tools.list_tools()
    except Exception:
        names = []
    normalized = {
        str(name or "").strip()
        for name in names
        if str(name or "").strip()
    }
    return sorted(normalized)


def tool_description(self, tool_name: str) -> str:
    """Return compact tool description text."""
    tool = self._tools.get(str(tool_name or "").strip())
    if tool is None:
        return ""
    return " ".join(str(getattr(tool, "description", "") or "").split())


def tool_parameters_schema(self, tool_name: str) -> dict[str, Any]:
    """Return JSON-schema-like parameters object for a tool when available."""
    tool = self._tools.get(str(tool_name or "").strip())
    if tool is None:
        return {}
    raw = getattr(tool, "parameters", {})
    if isinstance(raw, dict):
        return raw
    try:
        schema = tool.schema()
    except Exception:
        return {}
    payload = schema.get("parameters", {}) if isinstance(schema, dict) else {}
    return payload if isinstance(payload, dict) else {}


def tool_argument_lists(parameters: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (required, optional) argument names from tool parameters schema."""
    required_raw = parameters.get("required", []) if isinstance(parameters, dict) else []
    required: list[str] = []
    for item in required_raw if isinstance(required_raw, list) else []:
        name = str(item or "").strip()
        if name:
            required.append(name)

    properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    optional: list[str] = []
    if isinstance(properties, dict):
        for key in properties.keys():
            name = str(key or "").strip()
            if name and name not in required:
                optional.append(name)
    return required, optional


def tool_argument_placeholder(schema: dict[str, Any]) -> str:
    """Return a JSON literal placeholder for a schema property."""
    if not isinstance(schema, dict):
        return '"value"'
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        try:
            return json.dumps(enum_values[0], ensure_ascii=False)
        except (TypeError, ValueError):
            return '"value"'
    schema_type = schema.get("type", "")
    if isinstance(schema_type, list):
        schema_type = next(
            (str(item or "").strip() for item in schema_type if str(item or "").strip()),
            "",
        )
    schema_type = str(schema_type or "").strip().lower()
    if schema_type in {"integer", "number"}:
        return "0"
    if schema_type == "boolean":
        return "false"
    if schema_type == "array":
        return "[]"
    if schema_type == "object":
        return "{}"
    if schema_type == "null":
        return "null"
    return '"value"'


def tool_argument_example(self, tool_name: str, *, max_fields: int = 4) -> str:
    """Build a compact JSON example payload for `/tool` hints."""
    parameters = self._tool_parameters_schema(tool_name)
    properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    if not isinstance(properties, dict) or not properties:
        return "{}"
    required, optional = self._tool_argument_lists(parameters)
    ordered = [*required, *[name for name in optional if name not in required]]
    if not ordered:
        ordered = [
            str(key or "").strip()
            for key in properties.keys()
            if str(key or "").strip()
        ]
    if not ordered:
        return "{}"
    pairs: list[str] = []
    truncated = False
    for index, key in enumerate(ordered):
        if index >= max_fields:
            truncated = True
            break
        schema = properties.get(key, {})
        placeholder = self._tool_argument_placeholder(
            schema if isinstance(schema, dict) else {},
        )
        pairs.append(f"{json.dumps(key)}: {placeholder}")
    if truncated:
        pairs.append('"...": "..."')
    return "{ " + ", ".join(pairs) + " }"


def tool_argument_summary(self, tool_name: str) -> tuple[str, str]:
    """Return required/optional argument summary strings for UI hints."""
    parameters = self._tool_parameters_schema(tool_name)
    required, optional = self._tool_argument_lists(parameters)
    required_text = ", ".join(required) if required else "(none)"
    optional_text = ", ".join(optional[:8]) if optional else "(none)"
    if optional and len(optional) > 8:
        optional_text += ", ..."
    return required_text, optional_text


async def execute_slash_tool_command(
    self,
    resolved_tool_name: str,
    tool_args: dict[str, Any],
) -> None:
    """Execute `/tool` via run_tool and render output in chat."""
    chat = self.query_one("#chat-log")
    auth_context = (
        getattr(self._session, "_auth_context", None)
        if self._session is not None
        else None
    )
    start = time.monotonic()
    result = await self._tools.execute(
        "run_tool",
        {"name": resolved_tool_name, "arguments": tool_args},
        workspace=self._workspace,
        scratch_dir=self._cowork_scratch_dir(),
        auth_context=auth_context,
    )
    elapsed_ms = max(0, int((time.monotonic() - start) * 1000))
    chat.add_tool_call(
        "run_tool",
        {"name": resolved_tool_name, "arguments": tool_args},
        tool_call_id=f"slash-tool-{uuid.uuid4().hex[:12]}",
        success=bool(getattr(result, "success", False)),
        elapsed_ms=elapsed_ms,
        output=str(getattr(result, "output", "") or ""),
        error=str(getattr(result, "error", "") or ""),
    )
    if bool(getattr(result, "success", False)):
        changed = self._ingest_files_panel_from_paths(
            getattr(result, "files_changed", []),
            operation_hint=self._operation_hint_for_tool(resolved_tool_name),
        )
        if changed > 0:
            s = "s" if changed != 1 else ""
            self.notify(f"{changed} file{s} changed", timeout=3)
        if self._is_mutating_tool(resolved_tool_name):
            self._request_workspace_refresh("slash-tool")
