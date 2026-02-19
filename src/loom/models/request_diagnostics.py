"""Shared request diagnostics for model invocations.

Computes cheap, consistent size metrics for outbound LLM requests so callers
can log/emit origin + payload growth before provider dispatch.
"""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any

from loom.utils.tokens import estimate_tokens

_OVERSIZE_REQUEST_BYTES = 3_500_000
_LARGE_REQUEST_BYTES = 1_000_000


def _safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _json_bytes(value: Any) -> int:
    return len(_safe_json_dumps(value).encode("utf-8", errors="replace"))


def _content_chars(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if value is None:
        return 0
    if isinstance(value, (dict, list, tuple)):
        return len(_safe_json_dumps(value))
    return len(str(value))


def _tool_call_arg_chars(tool_calls: Any) -> int:
    if not isinstance(tool_calls, list):
        return 0
    total = 0
    for call in tool_calls:
        if not isinstance(call, dict):
            total += _content_chars(call)
            continue
        function = call.get("function")
        if isinstance(function, dict):
            total += _content_chars(function.get("arguments"))
        else:
            total += _content_chars(call.get("arguments"))
    return total


def infer_call_origin() -> str:
    """Infer caller module/function for provider-level diagnostics."""
    try:
        stack = inspect.stack(context=0)
    except Exception:
        return "unknown"

    try:
        for frame_info in stack[1:]:
            module = inspect.getmodule(frame_info.frame)
            module_name = module.__name__ if module else ""
            if module_name.startswith("loom.models."):
                continue
            if module_name:
                return f"{module_name}.{frame_info.function}"
            return f"{frame_info.filename}:{frame_info.function}"
    finally:
        del stack
    return "unknown"


@dataclass(frozen=True)
class RequestDiagnostics:
    """Compact request metrics for logging and event payloads."""

    origin: str
    request_bytes: int
    request_est_tokens: int
    message_count: int
    messages_bytes: int
    messages_chars: int
    largest_message_chars: int
    user_chars: int
    assistant_chars: int
    system_chars: int
    tool_chars: int
    other_chars: int
    assistant_tool_calls: int
    assistant_tool_arg_chars: int
    tool_count: int
    tools_bytes: int

    @property
    def is_large(self) -> bool:
        return self.request_bytes >= _LARGE_REQUEST_BYTES

    @property
    def is_oversize_risk(self) -> bool:
        return self.request_bytes >= _OVERSIZE_REQUEST_BYTES

    def to_event_payload(self) -> dict[str, Any]:
        """Serialize diagnostics for event-bus emission."""
        return {
            "origin": self.origin,
            "request_bytes": self.request_bytes,
            "request_est_tokens": self.request_est_tokens,
            "message_count": self.message_count,
            "messages_bytes": self.messages_bytes,
            "messages_chars": self.messages_chars,
            "largest_message_chars": self.largest_message_chars,
            "user_chars": self.user_chars,
            "assistant_chars": self.assistant_chars,
            "system_chars": self.system_chars,
            "tool_chars": self.tool_chars,
            "other_chars": self.other_chars,
            "assistant_tool_calls": self.assistant_tool_calls,
            "assistant_tool_arg_chars": self.assistant_tool_arg_chars,
            "tool_count": self.tool_count,
            "tools_bytes": self.tools_bytes,
            "request_size_tier": (
                "oversize_risk"
                if self.is_oversize_risk
                else "large" if self.is_large else "normal"
            ),
        }


def collect_request_diagnostics(
    *,
    messages: Any,
    tools: Any = None,
    payload: Any = None,
    origin: str = "",
) -> RequestDiagnostics:
    """Compute request diagnostics from model-call inputs."""
    message_list = messages if isinstance(messages, list) else []
    tool_list = tools if isinstance(tools, list) else []

    role_chars = {
        "user": 0,
        "assistant": 0,
        "system": 0,
        "tool": 0,
        "other": 0,
    }
    message_count = 0
    largest_message_chars = 0
    assistant_tool_calls = 0
    assistant_tool_arg_chars = 0

    for message in message_list:
        if not isinstance(message, dict):
            continue
        message_count += 1
        role = str(message.get("role", "")).strip().lower() or "other"
        content_size = _content_chars(message.get("content"))
        tool_call_size = _content_chars(message.get("tool_calls"))
        message_size = content_size + tool_call_size
        largest_message_chars = max(largest_message_chars, message_size)
        if role in role_chars:
            role_chars[role] += message_size
        else:
            role_chars["other"] += message_size

        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                assistant_tool_calls += len(tool_calls)
                assistant_tool_arg_chars += _tool_call_arg_chars(tool_calls)

    messages_json = _safe_json_dumps(message_list)
    messages_bytes = len(messages_json.encode("utf-8", errors="replace"))
    messages_chars = len(messages_json)
    tools_bytes = _json_bytes(tool_list) if tool_list else 0

    payload_obj = payload if payload is not None else {
        "messages": message_list,
        "tools": tool_list,
    }
    payload_json = _safe_json_dumps(payload_obj)
    request_bytes = len(payload_json.encode("utf-8", errors="replace"))
    request_est_tokens = estimate_tokens(payload_json)

    diag_origin = origin.strip() if isinstance(origin, str) else ""
    if not diag_origin:
        diag_origin = infer_call_origin()

    return RequestDiagnostics(
        origin=diag_origin,
        request_bytes=request_bytes,
        request_est_tokens=request_est_tokens,
        message_count=message_count,
        messages_bytes=messages_bytes,
        messages_chars=messages_chars,
        largest_message_chars=largest_message_chars,
        user_chars=role_chars["user"],
        assistant_chars=role_chars["assistant"],
        system_chars=role_chars["system"],
        tool_chars=role_chars["tool"],
        other_chars=role_chars["other"],
        assistant_tool_calls=assistant_tool_calls,
        assistant_tool_arg_chars=assistant_tool_arg_chars,
        tool_count=len(tool_list),
        tools_bytes=tools_bytes,
    )


def log_request_diagnostics(
    *,
    logger: logging.Logger,
    provider_name: str,
    model_name: str,
    operation: str,
    diagnostics: RequestDiagnostics,
) -> None:
    """Emit a one-line structured log for an outbound model request."""
    if diagnostics.is_oversize_risk:
        level = logging.WARNING
    elif diagnostics.is_large:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logger.log(
        level,
        (
            "llm_request origin=%s provider=%s model=%s op=%s "
            "request_bytes=%d request_est_tokens=%d messages=%d "
            "largest_message_chars=%d tools=%d assistant_tool_calls=%d "
            "assistant_tool_arg_chars=%d "
            "role_chars={user:%d,assistant:%d,system:%d,tool:%d,other:%d}"
        ),
        diagnostics.origin,
        provider_name,
        model_name,
        operation,
        diagnostics.request_bytes,
        diagnostics.request_est_tokens,
        diagnostics.message_count,
        diagnostics.largest_message_chars,
        diagnostics.tool_count,
        diagnostics.assistant_tool_calls,
        diagnostics.assistant_tool_arg_chars,
        diagnostics.user_chars,
        diagnostics.assistant_chars,
        diagnostics.system_chars,
        diagnostics.tool_chars,
        diagnostics.other_chars,
    )
