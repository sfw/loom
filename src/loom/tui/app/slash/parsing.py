"""Slash command parsing utilities."""

from __future__ import annotations

import json
import re
import shlex
from typing import Any


def split_slash_args(raw: str) -> list[str]:
    """Split slash-command argument string using shell-like quoting."""
    try:
        return shlex.split(raw)
    except ValueError as e:
        try:
            tokens = shlex.split(raw, posix=False)
        except ValueError:
            tokens = split_slash_args_forgiving(raw)
            if tokens:
                return tokens
            raise ValueError(f"Invalid quoted argument syntax: {e}") from e
        normalized: list[str] = []
        for token in tokens:
            text = str(token or "")
            if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
                normalized.append(text[1:-1])
            else:
                normalized.append(text)
        return normalized


def split_slash_args_forgiving(raw: str) -> list[str]:
    """Best-effort tokenizer that never fails on unmatched quote punctuation."""
    tokens: list[str] = []
    current: list[str] = []
    quote: str | None = None
    open_quote: str | None = None
    escaped = False
    for char in str(raw or ""):
        if quote is not None:
            if escaped:
                current.append(char)
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == quote:
                quote = None
                open_quote = None
                continue
            current.append(char)
            continue

        if char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
            continue

        if char in {"'", '"'}:
            if not current:
                quote = char
                open_quote = char
            else:
                current.append(char)
            continue

        current.append(char)

    if escaped:
        current.append("\\")
    if quote is not None and open_quote is not None:
        current.insert(0, open_quote)
    if current:
        tokens.append("".join(current))
    return tokens


def split_tool_slash_args(raw: str) -> tuple[str, str]:
    """Split `/tool` args into (tool_name, raw_json_args)."""
    text = str(raw or "").strip()
    if not text:
        return "", ""
    parts = text.split(None, 1)
    tool_name = str(parts[0] or "").strip()
    json_args = str(parts[1] or "").strip() if len(parts) > 1 else ""
    return tool_name, json_args


def parse_tool_kv_value(raw_value: str) -> Any:
    """Parse one `/tool` key=value literal into a typed JSON-compatible value."""
    text = str(raw_value or "")
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    if re.fullmatch(r"[+-]?\d+", text):
        try:
            return int(text)
        except ValueError:
            pass
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", text):
        try:
            return float(text)
        except ValueError:
            pass

    if text.startswith(("{", "[", '"')) or text in {"{}", "[]"}:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return text


def parse_tool_slash_arguments(raw: str) -> tuple[dict[str, Any] | None, str]:
    """Parse `/tool` arguments from either JSON object or key=value pairs."""
    text = str(raw or "").strip()
    if not text:
        return {}, ""

    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            return None, f"Invalid /tool JSON arguments: {e}"
        if parsed is None:
            return {}, ""
        if not isinstance(parsed, dict):
            return None, "/tool arguments must be a JSON object."
        return parsed, ""

    try:
        tokens = split_slash_args(text)
    except ValueError as e:
        return None, str(e)

    parsed_pairs: dict[str, Any] = {}
    for token in tokens:
        if "=" not in token:
            return (
                None,
                (
                    f"Invalid /tool argument '{token}'. "
                    "Use key=value pairs or a JSON object."
                ),
            )
        key, raw_value = token.split("=", 1)
        key = str(key or "").strip()
        if not key:
            return None, "Invalid /tool argument key in key=value pair."
        parsed_pairs[key] = parse_tool_kv_value(raw_value)
    return parsed_pairs, ""


def strip_wrapping_quotes(value: str) -> str:
    """Remove matching wrapping quotes from a command argument."""
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text
