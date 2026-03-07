"""Parse and validate semantic compactor model responses."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any


def strip_markdown_fences(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    content = "\n".join(lines[1:])
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        start = match.start()
        try:
            candidate, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    return None


def extract_compacted_text(raw_text: str) -> tuple[str | None, str]:
    payload_text = strip_markdown_fences(str(raw_text or "").strip())
    if not payload_text:
        return None, "empty_response"

    parsed: Any
    try:
        parsed = json.loads(payload_text)
    except json.JSONDecodeError:
        parsed = extract_first_json_object(payload_text)
        if parsed is None:
            return None, "invalid_json"

    if not isinstance(parsed, dict):
        return None, "json_not_object"
    if "compressed_text" not in parsed:
        return None, "missing_compressed_text"

    compacted = parsed.get("compressed_text")
    if not isinstance(compacted, str):
        return None, "compressed_text_not_string"
    return compacted.strip(), ""


def extract_partial_compressed_text(raw_text: str) -> str | None:
    payload_text = strip_markdown_fences(str(raw_text or "").strip())
    if not payload_text:
        return None

    key_match = re.search(r'"compressed_text"\s*:\s*"', payload_text)
    if key_match is None:
        return None

    idx = key_match.end()
    chars: list[str] = []
    escaping = False
    hex_chars = "0123456789abcdefABCDEF"

    while idx < len(payload_text):
        char = payload_text[idx]
        if escaping:
            if char == "n":
                chars.append("\n")
            elif char == "r":
                chars.append("\r")
            elif char == "t":
                chars.append("\t")
            elif char == "b":
                chars.append("\b")
            elif char == "f":
                chars.append("\f")
            elif char in {'"', "\\", "/"}:
                chars.append(char)
            elif char == "u":
                digits = payload_text[idx + 1:idx + 5]
                if len(digits) < 4 or not all(c in hex_chars for c in digits):
                    break
                chars.append(chr(int(digits, 16)))
                idx += 4
            else:
                chars.append(char)
            escaping = False
            idx += 1
            continue

        if char == "\\":
            escaping = True
            idx += 1
            continue
        if char == '"':
            break
        chars.append(char)
        idx += 1

    recovered = "".join(chars).strip()
    return recovered or None


def validate_compacted_text(
    text: str,
    *,
    limit: int,
    meta_commentary_prefixes: Iterable[str],
    meta_commentary_phrases: Iterable[str],
) -> str:
    content = str(text or "").strip()
    if not content:
        return "empty_compacted_text"
    if len(content) > limit:
        return "output_exceeds_target"

    lower_head = content[:220].lower()
    for prefix in meta_commentary_prefixes:
        if lower_head.startswith(prefix):
            return "meta_commentary_prefix"
    for phrase in meta_commentary_phrases:
        if phrase in lower_head:
            return "meta_commentary_phrase"
    return ""


def normalize_whitespace(text: str) -> str:
    """Lossless fallback compaction: normalize repeated whitespace."""
    if not text:
        return text
    lines = [line.rstrip() for line in text.splitlines()]
    collapsed: list[str] = []
    blank = False
    for line in lines:
        if line:
            collapsed.append(" ".join(line.split()))
            blank = False
        elif not blank:
            collapsed.append("")
            blank = True
    return "\n".join(collapsed).strip()
