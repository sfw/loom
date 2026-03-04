"""AskUser tool: lets the model ask the developer for clarification.

When the model needs input (a decision, clarification, or preference),
it calls this tool.  The cowork CLI will display the question and wait
for the user's response, which becomes the tool result.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from loom.tools.registry import Tool, ToolContext, ToolResult

ASK_USER_QUESTION_TYPES = frozenset({"free_text", "single_choice", "multi_choice"})
ASK_USER_URGENCY = frozenset({"low", "normal", "high"})
_DEFAULT_QUESTION_TYPE = "free_text"
_DEFAULT_URGENCY = "normal"
_MAX_OPTIONS = 25
_MAX_QUESTION_CHARS = 1200
_MAX_CONTEXT_NOTE_CHARS = 400


def _normalize_text(value: Any, *, max_chars: int = 0) -> str:
    text = str(value or "").strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


def _normalize_option_id(label: str, *, fallback_index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    if slug:
        return slug[:48]
    digest = hashlib.sha1(label.encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"option-{fallback_index}-{digest}"


def _normalize_options(raw_options: Any) -> tuple[list[dict[str, str]], list[str]]:
    entries: list[Any] = []
    if isinstance(raw_options, list):
        entries = list(raw_options)
    elif isinstance(raw_options, tuple):
        entries = list(raw_options)
    elif isinstance(raw_options, dict):
        for key, value in raw_options.items():
            key_text = _normalize_text(key, max_chars=80).lower()
            if isinstance(value, dict):
                item = dict(value)
                if key_text and not str(item.get("id", "")).strip():
                    item["id"] = key_text
                entries.append(item)
            else:
                label = _normalize_text(value, max_chars=300)
                if label:
                    entries.append({
                        "id": key_text,
                        "label": label,
                    })
    elif isinstance(raw_options, str):
        text = _normalize_text(raw_options, max_chars=2400)
        if text:
            split_lines = [
                re.sub(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)", "", line).strip()
                for line in re.split(r"[\r\n]+", text)
                if line.strip()
            ]
            lines = [line for line in split_lines if line]
            if len(lines) >= 2:
                entries = lines
            else:
                comma_parts = [part.strip() for part in text.split(",") if part.strip()]
                if len(comma_parts) >= 2:
                    entries = comma_parts

    structured: list[dict[str, str]] = []
    labels: list[str] = []
    if not entries:
        return structured, labels

    seen_ids: set[str] = set()
    for idx, raw in enumerate(entries[:_MAX_OPTIONS], start=1):
        if isinstance(raw, str):
            label = _normalize_text(raw, max_chars=300)
            if not label:
                continue
            option_id = _normalize_option_id(label, fallback_index=idx)
            description = ""
        elif isinstance(raw, dict):
            label = _normalize_text(
                raw.get(
                    "label",
                    raw.get(
                        "title",
                        raw.get(
                            "text",
                            raw.get(
                                "name",
                                raw.get(
                                    "value",
                                    raw.get("id", ""),
                                ),
                            ),
                        ),
                    ),
                ),
                max_chars=300,
            )
            if not label:
                continue
            option_id = _normalize_text(
                raw.get(
                    "id",
                    raw.get(
                        "key",
                        raw.get("slug", ""),
                    ),
                ),
                max_chars=80,
            ).lower()
            if not option_id:
                option_id = _normalize_option_id(label, fallback_index=idx)
            description = _normalize_text(
                raw.get(
                    "description",
                    raw.get("detail", raw.get("help", raw.get("hint", ""))),
                ),
                max_chars=300,
            )
        else:
            continue

        base = option_id
        dedupe = 2
        while option_id in seen_ids:
            option_id = f"{base}-{dedupe}"
            dedupe += 1
        seen_ids.add(option_id)
        structured.append({
            "id": option_id,
            "label": label,
            "description": description,
        })
        labels.append(label)

    return structured, labels


def normalize_ask_user_args(args: dict[str, Any]) -> dict[str, Any]:
    """Normalize ask_user arguments to a stable v2 payload.

    Backward compatibility:
    - Legacy options list[str] is retained via `legacy_options`.
    - If options exist and question_type is omitted, defaults to single_choice.
    """
    payload = dict(args or {})
    question = _normalize_text(payload.get("question", ""), max_chars=_MAX_QUESTION_CHARS)
    if not question:
        question = "Please provide clarification."

    raw_options = payload.get("options", [])
    if raw_options in (None, "", []):
        for alias in ("choices", "responses", "candidates", "selection_options"):
            candidate = payload.get(alias)
            if candidate not in (None, "", []):
                raw_options = candidate
                break
    options, legacy_options = _normalize_options(raw_options)
    requested_type = _normalize_text(
        payload.get("question_type", payload.get("response_type", payload.get("type", ""))),
        max_chars=40,
    ).lower()
    if requested_type not in ASK_USER_QUESTION_TYPES:
        requested_type = "single_choice" if options else _DEFAULT_QUESTION_TYPE

    allow_custom_response = bool(payload.get("allow_custom_response", True))
    urgency = _normalize_text(payload.get("urgency", ""), max_chars=32).lower()
    if urgency not in ASK_USER_URGENCY:
        urgency = _DEFAULT_URGENCY

    min_selections = payload.get("min_selections", 0)
    max_selections = payload.get("max_selections", len(options) if options else 0)
    try:
        min_sel = int(min_selections)
    except (TypeError, ValueError):
        min_sel = 0
    try:
        max_sel = int(max_selections)
    except (TypeError, ValueError):
        max_sel = len(options) if options else 0

    if requested_type == "multi_choice":
        option_count = len(options)
        if option_count == 0:
            min_sel = 0
            max_sel = 0
        else:
            min_sel = max(0, min(min_sel, option_count))
            if max_sel <= 0:
                max_sel = option_count
            max_sel = max(min_sel, min(max_sel, option_count))
    else:
        min_sel = 0
        max_sel = 0

    default_option_id = _normalize_text(payload.get("default_option_id", ""), max_chars=80).lower()
    if default_option_id and default_option_id not in {opt["id"] for opt in options}:
        default_option_id = ""

    return {
        "question": question,
        "question_type": requested_type,
        "options": options,
        "legacy_options": legacy_options,
        "allow_custom_response": allow_custom_response,
        "min_selections": min_sel,
        "max_selections": max_sel,
        "context_note": _normalize_text(
            payload.get("context_note", ""),
            max_chars=_MAX_CONTEXT_NOTE_CHARS,
        ),
        "urgency": urgency,
        "default_option_id": default_option_id,
    }


def ask_user_choice_labels(normalized: dict[str, Any]) -> list[str]:
    if not isinstance(normalized, dict):
        return []
    options = normalized.get("options", [])
    if not isinstance(options, list):
        return []
    labels: list[str] = []
    for opt in options:
        if not isinstance(opt, dict):
            continue
        label = _normalize_text(opt.get("label", ""), max_chars=300)
        if label:
            labels.append(label)
    return labels


class AskUserTool(Tool):
    """Ask the user a question and return their response."""

    name = "ask_user"
    description = (
        "Ask the developer a question when you need clarification, a decision, "
        "or additional information. The question will be displayed and the "
        "developer's response returned. Use this instead of guessing. "
        "This tool is only valid on interactive TUI execution surfaces."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the developer.",
            },
            "question_type": {
                "type": "string",
                "enum": ["free_text", "single_choice", "multi_choice"],
                "description": "Expected response mode for the question.",
            },
            "options": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "label": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["id", "label"],
                        },
                    ],
                },
                "description": "Optional list of options (legacy list[str] remains supported).",
            },
            "allow_custom_response": {
                "type": "boolean",
                "description": "Allow free-form response beyond provided options.",
            },
            "min_selections": {
                "type": "integer",
                "description": "Minimum selections for multi_choice questions.",
            },
            "max_selections": {
                "type": "integer",
                "description": "Maximum selections for multi_choice questions.",
            },
            "context_note": {
                "type": "string",
                "description": "Short note explaining why this clarification is needed.",
            },
            "urgency": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "description": "Relative urgency of the clarification request.",
            },
            "default_option_id": {
                "type": "string",
                "description": "Default option id for timeout-default policies.",
            },
        },
        "required": ["question"],
    }

    @property
    def supported_execution_surfaces(self) -> tuple[str, ...]:
        return ("tui",)

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Execute is handled specially by the cowork session.

        The cowork CLI intercepts this tool call, displays the question,
        waits for user input, and returns it as the result.  If we get
        here (e.g. in non-interactive mode), return a placeholder.
        """
        normalized = normalize_ask_user_args(args)
        question = str(normalized.get("question", "") or "")
        choice_labels = ask_user_choice_labels(normalized)

        # In non-interactive mode, return the question as a prompt
        # The caller is expected to intercept this tool call
        formatted = f"QUESTION: {question}"
        if choice_labels:
            formatted += "\nOptions: " + ", ".join(choice_labels)

        return ToolResult(
            success=True,
            output=formatted,
            data={
                "question": question,
                "question_type": normalized["question_type"],
                "options": list(normalized["legacy_options"]),
                "options_v2": list(normalized["options"]),
                "allow_custom_response": normalized["allow_custom_response"],
                "min_selections": normalized["min_selections"],
                "max_selections": normalized["max_selections"],
                "context_note": normalized["context_note"],
                "urgency": normalized["urgency"],
                "default_option_id": normalized["default_option_id"],
                "awaiting_input": True,
            },
        )
