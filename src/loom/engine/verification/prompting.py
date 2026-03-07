"""Tier-2 verifier prompt assembly helpers."""

from __future__ import annotations

from typing import Any

from loom.state.task_state import Subtask


def build_verifier_prompt(
    prompts,
    *,
    subtask: Subtask,
    result_summary: str,
    tool_calls_formatted: str,
    llm_rules: list | None = None,
    phase_scope_default: str = "current_phase",
) -> str:
    return prompts.build_verifier_prompt(
        subtask=subtask,
        result_summary=result_summary,
        tool_calls_formatted=tool_calls_formatted,
        llm_rules=llm_rules,
        phase_scope_default=phase_scope_default,
    )


def phase_scope_default(config: Any) -> str:
    value = str(getattr(config, "phase_scope_default", "current_phase") or "")
    normalized = value.strip().lower()
    if normalized in {"current_phase", "global"}:
        return normalized
    return "current_phase"


def expected_verifier_response_keys(prompts) -> list[str]:
    process = getattr(prompts, "process", None)
    if process is None:
        return ["passed"]
    getter = getattr(process, "verifier_required_response_fields", None)
    if callable(getter):
        keys = getter()
        if isinstance(keys, list):
            normalized = [
                str(item).strip()
                for item in keys
                if str(item).strip()
            ]
            if "passed" not in normalized:
                normalized.insert(0, "passed")
            return normalized
    return ["passed"]


def verifier_metadata_fields(prompts) -> list[str]:
    process = getattr(prompts, "process", None)
    if process is None:
        return []
    metadata_getter = getattr(process, "verifier_metadata_fields", None)
    if not callable(metadata_getter):
        return []
    metadata_fields = metadata_getter()
    if not isinstance(metadata_fields, list):
        return []
    return [
        str(item).strip()
        for item in metadata_fields
        if str(item).strip()
    ]


def build_repair_prompt(
    *,
    expected_keys: list[str],
    raw_text: str,
    metadata_fields: list[str] | None = None,
) -> str:
    expected_display = ", ".join(f'"{key}"' for key in expected_keys)
    metadata_hint = ""
    names = ", ".join(
        str(item).strip()
        for item in (metadata_fields or [])
        if str(item).strip()
    )
    if names:
        metadata_hint = (
            "\nWhen inferable, include these metadata keys: "
            f"{names}."
        )
    return (
        "Repair the following verifier output into a strict JSON object with keys:\n"
        "{"
        + expected_display
        + "}\n"
        "Use only values directly inferable from the text. If unknown, use empty "
        "strings, [] or {}. Respond with JSON only."
        + metadata_hint
        + "\n\n"
        "RAW OUTPUT:\n"
        f"{raw_text}"
    )
