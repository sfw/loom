"""Launch/workspace utility helpers for process runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loom.processes.run_workspace import (
    extract_run_folder_slug,
    fallback_process_run_folder_name,
    is_low_quality_run_folder_slug,
    normalize_process_run_workspace_selection,
    run_goal_for_folder_name,
    slugify_process_run_folder,
)

from ..constants import _RUN_GOAL_FILE_CONTENT_MAX_CHARS

__all__ = [
    "extract_run_folder_slug",
    "expand_run_goal_file_input",
    "fallback_process_run_folder_name",
    "is_low_quality_run_folder_slug",
    "normalize_process_run_workspace_selection",
    "resolve_run_goal_file_path",
    "run_goal_for_folder_name",
    "slugify_process_run_folder",
    "truncate_run_goal_file_content",
]


def truncate_run_goal_file_content(content: str) -> tuple[str, bool]:
    """Bound file-input size for `/run` goal context and synthesis prompts."""
    if len(content) <= _RUN_GOAL_FILE_CONTENT_MAX_CHARS:
        return content, False
    return content[:_RUN_GOAL_FILE_CONTENT_MAX_CHARS].rstrip(), True


def resolve_run_goal_file_path(self, raw_path: str) -> Path | None:
    """Resolve a `/run` file-input token to a workspace-local file path."""
    token = str(raw_path or "").strip()
    if not token:
        return None

    candidate = Path(token).expanduser()
    if not candidate.is_absolute():
        candidate = self._workspace / candidate
    try:
        resolved = candidate.resolve()
        workspace_root = self._workspace.resolve()
    except OSError:
        return None
    if not resolved.is_file():
        return None
    try:
        resolved.relative_to(workspace_root)
    except ValueError:
        return None
    return resolved


def expand_run_goal_file_input(
    self,
    goal_tokens: list[str],
) -> tuple[str, str, dict[str, Any], str | None]:
    """Expand optional `/run` file shorthand into goal/context payloads."""
    goal_text = " ".join(str(token or "").strip() for token in goal_tokens).strip()
    if not goal_tokens:
        return goal_text, goal_text, {}, None

    first = str(goal_tokens[0] or "").strip()
    first_path_token = first[1:] if first.startswith("@") else first
    should_treat_as_file = bool(first.startswith("@") or len(goal_tokens) == 1)
    if not should_treat_as_file:
        return goal_text, goal_text, {}, None

    resolved = self._resolve_run_goal_file_path(first_path_token)
    if resolved is None:
        if first.startswith("@"):
            return (
                goal_text,
                goal_text,
                {},
                f"Run goal file not found (or outside workspace): {first_path_token}",
            )
        return goal_text, goal_text, {}, None

    try:
        raw_content = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return (
            goal_text,
            goal_text,
            {},
            f"Failed reading run goal file '{first_path_token}': {e}",
        )

    content, truncated = self._truncate_run_goal_file_content(raw_content)
    workspace_root = self._workspace.resolve()
    try:
        file_label = str(resolved.relative_to(workspace_root))
    except ValueError:
        file_label = str(resolved)

    user_goal = ""
    if first.startswith("@") and len(goal_tokens) > 1:
        user_goal = " ".join(
            str(token or "").strip() for token in goal_tokens[1:]
        ).strip()

    if user_goal:
        execution_goal = user_goal
        preface = (
            f"{user_goal}\n\n"
            f"Supplemental task specification file: {file_label}\n"
            "Use the file content below as authoritative detail."
        )
    else:
        execution_goal = file_label
        preface = (
            f"Use the following task specification from file "
            f"'{file_label}' as the primary goal."
        )

    truncated_note = ""
    if truncated:
        truncated_note = (
            f"\n\n[File content truncated to first "
            f"{_RUN_GOAL_FILE_CONTENT_MAX_CHARS} characters.]"
        )
    synthesis_goal = (
        f"{preface}\n\n"
        f"--- BEGIN FILE: {file_label} ---\n"
        f"{content}\n"
        f"--- END FILE: {file_label} ---"
        f"{truncated_note}"
    )
    return (
        execution_goal,
        synthesis_goal,
        {
            "run_goal_file_input": {
                "path": file_label,
                "content": content,
                "truncated": truncated,
                "max_chars": _RUN_GOAL_FILE_CONTENT_MAX_CHARS,
            },
        },
        None,
    )
