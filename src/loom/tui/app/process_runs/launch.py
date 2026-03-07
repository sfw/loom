"""Launch/workspace utility helpers for process runs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..constants import _RUN_GOAL_FILE_CONTENT_MAX_CHARS


def slugify_process_run_folder(value: str, *, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower())
    slug = slug.strip("-")
    if not slug:
        return ""
    if len(slug) > max_len:
        slug = slug[:max_len].strip("-")
    return slug


def run_goal_for_folder_name(goal: str) -> str:
    text = " ".join(str(goal or "").split()).strip()
    if not text:
        return ""
    prefixes = (
        r"^(?:please\s+)?i\s+need\s+you\s+to\s+",
        r"^(?:please\s+)?can\s+you\s+",
        r"^(?:please\s+)?could\s+you\s+",
        r"^(?:please\s+)?help\s+me\s+(?:to\s+)?",
        r"^the\s+user\s+wants(?:\s+me)?\s+to\s+",
    )
    for pattern in prefixes:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return text[:240]


def extract_run_folder_slug(response_text: str) -> str:
    text = str(response_text or "").strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip().strip("`").strip("\"'")
    match = re.search(r"[a-z0-9]+(?:-[a-z0-9]+)+", first_line.lower())
    candidate = match.group(0) if match else first_line
    return slugify_process_run_folder(candidate, max_len=48)


def is_low_quality_run_folder_slug(slug: str) -> bool:
    value = str(slug or "").strip().lower()
    if not value:
        return True
    tokens = [part for part in value.split("-") if part]
    if len(tokens) < 2:
        return True
    if tokens[:3] == ["the", "user", "wants"]:
        return True
    if tokens[:4] == ["i", "need", "you", "to"]:
        return True

    banned_tokens = {
        "the",
        "user",
        "wants",
        "i",
        "need",
        "you",
        "to",
        "folder",
        "name",
        "kebab",
        "case",
        "run",
        "process",
        "task",
        "request",
        "prompt",
        "query",
        "for",
        "a",
        "pr",
    }
    banned_hits = sum(1 for token in tokens if token in banned_tokens)
    return banned_hits >= max(2, (len(tokens) + 1) // 2)


def fallback_process_run_folder_name(process_name: str, goal: str) -> str:
    merged = f"{process_name} {goal}".strip().lower()
    tokens = re.findall(r"[a-z0-9]+", merged)
    base = "-".join(tokens[:6])
    slug = slugify_process_run_folder(base)
    return slug or "process-run"


def normalize_process_run_workspace_selection(raw_value: str) -> str:
    """Normalize a user-selected run workspace relative path."""
    raw = str(raw_value or "").strip()
    if not raw:
        return ""
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        raise ValueError("Folder must be relative to workspace root.")
    parts: list[str] = []
    for part in candidate.parts:
        clean = str(part).strip()
        if not clean or clean == ".":
            continue
        if clean == "..":
            raise ValueError("Folder cannot traverse outside workspace root.")
        parts.append(clean)
    return "/".join(parts)


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
