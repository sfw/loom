"""Shared helpers for process-run working folder naming and provisioning."""

from __future__ import annotations

import re
from pathlib import Path


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


def next_available_process_run_folder_name(workspace_root: Path, base_slug: str) -> str:
    """Return the first non-existing run folder name for a base slug."""
    root = Path(workspace_root).expanduser().resolve()
    slug = slugify_process_run_folder(base_slug) or "process-run"
    for suffix in range(1, 1000):
        candidate_name = slug if suffix == 1 else f"{slug}-{suffix}"
        candidate = root / candidate_name
        if not candidate.exists():
            return candidate_name
    return slug


def materialize_process_run_workspace(workspace_root: Path, relative_path: str) -> Path:
    """Create or reuse a selected run workspace path under the workspace root."""
    root = Path(workspace_root).expanduser().resolve()
    clean = str(relative_path or "").strip()
    if not clean:
        return root

    candidate = (root / clean).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise ValueError("Folder must stay inside workspace root.") from e

    if candidate.exists():
        if candidate.is_dir():
            return candidate
        raise ValueError("Selected folder path exists but is not a directory.")

    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def provision_scoped_run_workspace(
    workspace_root: Path,
    *,
    process_name: str,
    goal: str,
) -> tuple[Path, str]:
    """Create the next available scoped working folder under a workspace root."""
    root = Path(workspace_root).expanduser().resolve()
    base_slug = fallback_process_run_folder_name(process_name, goal)
    candidate_name = next_available_process_run_folder_name(root, base_slug)
    candidate = root / candidate_name
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate, candidate_name
