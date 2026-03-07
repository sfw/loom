"""Files panel ingestion and changed-path normalization helpers."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from loom.cowork.session import CoworkTurn, ToolCallEvent
from loom.tui.widgets import FilesChangedPanel


def _now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _resolve_workspace_file(self, path: Path) -> Path | None:
    """Resolve a selected file and ensure it remains inside workspace."""
    try:
        resolved = path.resolve()
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

def _normalize_files_changed_paths(self, raw_paths: object) -> list[str]:
    """Normalize tool result file paths into display-safe relative paths."""
    if not isinstance(raw_paths, (list, tuple, set)):
        return []
    normalized: list[str] = []
    workspace_root: Path | None = None
    try:
        workspace_root = self._workspace.resolve()
    except OSError:
        workspace_root = None
    for item in raw_paths:
        text = str(item or "").strip()
        if not text:
            continue
        if text.startswith("(") and text.endswith(")") and "files " in text:
            # delegate_task summary markers are not concrete file paths.
            continue
        if " -> " in text:
            if text not in normalized:
                normalized.append(text)
            continue
        candidate = Path(text).expanduser()
        if candidate.is_absolute() and workspace_root is not None:
            try:
                candidate = candidate.resolve().relative_to(workspace_root)
            except Exception:
                candidate = Path(text)
        rel = str(candidate).strip()
        if rel and rel not in normalized:
            normalized.append(rel)
    return normalized

def _summary_files_changed_markers(raw_paths: object) -> list[tuple[str, str]]:
    """Extract fallback summary markers from files_changed payloads."""
    if not isinstance(raw_paths, (list, tuple, set)):
        return []
    markers: list[tuple[str, str]] = []
    for item in raw_paths:
        text = str(item or "").strip()
        lower = text.lower()
        if not text.startswith("(") or not text.endswith(")") or "file" not in lower:
            continue
        op = "modify"
        if "created" in lower:
            op = "create"
        elif "deleted" in lower:
            op = "delete"
        marker = (op, text)
        if marker not in markers:
            markers.append(marker)
    return markers

def _operation_hint_for_tool(tool_name: str) -> str:
    name = str(tool_name or "").strip()
    if name == "write_file":
        return "create"
    if name == "edit_file":
        return "modify"
    if name == "delete_file":
        return "delete"
    if name == "move_file":
        return "rename"
    return "modify"

def _ingest_files_panel_from_paths(
    self,
    raw_paths: object,
    *,
    operation_hint: str = "modify",
) -> int:
    """Append file rows into the Files panel with dedupe + bounded history."""
    paths = self._normalize_files_changed_paths(raw_paths)
    path_entries: list[tuple[str, str]] = []
    for path in paths:
        op = operation_hint
        if " -> " in path:
            op = "rename"
        path_entries.append((op, path))
    if not path_entries:
        path_entries = self._summary_files_changed_markers(raw_paths)
    if not path_entries:
        return 0
    try:
        panel = self.query_one("#files-panel", FilesChangedPanel)
    except Exception:
        return 0

    now = time.monotonic()
    for key, seen_at in list(self._files_panel_recent_ops.items()):
        if (now - seen_at) > self._files_panel_dedupe_window_seconds:
            self._files_panel_recent_ops.pop(key, None)

    accepted: list[dict] = []
    for op, path in path_entries:
        dedupe_key = f"{op}:{path}"
        seen_at = self._files_panel_recent_ops.get(dedupe_key, 0.0)
        if (now - seen_at) < self._files_panel_dedupe_window_seconds:
            continue
        self._files_panel_recent_ops[dedupe_key] = now
        accepted.append({
            "operation": op,
            "path": path,
            "timestamp": _now_str(),
        })
    if not accepted:
        return 0

    panel.update_files(accepted)
    max_rows = self._tui_files_panel_max_rows()
    all_entries = getattr(panel, "_all_entries", None)
    if isinstance(all_entries, list) and len(all_entries) > max_rows:
        overflow = len(all_entries) - max_rows
        del all_entries[:overflow]
        try:
            panel._refresh_table()
        except Exception:
            pass
    return len(accepted)

def _ingest_files_panel_from_tool_call_event(self, event: ToolCallEvent) -> int:
    """Consume one completed tool call into the Files panel, if relevant."""
    result = getattr(event, "result", None)
    if result is None or not bool(getattr(result, "success", False)):
        return 0

    count = self._ingest_files_panel_from_paths(
        getattr(result, "files_changed", []),
        operation_hint=self._operation_hint_for_tool(event.name),
    )

    fallback_entries: list[str] = []
    if count <= 0:
        if event.name in {"write_file", "edit_file", "delete_file"}:
            path = str(
                event.args.get("path", event.args.get("file_path", "")) or "",
            ).strip()
            if path:
                fallback_entries.append(path)
        elif event.name == "move_file":
            src = str(event.args.get("source", "") or "").strip()
            dst = str(event.args.get("destination", "") or "").strip()
            if src and dst:
                fallback_entries.append(f"{src} -> {dst}")
        if fallback_entries:
            count += self._ingest_files_panel_from_paths(
                fallback_entries,
                operation_hint=self._operation_hint_for_tool(event.name),
            )

    if event.name == "edit_file":
        output = str(getattr(result, "output", "") or "")
        marker = "--- a/"
        idx = output.find(marker)
        if idx != -1:
            try:
                panel = self.query_one("#files-panel", FilesChangedPanel)
                panel.show_diff(output[idx:])
            except Exception:
                pass

    return count

def _update_files_panel(self, turn: CoworkTurn) -> None:
    """Update the Files Changed panel from tool call events."""
    changed_count = 0
    refresh_workspace = False
    for tc in turn.tool_calls:
        if not tc.result or not tc.result.success:
            continue
        if self._is_mutating_tool(tc.name):
            refresh_workspace = True
        changed_count += self._ingest_files_panel_from_tool_call_event(tc)
    if changed_count > 0:
        count = int(changed_count)
        s = "s" if count != 1 else ""
        self.notify(
            f"{count} file{s} changed", timeout=3,
        )
    if refresh_workspace:
        self._request_workspace_refresh("turn-summary")
