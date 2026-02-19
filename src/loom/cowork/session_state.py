"""Session state: always-in-context structured metadata for cowork.

This is the cowork equivalent of task mode's Layer 1 YAML state.
It tracks files touched, key decisions, current focus, and errors
— all extracted mechanically from tool results, no LLM call needed.

The session state stays in the system prompt at all times so the model
never loses track of what has happened, even when old messages fall out
of the context window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import yaml


@dataclass
class FileTouch:
    path: str
    action: str  # created | edited | deleted | read
    turn: int


@dataclass
class SessionState:
    """Structured session metadata, always injected into system prompt."""

    session_id: str = ""
    workspace: str = ""
    model_name: str = ""
    turn_count: int = 0
    total_tokens: int = 0

    files_touched: list[FileTouch] = field(default_factory=list)
    key_decisions: list[str] = field(default_factory=list)
    current_focus: str = ""
    errors_resolved: list[str] = field(default_factory=list)
    # UI-only metadata (not injected into prompts), e.g. restored TUI tab state.
    ui_state: dict = field(default_factory=dict)

    # Pruning limits
    MAX_FILES: int = 20
    MAX_DECISIONS: int = 10
    MAX_ERRORS: int = 5

    def record_file(self, path: str, action: str, turn: int) -> None:
        """Record a file being touched."""
        self.files_touched.append(FileTouch(path=path, action=action, turn=turn))
        if len(self.files_touched) > self.MAX_FILES:
            self.files_touched = self.files_touched[-self.MAX_FILES:]

    def record_decision(self, decision: str, turn: int) -> None:
        """Record a key decision."""
        entry = f"{decision} (turn {turn})"
        self.key_decisions.append(entry)
        if len(self.key_decisions) > self.MAX_DECISIONS:
            self.key_decisions = self.key_decisions[-self.MAX_DECISIONS:]

    def record_error(self, description: str, turn: int) -> None:
        """Record an error (not yet resolved)."""
        entry = f"[UNRESOLVED] {description} (turn {turn})"
        self.errors_resolved.append(entry)
        if len(self.errors_resolved) > self.MAX_ERRORS:
            self.errors_resolved = self.errors_resolved[-self.MAX_ERRORS:]

    def record_error_resolved(self, description: str, turn: int) -> None:
        """Record an error that was resolved."""
        entry = f"{description} (turn {turn})"
        self.errors_resolved.append(entry)
        if len(self.errors_resolved) > self.MAX_ERRORS:
            self.errors_resolved = self.errors_resolved[-self.MAX_ERRORS:]

    def set_focus(self, focus: str) -> None:
        """Update the current focus from the user's most recent message."""
        self.current_focus = _normalize_inline(focus)

    def to_yaml(self) -> str:
        """Render as compact YAML for system prompt injection."""
        data: dict = {
            "session": {
                "id": self.session_id,
                "workspace": self.workspace,
                "model": self.model_name,
                "turn_count": self.turn_count,
                "total_tokens": self.total_tokens,
            },
        }

        if self.files_touched:
            data["files_touched"] = [
                {"path": f.path, "action": f.action, "turn": f.turn}
                for f in self.files_touched
            ]

        if self.key_decisions:
            data["key_decisions"] = self.key_decisions

        if self.current_focus:
            data["current_focus"] = self.current_focus

        if self.errors_resolved:
            data["errors_resolved"] = self.errors_resolved

        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return {
            "session_id": self.session_id,
            "workspace": self.workspace,
            "model_name": self.model_name,
            "turn_count": self.turn_count,
            "total_tokens": self.total_tokens,
            "files_touched": [
                {"path": f.path, "action": f.action, "turn": f.turn}
                for f in self.files_touched
            ],
            "key_decisions": self.key_decisions,
            "current_focus": self.current_focus,
            "errors_resolved": self.errors_resolved,
            "ui_state": self.ui_state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SessionState:
        """Reconstruct from stored dict."""
        state = cls(
            session_id=data.get("session_id", ""),
            workspace=data.get("workspace", ""),
            model_name=data.get("model_name", ""),
            turn_count=data.get("turn_count", 0),
            total_tokens=data.get("total_tokens", 0),
            current_focus=data.get("current_focus", ""),
            key_decisions=data.get("key_decisions", []),
            errors_resolved=data.get("errors_resolved", []),
            ui_state=(
                data.get("ui_state", {})
                if isinstance(data.get("ui_state"), dict)
                else {}
            ),
        )
        for f in data.get("files_touched", []):
            if isinstance(f, dict) and "path" in f and "action" in f:
                state.files_touched.append(
                    FileTouch(
                        path=f.get("path", ""),
                        action=f.get("action", ""),
                        turn=f.get("turn", 0),
                    )
                )
        return state

    @classmethod
    def from_json(cls, raw: str | None) -> SessionState:
        """Reconstruct from a JSON string (from database)."""
        if not raw:
            return cls()
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return cls()
        if not isinstance(data, dict):
            return cls()
        return cls.from_dict(data)


def extract_state_from_tool_events(
    state: SessionState,
    turn_number: int,
    tool_events: list,
) -> None:
    """Update session state from tool call events in a turn.

    Mechanically extracts file touches, errors, and git commits
    from tool results.  No LLM call needed.
    """
    for event in tool_events:
        name = event.name
        args = event.args
        result = event.result

        if name in ("write_file", "edit_file"):
            path = args.get("path", args.get("file_path", ""))
            if path:
                action = "created" if name == "write_file" else "edited"
                state.record_file(path, action, turn_number)

        elif name == "delete_file":
            path = args.get("path", "")
            if path:
                state.record_file(path, "deleted", turn_number)

        elif name == "read_file":
            path = args.get("path", args.get("file_path", ""))
            if path:
                state.record_file(path, "read", turn_number)

        elif name == "git_command":
            git_args = args.get("args", [])
            if git_args and git_args[0] == "commit":
                # Extract commit message from -m flag
                msg = ""
                for i, a in enumerate(git_args):
                    if a == "-m" and i + 1 < len(git_args):
                        msg = git_args[i + 1]
                        break
                if msg:
                    state.record_decision(
                        f"git commit: {_normalize_inline(msg)}",
                        turn_number,
                    )

        elif name == "shell_execute":
            if result and not result.success:
                cmd = _normalize_inline(str(args.get("command", "")))
                error_msg = _normalize_inline(result.error or "")
                state.record_error(
                    f"Command failed: {cmd} — {error_msg}", turn_number,
                )


def _normalize_inline(value: str) -> str:
    """Collapse whitespace for inline session-state fields."""
    return " ".join(str(value or "").split())
