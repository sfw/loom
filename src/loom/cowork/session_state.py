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
    active_decisions: list[dict] = field(default_factory=list)
    active_proposals: list[dict] = field(default_factory=list)
    recent_research: list[dict] = field(default_factory=list)
    open_questions: list[dict] = field(default_factory=list)
    memory_index_last_indexed_turn: int = 0
    memory_index_degraded: bool = False
    memory_index_failure_count: int = 0
    memory_index_last_error: str = ""
    compact_boundary_message_count: int = 0
    compact_summary_message_count: int = 0
    compact_summary_tool_message_count: int = 0
    compact_summary: str = ""
    # UI-only metadata (not injected into prompts), e.g. restored TUI tab state.
    ui_state: dict = field(default_factory=dict)

    # Pruning limits
    MAX_FILES: int = 20
    MAX_DECISIONS: int = 10
    MAX_ERRORS: int = 5
    MAX_MEMORY_ITEMS: int = 4

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

    @property
    def has_memory_snapshot(self) -> bool:
        return bool(
            self.active_decisions
            or self.active_proposals
            or self.recent_research
            or self.open_questions
        )

    @property
    def has_compact_memory(self) -> bool:
        return bool(str(self.compact_summary or "").strip())

    def update_memory_snapshot(self, snapshot: dict[str, list[dict]]) -> None:
        """Update compact marker-oriented memory slices used in prompts."""
        if not isinstance(snapshot, dict):
            return
        self.active_decisions = self._normalize_memory_items(
            snapshot.get("active_decisions", []),
            limit=self.MAX_MEMORY_ITEMS,
        )
        self.active_proposals = self._normalize_memory_items(
            snapshot.get("active_proposals", []),
            limit=self.MAX_MEMORY_ITEMS,
        )
        self.recent_research = self._normalize_memory_items(
            snapshot.get("recent_research", []),
            limit=self.MAX_MEMORY_ITEMS,
        )
        self.open_questions = self._normalize_memory_items(
            snapshot.get("open_questions", []),
            limit=self.MAX_MEMORY_ITEMS,
        )

    def update_memory_index_meta(
        self,
        *,
        last_indexed_turn: int | None = None,
        degraded: bool | None = None,
        failure_count: int | None = None,
        last_error: str | None = None,
    ) -> None:
        if last_indexed_turn is not None:
            self.memory_index_last_indexed_turn = max(0, int(last_indexed_turn))
        if degraded is not None:
            self.memory_index_degraded = bool(degraded)
        if failure_count is not None:
            self.memory_index_failure_count = max(0, int(failure_count))
        if last_error is not None:
            self.memory_index_last_error = _normalize_inline(last_error)

    def update_compact_memory(
        self,
        *,
        summary: str,
        boundary_message_count: int | None = None,
        message_count: int | None = None,
        tool_message_count: int | None = None,
    ) -> None:
        self.compact_summary = _normalize_multiline(summary)
        if boundary_message_count is not None:
            self.compact_boundary_message_count = max(0, int(boundary_message_count))
        if message_count is not None:
            self.compact_summary_message_count = max(0, int(message_count))
        if tool_message_count is not None:
            self.compact_summary_tool_message_count = max(0, int(tool_message_count))

    def clear_compact_memory(self) -> None:
        self.compact_boundary_message_count = 0
        self.compact_summary_message_count = 0
        self.compact_summary_tool_message_count = 0
        self.compact_summary = ""

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

        active_memory: dict[str, list[str]] = {}
        if self.active_decisions:
            active_memory["decisions"] = [
                self._memory_line(item, marker="DECISION")
                for item in self.active_decisions
            ]
        if self.active_proposals:
            active_memory["proposals"] = [
                self._memory_line(item, marker="PROPOSAL")
                for item in self.active_proposals
            ]
        if self.recent_research:
            active_memory["research"] = [
                self._memory_line(item, marker="RESEARCH")
                for item in self.recent_research
            ]
        if self.open_questions:
            active_memory["open_questions"] = [
                self._memory_line(item, marker="OPEN_QUESTION")
                for item in self.open_questions
            ]
        if active_memory:
            data["active_memory"] = active_memory
        if self.has_compact_memory:
            data["compact_memory"] = {
                "boundary_message_count": self.compact_boundary_message_count,
                "message_count": self.compact_summary_message_count,
                "tool_message_count": self.compact_summary_tool_message_count,
                "summary": self.compact_summary.splitlines(),
            }
        if self.memory_index_degraded or self.memory_index_last_indexed_turn > 0:
            data["memory_index"] = {
                "last_indexed_turn": self.memory_index_last_indexed_turn,
                "degraded": self.memory_index_degraded,
                "failure_count": self.memory_index_failure_count,
            }

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
            "active_decisions": self.active_decisions,
            "active_proposals": self.active_proposals,
            "recent_research": self.recent_research,
            "open_questions": self.open_questions,
            "memory_index_last_indexed_turn": self.memory_index_last_indexed_turn,
            "memory_index_degraded": self.memory_index_degraded,
            "memory_index_failure_count": self.memory_index_failure_count,
            "memory_index_last_error": self.memory_index_last_error,
            "compact_boundary_message_count": self.compact_boundary_message_count,
            "compact_summary_message_count": self.compact_summary_message_count,
            "compact_summary_tool_message_count": self.compact_summary_tool_message_count,
            "compact_summary": self.compact_summary,
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
            active_decisions=cls._normalize_memory_items(
                data.get("active_decisions", []),
                limit=cls.MAX_MEMORY_ITEMS,
            ),
            active_proposals=cls._normalize_memory_items(
                data.get("active_proposals", []),
                limit=cls.MAX_MEMORY_ITEMS,
            ),
            recent_research=cls._normalize_memory_items(
                data.get("recent_research", []),
                limit=cls.MAX_MEMORY_ITEMS,
            ),
            open_questions=cls._normalize_memory_items(
                data.get("open_questions", []),
                limit=cls.MAX_MEMORY_ITEMS,
            ),
            memory_index_last_indexed_turn=max(
                0,
                int(data.get("memory_index_last_indexed_turn", 0) or 0),
            ),
            memory_index_degraded=bool(data.get("memory_index_degraded", False)),
            memory_index_failure_count=max(
                0,
                int(data.get("memory_index_failure_count", 0) or 0),
            ),
            memory_index_last_error=_normalize_inline(
                str(data.get("memory_index_last_error", "") or "")
            ),
            compact_boundary_message_count=max(
                0,
                int(data.get("compact_boundary_message_count", 0) or 0),
            ),
            compact_summary_message_count=max(
                0,
                int(data.get("compact_summary_message_count", 0) or 0),
            ),
            compact_summary_tool_message_count=max(
                0,
                int(data.get("compact_summary_tool_message_count", 0) or 0),
            ),
            compact_summary=_normalize_multiline(
                str(data.get("compact_summary", "") or ""),
            ),
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

    @staticmethod
    def _normalize_memory_items(value: object, *, limit: int) -> list[dict]:
        if not isinstance(value, list):
            return []
        normalized: list[dict] = []
        for item in value:
            parsed = SessionState._normalize_memory_item(item)
            if parsed is None:
                continue
            normalized.append(parsed)
            if len(normalized) >= max(1, int(limit)):
                break
        return normalized

    @staticmethod
    def _normalize_memory_item(value: object) -> dict | None:
        if isinstance(value, str):
            summary = _normalize_inline(value)
            if not summary:
                return None
            return {
                "id": 0,
                "entry_type": "",
                "status": "active",
                "summary": summary,
                "rationale": "",
                "topic": "",
                "evidence_excerpt": "",
                "source_turn_start": 0,
                "source_turn_end": 0,
                "confidence": 0.0,
            }
        if not isinstance(value, dict):
            return None
        summary = _normalize_inline(str(value.get("summary", "") or ""))
        if not summary:
            return None
        start = max(0, int(value.get("source_turn_start", 0) or 0))
        end = max(start, int(value.get("source_turn_end", start) or start))
        try:
            confidence = float(value.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "id": max(0, int(value.get("id", 0) or 0)),
            "entry_type": _normalize_inline(str(value.get("entry_type", "") or "")).lower(),
            "status": _normalize_inline(str(value.get("status", "active") or "active")).lower(),
            "summary": summary,
            "rationale": _normalize_inline(str(value.get("rationale", "") or "")),
            "topic": _normalize_inline(str(value.get("topic", "") or "")).lower(),
            "evidence_excerpt": _normalize_inline(str(value.get("evidence_excerpt", "") or "")),
            "source_turn_start": start,
            "source_turn_end": end,
            "confidence": max(0.0, min(1.0, confidence)),
        }

    @staticmethod
    def _memory_line(item: dict, *, marker: str) -> str:
        summary = _normalize_inline(str(item.get("summary", "") or ""))
        start = max(0, int(item.get("source_turn_start", 0) or 0))
        end = max(start, int(item.get("source_turn_end", start) or start))
        turns = f"turn {start}" if start == end else f"turns {start}-{end}"
        status = _normalize_inline(str(item.get("status", "active") or "active")).lower()
        return f"[{marker}][{status}] {summary} ({turns})"


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


def _normalize_multiline(value: str) -> str:
    """Normalize multiline summaries while preserving line boundaries."""
    lines = [
        _normalize_inline(line)
        for line in str(value or "").splitlines()
        if _normalize_inline(line)
    ]
    return "\n".join(lines)
