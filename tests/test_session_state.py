"""Tests for SessionState and auto-extraction."""

from __future__ import annotations

import yaml

from loom.cowork.session import ToolCallEvent
from loom.cowork.session_state import (
    SessionState,
    extract_state_from_tool_events,
)
from loom.tools.registry import ToolResult


class TestSessionState:
    def test_record_file(self):
        state = SessionState()
        state.record_file("src/main.py", "edited", 5)
        assert len(state.files_touched) == 1
        assert state.files_touched[0].path == "src/main.py"
        assert state.files_touched[0].turn == 5

    def test_record_decision(self):
        state = SessionState()
        state.record_decision("Use JWT for auth", 3)
        assert len(state.key_decisions) == 1
        assert "JWT" in state.key_decisions[0]
        assert "turn 3" in state.key_decisions[0]

    def test_record_error_resolved(self):
        state = SessionState()
        state.record_error_resolved("ImportError fixed", 7)
        assert len(state.errors_resolved) == 1

    def test_set_focus(self):
        state = SessionState()
        state.set_focus("Implement user registration")
        assert state.current_focus == "Implement user registration"

    def test_set_focus_truncation(self):
        state = SessionState()
        state.set_focus("x" * 200)
        assert len(state.current_focus) == 100

    def test_pruning_files(self):
        state = SessionState()
        for i in range(30):
            state.record_file(f"file{i}.py", "edited", i)
        assert len(state.files_touched) == state.MAX_FILES

    def test_pruning_decisions(self):
        state = SessionState()
        for i in range(15):
            state.record_decision(f"Decision {i}", i)
        assert len(state.key_decisions) == state.MAX_DECISIONS

    def test_pruning_errors(self):
        state = SessionState()
        for i in range(10):
            state.record_error_resolved(f"Error {i}", i)
        assert len(state.errors_resolved) == state.MAX_ERRORS

    def test_to_yaml(self):
        state = SessionState(
            session_id="abc123",
            workspace="/tmp/project",
            model_name="claude-sonnet",
            turn_count=5,
            total_tokens=1000,
            current_focus="Working on auth",
        )
        state.record_file("src/auth.py", "edited", 3)
        state.record_decision("Using JWT", 2)

        result = state.to_yaml()
        parsed = yaml.safe_load(result)

        assert parsed["session"]["id"] == "abc123"
        assert parsed["session"]["turn_count"] == 5
        assert parsed["files_touched"][0]["path"] == "src/auth.py"
        assert parsed["current_focus"] == "Working on auth"

    def test_to_yaml_minimal(self):
        """Empty state produces minimal YAML."""
        state = SessionState(session_id="x", workspace="/tmp", model_name="m")
        result = state.to_yaml()
        parsed = yaml.safe_load(result)
        assert "files_touched" not in parsed
        assert "key_decisions" not in parsed

    def test_to_dict_and_from_dict(self):
        state = SessionState(
            session_id="test",
            workspace="/tmp",
            model_name="m",
            turn_count=10,
            current_focus="auth",
        )
        state.record_file("a.py", "created", 1)
        state.record_decision("use JWT", 2)

        d = state.to_dict()
        restored = SessionState.from_dict(d)

        assert restored.session_id == "test"
        assert restored.turn_count == 10
        assert restored.current_focus == "auth"
        assert len(restored.files_touched) == 1
        assert restored.files_touched[0].path == "a.py"
        assert len(restored.key_decisions) == 1

    def test_from_json(self):
        state = SessionState(session_id="x", workspace="/tmp", model_name="m")
        import json
        raw = json.dumps(state.to_dict())
        restored = SessionState.from_json(raw)
        assert restored.session_id == "x"

    def test_from_json_none(self):
        state = SessionState.from_json(None)
        assert state.session_id == ""

    def test_from_json_empty(self):
        state = SessionState.from_json("")
        assert state.session_id == ""


class TestExtractState:
    def _make_event(self, name, args, result=None):
        return ToolCallEvent(
            name=name,
            args=args,
            result=result or ToolResult.ok("ok"),
        )

    def test_write_file(self):
        state = SessionState()
        events = [self._make_event("write_file", {"path": "src/new.py"})]
        extract_state_from_tool_events(state, 5, events)
        assert len(state.files_touched) == 1
        assert state.files_touched[0].action == "created"

    def test_edit_file(self):
        state = SessionState()
        events = [self._make_event("edit_file", {"file_path": "src/main.py"})]
        extract_state_from_tool_events(state, 3, events)
        assert state.files_touched[0].action == "edited"

    def test_delete_file(self):
        state = SessionState()
        events = [self._make_event("delete_file", {"path": "old.py"})]
        extract_state_from_tool_events(state, 2, events)
        assert state.files_touched[0].action == "deleted"

    def test_read_file(self):
        state = SessionState()
        events = [self._make_event("read_file", {"file_path": "config.py"})]
        extract_state_from_tool_events(state, 1, events)
        assert state.files_touched[0].action == "read"

    def test_git_commit(self):
        state = SessionState()
        events = [self._make_event("git_command", {"args": ["commit", "-m", "Add feature"]})]
        extract_state_from_tool_events(state, 4, events)
        assert len(state.key_decisions) == 1
        assert "Add feature" in state.key_decisions[0]

    def test_shell_failure(self):
        state = SessionState()
        events = [self._make_event(
            "shell_execute",
            {"command": "pytest tests/"},
            result=ToolResult(success=False, output="FAILED"),
        )]
        extract_state_from_tool_events(state, 6, events)
        assert len(state.errors_resolved) == 1

    def test_multiple_events(self):
        state = SessionState()
        events = [
            self._make_event("read_file", {"file_path": "a.py"}),
            self._make_event("edit_file", {"file_path": "a.py"}),
            self._make_event("write_file", {"path": "b.py"}),
        ]
        extract_state_from_tool_events(state, 7, events)
        assert len(state.files_touched) == 3
