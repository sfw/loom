"""Focused tests for extracted runner session state helpers."""

from __future__ import annotations

from loom.engine.runner.session import RunnerSession, new_runner_session
from loom.engine.runner.types import ToolCallRecord
from loom.tools.registry import ToolResult


def test_new_runner_session_normalizes_prior_tool_and_evidence_state() -> None:
    record = ToolCallRecord(
        tool="read_file",
        args={"path": "notes.md"},
        result=ToolResult.ok("ok"),
    )

    session = new_runner_session(
        prompt="Complete subtask.",
        prior_successful_tool_calls=[record, "invalid"],  # type: ignore[list-item]
        prior_evidence_records=[
            {"evidence_id": "ev-1"},
            {"evidence_id": ""},
            {"no_evidence_id": "skip"},
            "invalid",  # type: ignore[list-item]
        ],
    )

    assert isinstance(session, RunnerSession)
    assert session.messages == [{"role": "user", "content": "Complete subtask."}]
    assert session.known_evidence_ids == {"ev-1"}
    assert session.historical_successful_tool_calls == [record]


def test_runner_session_defaults_are_empty_mutable_state() -> None:
    session = RunnerSession(messages=[{"role": "user", "content": "x"}])

    assert session.tool_calls_record == []
    assert session.evidence_records_current == []
    assert session.known_evidence_ids == set()
    assert session.historical_successful_tool_calls == []
    assert session.total_tokens == 0
    assert session.response is None
    assert session.completed_normally is False
    assert session.interruption_reason is None
