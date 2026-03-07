"""Per-subtask execution session state for the runner loop."""

from __future__ import annotations

from dataclasses import dataclass, field

from loom.models.base import ModelResponse

from .types import ToolCallRecord


@dataclass
class RunnerSession:
    """Mutable state tracked for one `SubtaskRunner.run()` invocation."""

    messages: list[dict]
    tool_calls_record: list[ToolCallRecord] = field(default_factory=list)
    evidence_records_current: list[dict] = field(default_factory=list)
    known_evidence_ids: set[str] = field(default_factory=set)
    historical_successful_tool_calls: list[ToolCallRecord] = field(default_factory=list)
    total_tokens: int = 0
    response: ModelResponse | None = None
    completed_normally: bool = False
    interruption_reason: str | None = None
    budget_exhaustion_note: str | None = None
    ask_user_questions_asked: int = 0
    last_ask_user_requested_at: float = 0.0


def new_runner_session(
    *,
    prompt: str,
    prior_successful_tool_calls: list[ToolCallRecord] | None = None,
    prior_evidence_records: list[dict] | None = None,
) -> RunnerSession:
    """Build a session with normalized prior evidence/tool state."""
    known_evidence_ids = {
        str(item.get("evidence_id", "")).strip()
        for item in (prior_evidence_records or [])
        if isinstance(item, dict)
    }
    known_evidence_ids.discard("")
    historical_successful_tool_calls = [
        call
        for call in (prior_successful_tool_calls or [])
        if isinstance(call, ToolCallRecord)
    ]
    return RunnerSession(
        messages=[{"role": "user", "content": prompt}],
        known_evidence_ids=known_evidence_ids,
        historical_successful_tool_calls=historical_successful_tool_calls,
    )
