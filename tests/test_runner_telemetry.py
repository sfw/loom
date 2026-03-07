"""Focused tests for extracted runner telemetry helpers."""

from __future__ import annotations

from loom.engine.runner import telemetry as runner_telemetry
from loom.events.bus import EventBus
from loom.events.types import COMPACTION_POLICY_DECISION, OVERFLOW_FALLBACK_APPLIED


class _TelemetryRunnerStub:
    RUNNER_COMPACTION_POLICY_MODE = "tiered"
    MAX_MODEL_CONTEXT_TOKENS = 24000

    def __init__(self) -> None:
        self._event_bus = EventBus()
        self._enable_artifact_telemetry_events = True
        self._max_model_context_tokens = 2400
        self._runner_compaction_policy_mode = "tiered"
        self._last_compaction_diagnostics = {
            "compaction_policy_mode": "tiered",
            "compaction_pressure_ratio": 0.91,
            "compaction_stage": "stage_2_tool_outputs",
            "compaction_skipped_reason": "",
        }
        self._active_subtask_telemetry_counters = (
            runner_telemetry.new_subtask_telemetry_counters()
        )

    def _runner_compaction_mode(self) -> str:
        return self._runner_compaction_policy_mode


def test_emit_compaction_policy_decision_from_diagnostics_emits_event() -> None:
    runner = _TelemetryRunnerStub()
    events = []
    runner._event_bus.subscribe_all(lambda event: events.append(event))

    runner_telemetry.emit_compaction_policy_decision_from_diagnostics(
        runner,
        task_id="task-1",
        subtask_id="subtask-1",
    )

    decision_events = [
        event for event in events
        if event.event_type == COMPACTION_POLICY_DECISION
    ]
    assert len(decision_events) == 1
    payload = decision_events[0].data
    assert payload["decision"] == "compact_tool"
    assert payload["reason"] == "tool_output_compacted"
    assert payload["policy_mode"] == "tiered"
    assert runner._active_subtask_telemetry_counters["compaction_policy_decisions"] == 1


def test_emit_overflow_fallback_telemetry_emits_dual_events() -> None:
    runner = _TelemetryRunnerStub()
    events = []
    runner._event_bus.subscribe_all(lambda event: events.append(event))

    runner_telemetry.emit_overflow_fallback_telemetry(
        runner,
        task_id="task-1",
        subtask_id="subtask-1",
        report={
            "overflow_fallback_applied": True,
            "overflow_fallback_rewritten_messages": 2,
            "overflow_fallback_chars_reduced": 4000,
            "overflow_fallback_preserved_recent_messages": 1,
        },
    )

    decision_events = [
        event for event in events
        if event.event_type == COMPACTION_POLICY_DECISION
    ]
    overflow_events = [
        event for event in events
        if event.event_type == OVERFLOW_FALLBACK_APPLIED
    ]
    assert len(decision_events) == 1
    assert len(overflow_events) == 1
    overflow_payload = overflow_events[0].data
    assert overflow_payload["decision"] == "fallback_rewrite"
    assert overflow_payload["rewritten_messages"] == 2
    assert overflow_payload["chars_reduced"] == 4000
    assert runner._active_subtask_telemetry_counters["overflow_fallback_count"] == 1
