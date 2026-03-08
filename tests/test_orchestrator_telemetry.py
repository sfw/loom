"""Focused tests for extracted orchestrator telemetry helpers."""

from __future__ import annotations

from types import SimpleNamespace

from loom.engine.orchestrator import telemetry as orchestrator_telemetry
from loom.events.bus import Event, EventBus
from loom.events.types import VERIFICATION_OUTCOME


def test_new_telemetry_rollup_has_expected_keys() -> None:
    rollup = orchestrator_telemetry.new_telemetry_rollup()

    assert rollup["model_invocations"] == 0
    assert rollup["tool_calls"] == 0
    assert rollup["compactor_warning_count"] == 0
    assert rollup["sealed_policy_preflight_blocked"] == 0
    assert rollup["sealed_reseal_applied"] == 0
    assert rollup["sealed_unexpected_mutation_detected"] == 0


def test_accumulate_subtask_telemetry_updates_rollup() -> None:
    orchestrator = SimpleNamespace(_telemetry_rollup=None)
    result = SimpleNamespace(
        telemetry_counters={
            "model_invocations": 2,
            "tool_calls": 3,
            "mutating_tool_calls": 1,
            "compactor_warning_count": 4,
            "sealed_reseal_applied": 2,
        },
    )

    orchestrator_telemetry.accumulate_subtask_telemetry(
        orchestrator,
        result,  # type: ignore[arg-type]
    )

    assert orchestrator._telemetry_rollup["model_invocations"] == 2
    assert orchestrator._telemetry_rollup["tool_calls"] == 3
    assert orchestrator._telemetry_rollup["compactor_warning_count"] == 4
    assert orchestrator._telemetry_rollup["sealed_reseal_applied"] == 2


def test_task_event_counts_and_verification_reason_counts() -> None:
    bus = EventBus()
    bus.emit(Event(event_type="task_started", task_id="t1", data={}))
    bus.emit(Event(
        event_type=VERIFICATION_OUTCOME,
        task_id="t1",
        data={"reason_code": "parse_inconclusive"},
    ))
    bus.emit(Event(
        event_type=VERIFICATION_OUTCOME,
        task_id="t1",
        data={"reason_code": "parse_inconclusive"},
    ))
    bus.emit(Event(
        event_type=VERIFICATION_OUTCOME,
        task_id="t1",
        data={"reason_code": ""},
    ))
    bus.emit(Event(event_type="task_started", task_id="t2", data={}))

    counts = orchestrator_telemetry.task_event_counts(bus, "t1")
    reasons = orchestrator_telemetry.verification_reason_counts(
        event_bus=bus,
        task_id="t1",
        verification_outcome_event_type=VERIFICATION_OUTCOME,
    )

    assert counts["task_started"] == 1
    assert counts[VERIFICATION_OUTCOME] == 3
    assert reasons["parse_inconclusive"] == 2
    assert reasons["unspecified"] == 1
