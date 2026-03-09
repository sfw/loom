"""Focused tests for extracted orchestrator telemetry helpers."""

from __future__ import annotations

from types import SimpleNamespace

from loom.engine.orchestrator import telemetry as orchestrator_telemetry
from loom.events.bus import Event, EventBus
from loom.events.types import (
    VERIFICATION_FALSE_NEGATIVE_CANDIDATE,
    VERIFICATION_OUTCOME,
    VERIFICATION_SHADOW_DIFF,
)


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


def test_emit_telemetry_run_summary_includes_reliability_metrics() -> None:
    bus = EventBus()
    events = []
    bus.subscribe_all(lambda event: events.append(event))
    orchestrator = SimpleNamespace(
        _emitted_telemetry_summary_runs=set(),
        _telemetry_rollup=orchestrator_telemetry.new_telemetry_rollup(),
        _events=bus,
        _task_run_id=lambda task: "run-1",
        _new_telemetry_rollup=orchestrator_telemetry.new_telemetry_rollup,
        _task_event_counts=lambda task_id: orchestrator_telemetry.task_event_counts(bus, task_id),
        _verification_reason_counts=lambda task_id: orchestrator_telemetry.verification_reason_counts(
            event_bus=bus,
            task_id=task_id,
            verification_outcome_event_type=VERIFICATION_OUTCOME,
        ),
        _run_budget=SimpleNamespace(snapshot=lambda: {}),
        _emit=lambda event_type, task_id, data: bus.emit(Event(event_type=event_type, task_id=task_id, data=data)),
    )
    task = SimpleNamespace(id="t-run", metadata={})
    bus.emit(Event(event_type=VERIFICATION_OUTCOME, task_id="t-run", data={"reason_code": "parse_inconclusive"}))
    bus.emit(Event(event_type=VERIFICATION_OUTCOME, task_id="t-run", data={"reason_code": "hard_invariant_failed"}))
    bus.emit(Event(event_type=VERIFICATION_OUTCOME, task_id="t-run", data={"reason_code": "claim_inconclusive"}))
    bus.emit(Event(event_type="verification_failed", task_id="t-run", data={}))
    bus.emit(Event(event_type="verification_failed", task_id="t-run", data={}))
    bus.emit(Event(event_type="verification_started", task_id="t-run", data={}))
    bus.emit(Event(event_type="verification_passed", task_id="t-run", data={}))
    bus.emit(Event(event_type="remediation_attempt", task_id="t-run", data={}))
    bus.emit(Event(event_type="remediation_resolved", task_id="t-run", data={}))
    bus.emit(Event(event_type=VERIFICATION_SHADOW_DIFF, task_id="t-run", data={}))
    bus.emit(Event(event_type=VERIFICATION_FALSE_NEGATIVE_CANDIDATE, task_id="t-run", data={}))

    orchestrator_telemetry._emit_telemetry_run_summary(orchestrator, task)  # type: ignore[arg-type]

    summary_events = [event for event in events if event.event_type == "telemetry_run_summary"]
    assert len(summary_events) == 1
    metrics = summary_events[0].data.get("reliability_metrics", {})
    assert metrics["verifier_terminal_failure_rate"] == 0.6667
    assert metrics["inconclusive_outcome_rate"] == 0.6667
    assert metrics["inconclusive_rescue_rate"] == 1.0
    assert metrics["false_block_audit_rate"] == 1.0
