"""Focused tests for extracted verification event helpers."""

from __future__ import annotations

from types import SimpleNamespace

from loom.engine.verification import events as verification_events
from loom.engine.verification.types import Check, VerificationResult
from loom.events.bus import EventBus
from loom.events.types import (
    MODEL_INVOCATION,
    PLACEHOLDER_FINDINGS_EXTRACTED,
    VERIFICATION_CONTRADICTION_DETECTED,
    VERIFICATION_FAILED,
    VERIFICATION_INCONCLUSIVE_RATE,
    VERIFICATION_OUTCOME,
    VERIFICATION_RULE_APPLIED,
    VERIFICATION_RULE_FAILURE_BY_TYPE,
    VERIFICATION_RULE_SKIPPED,
    VERIFICATION_SHADOW_DIFF,
    VERIFICATION_STARTED,
)


def test_emit_rule_scope_event_emits_applied_and_skipped() -> None:
    bus = EventBus()
    events = []
    bus.subscribe_all(lambda event: events.append(event))
    rule = SimpleNamespace(
        name="citations_required",
        type="regex",
        severity="error",
        enforcement="hard",
        scope="current_phase",
        applies_to_phases=["phase-a"],
    )

    verification_events.emit_rule_scope_event(
        bus,
        task_id="task-1",
        subtask_id="phase-a",
        applied=True,
        rule=rule,
        reason="phase_match",
    )
    verification_events.emit_rule_scope_event(
        bus,
        task_id="task-1",
        subtask_id="phase-a",
        applied=False,
        rule=rule,
        reason="phase_mismatch",
    )

    event_types = [event.event_type for event in events]
    assert event_types == [VERIFICATION_RULE_APPLIED, VERIFICATION_RULE_SKIPPED]
    payload = events[0].data
    assert payload["subtask_id"] == "phase-a"
    assert payload["rule_id"] == "citations_required"
    assert payload["reason"] == "phase_match"


def test_emit_verification_lifecycle_events_preserve_outcome_fields() -> None:
    bus = EventBus()
    events = []
    bus.subscribe_all(lambda event: events.append(event))
    result = VerificationResult(
        tier=2,
        passed=False,
        confidence=0.42,
        outcome="fail",
        reason_code="llm_semantic_failed",
        severity_class="semantic",
    )

    verification_events.emit_verification_started(
        bus,
        task_id="task-1",
        subtask_id="phase-a",
        target_tier=2,
    )
    verification_events.emit_verification_outcome(
        bus,
        task_id="task-1",
        subtask_id="phase-a",
        result=result,
    )
    verification_events.emit_verification_terminal(
        bus,
        task_id="task-1",
        subtask_id="phase-a",
        result=result,
    )

    event_types = [event.event_type for event in events]
    assert event_types == [VERIFICATION_STARTED, VERIFICATION_OUTCOME, VERIFICATION_FAILED]
    outcome_payload = events[1].data
    assert outcome_payload["outcome"] == "fail"
    assert outcome_payload["reason_code"] == "llm_semantic_failed"
    assert outcome_payload["source_component"] == "verification"
    terminal_payload = events[2].data
    assert terminal_payload["outcome"] == "fail"
    assert terminal_payload["reason_code"] == "llm_semantic_failed"


def test_emit_placeholder_findings_event_truncates_and_normalizes() -> None:
    bus = EventBus()
    events = []
    bus.subscribe_all(lambda event: events.append(event))
    raw_findings = [{
        "rule_name": "no-placeholders",
        "file_path": "report.md",
        "line": str(index + 1),
        "column": "2",
        "token": "N/A",
        "context": "N/A in cell",
    } for index in range(130)]
    result = VerificationResult(
        tier=1,
        passed=False,
        outcome="fail",
        reason_code="incomplete_deliverable_placeholder",
        metadata={
            "placeholder_findings": raw_findings,
            "placeholder_finding_count": 130,
            "missing_targets": ["report.md"],
            "remediation_mode": "retry",
            "failure_class": "placeholder_token",
        },
    )

    verification_events.emit_placeholder_findings_extracted(
        bus,
        task_id="task-1",
        subtask_id="phase-a",
        result=result,
    )

    extracted = [event for event in events if event.event_type == PLACEHOLDER_FINDINGS_EXTRACTED]
    assert len(extracted) == 1
    payload = extracted[0].data
    assert payload["finding_count"] == 130
    assert len(payload["findings"]) == 120
    assert payload["findings"][0]["line"] == 1
    assert payload["missing_targets"] == ["report.md"]


def test_emit_instrumentation_events_emits_expected_event_family() -> None:
    bus = EventBus()
    events = []
    bus.subscribe_all(lambda event: events.append(event))
    result = VerificationResult(
        tier=2,
        passed=False,
        outcome="fail",
        reason_code="parse_inconclusive",
        checks=[Check(name="process_rule_required_fact_checker", passed=False)],
        metadata={
            "contradicted_reason_code": "incomplete_deliverable_placeholder",
            "contradiction_downgraded": True,
            "deterministic_placeholder_scan": {
                "scanned_file_count": 2,
                "scanned_total_bytes": 2048,
                "matched_file_count": 1,
                "scan_mode": "targeted_plus_fallback",
                "coverage_sufficient": True,
                "coverage_insufficient_reason": "",
                "candidate_source_counts": {"canonical": 1},
                "cap_exhausted": False,
            },
        },
    )
    legacy = VerificationResult(
        tier=2,
        passed=True,
        outcome="pass",
        reason_code="",
    )

    verification_events.emit_instrumentation_events(
        bus,
        task_id="task-1",
        subtask_id="phase-a",
        result=result,
        legacy_result=legacy,
    )

    event_types = [event.event_type for event in events]
    assert VERIFICATION_CONTRADICTION_DETECTED in event_types
    assert VERIFICATION_INCONCLUSIVE_RATE in event_types
    assert VERIFICATION_RULE_FAILURE_BY_TYPE in event_types
    assert VERIFICATION_SHADOW_DIFF in event_types
    contradiction = next(
        event.data
        for event in events
        if event.event_type == VERIFICATION_CONTRADICTION_DETECTED
    )
    assert contradiction["reason_code"] == "parse_inconclusive"
    assert contradiction["candidate_source_counts"] == {"canonical": 1}


def test_emit_model_invocation_event_merges_details() -> None:
    bus = EventBus()
    events = []
    bus.subscribe_all(lambda event: events.append(event))

    verification_events.emit_model_invocation_event(
        bus,
        task_id="task-1",
        subtask_id="phase-a",
        model_name="gpt-test",
        phase="verify",
        details={"attempt": 2},
    )

    model_events = [event for event in events if event.event_type == MODEL_INVOCATION]
    assert len(model_events) == 1
    payload = model_events[0].data
    assert payload["subtask_id"] == "phase-a"
    assert payload["model"] == "gpt-test"
    assert payload["phase"] == "verify"
    assert payload["attempt"] == 2
