"""Focused tests for extracted verification claim helpers."""

from __future__ import annotations

from types import SimpleNamespace

from loom.engine.verification import claims as verification_claims
from loom.engine.verification.types import VerificationResult
from loom.events.bus import EventBus
from loom.events.types import CLAIM_VERIFICATION_SUMMARY
from loom.tools.registry import ToolResult


def test_extract_claim_lifecycle_from_fact_checker_verdicts() -> None:
    gates = SimpleNamespace()
    tool_calls = [
        SimpleNamespace(
            tool="fact_checker",
            result=ToolResult.ok(
                "ok",
                data={
                    "verdicts": [
                        {"claim": "Revenue was 42 in 2024", "verdict": "supported"},
                        {"claim": "Forecast remains unchanged", "verdict": "stale"},
                    ],
                },
            ),
        ),
    ]
    result = VerificationResult(tier=1, passed=True)

    claims = verification_claims.extract_claim_lifecycle(
        gates,
        tool_calls=tool_calls,
        result=result,
        validity_contract={"critical_claim_types": ["numeric"]},
    )

    assert len(claims) == 2
    assert claims[0]["claim_type"] == "numeric"
    assert claims[0]["criticality"] == "critical"
    assert claims[1]["status"] == "stale"


def test_attach_claim_lifecycle_adds_metadata_and_emits_summary_event() -> None:
    bus = EventBus()
    events = []
    bus.subscribe_all(lambda event: events.append(event))
    gates = SimpleNamespace(_event_bus=bus)
    tool_calls = [
        SimpleNamespace(
            tool="fact_checker",
            result=ToolResult.ok(
                "ok",
                data={
                    "verdicts": [
                        {"claim": "Profit margin improved", "verdict": "supported"},
                        {"claim": "Debt declined", "verdict": "contradicted"},
                    ],
                },
            ),
        ),
    ]
    base = VerificationResult(
        tier=2,
        passed=False,
        outcome="fail",
        reason_code="llm_semantic_failed",
    )

    enriched = verification_claims.attach_claim_lifecycle(
        gates,
        task_id="task-1",
        subtask_id="phase-a",
        result=base,
        tool_calls=tool_calls,
        validity_contract={},
    )

    counts = enriched.metadata.get("claim_status_counts", {})
    assert counts["supported"] == 1
    assert counts["contradicted"] == 1
    summary_events = [
        event for event in events
        if event.event_type == CLAIM_VERIFICATION_SUMMARY
    ]
    assert len(summary_events) == 1
    assert summary_events[0].data["subtask_id"] == "phase-a"
