"""Regression and historical replay coverage for durable self-correction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from loom.engine.correction import (
    CorrectionController,
    CorrectionHandler,
    CorrectionState,
    Repairability,
)
from loom.engine.correction.types import ProgressVector
from loom.engine.verification import Check, VerificationResult
from loom.state.memory import Database, MemoryManager


@pytest.fixture
async def correction_runtime(tmp_path):
    database = Database(tmp_path / "correction.db")
    await database.initialize()
    await database.execute(
        """INSERT INTO tasks (id, goal, status, created_at, updated_at)
           VALUES ('task-1', 'historical replay', 'executing', datetime('now'), datetime('now'))"""
    )
    memory = MemoryManager(database)
    events: list[tuple[str, str, dict]] = []
    controller = CorrectionController(
        memory,
        lambda event_type, task_id, data: events.append((event_type, task_id, data)),
        max_no_progress_attempts=2,
    )
    return database, memory, controller, events


def _result():
    return SimpleNamespace(tool_calls=[], evidence_records=[], summary="failed")


def test_attempt_local_artifact_counts_and_confidence_do_not_fake_progress() -> None:
    previous = ProgressVector(1, 1, 1, 0, 0, 0, 0.2)
    current = ProgressVector(1, 1, 1, 0, 0, 3, 1.0)

    assert current.improved_from(previous) is False


class TestHistoricalFailureReplay:
    async def test_403_tool_failure_routes_to_source_fallback(self, correction_runtime):
        _database, memory, controller, _events = correction_runtime
        verification = VerificationResult(
            tier=1,
            passed=False,
            outcome="fail",
            reason_code="tool_runtime_retryable",
            severity_class="infra",
            feedback="web_fetch: HTTP 403: target denied automated access",
            checks=[Check(name="web_fetch", passed=False, detail="HTTP 403")],
        )

        decision = await controller.record_failure(
            task_id="task-1",
            run_id="run-1",
            subtask_id="scope-company",
            result=_result(),
            verification=verification,
        )

        assert decision.repairability == Repairability.AUTOMATIC
        assert decision.handler == CorrectionHandler.SOURCE_FALLBACK
        assert decision.state == CorrectionState.PLANNED
        cycles = await memory.list_correction_cycles(task_id="task-1")
        assert cycles[0]["state"] == "planned"
        assert cycles[0]["blocker_snapshot"][0]["code"] == "tool_runtime_retryable"

    async def test_required_verifier_failure_retries_only_verification(
        self,
        correction_runtime,
    ):
        _database, _memory, controller, _events = correction_runtime
        verification = VerificationResult(
            tier=2,
            passed=False,
            outcome="fail",
            reason_code="required_verifier_empty",
            severity_class="infra",
            feedback="Verifier returned no parseable output.",
        )

        decision = await controller.record_failure(
            task_id="task-1",
            run_id="run-1",
            subtask_id="verify",
            result=_result(),
            verification=verification,
        )

        assert decision.handler == CorrectionHandler.RETRY_VERIFICATION
        assert decision.actions[0].action_type == "rerun_verifier"

    async def test_csv_mismatch_routes_to_bounded_schema_repair(
        self,
        correction_runtime,
    ):
        _database, _memory, controller, _events = correction_runtime
        verification = VerificationResult(
            tier=1,
            passed=False,
            outcome="fail",
            reason_code="csv_schema_mismatch",
            severity_class="semantic",
            feedback="A structured output row does not match its header.",
            checks=[
                Check(
                    name="syntax_comparison-matrix.csv",
                    passed=False,
                    detail=(
                        "reason_code=csv_schema_mismatch; "
                        "CSV row 8 has 13 columns (expected 11)."
                    ),
                ),
            ],
        )

        decision = await controller.record_failure(
            task_id="task-1",
            run_id="run-1",
            subtask_id="structured-output",
            result=_result(),
            verification=verification,
        )

        assert decision.repairability == Repairability.AUTOMATIC
        assert decision.handler == CorrectionHandler.SCHEMA_REPAIR
        assert decision.actions[0].action_type == "repair_structured_output_schema"
        assert decision.actions[0].arguments["targets"] == ["comparison-matrix.csv"]
        assert decision.actions[0].arguments["diagnostics"] == [{
            "target": "comparison-matrix.csv",
            "row_number": 8,
            "actual_columns": 13,
            "expected_columns": 11,
        }]
        assert "edit the existing structured output in place" in (
            decision.actions[0].arguments["guardrails"]
        )

    async def test_mixed_failed_checks_are_classified_in_one_pass(
        self,
        correction_runtime,
    ):
        _database, _memory, controller, _events = correction_runtime
        verification = VerificationResult(
            tier=1,
            passed=False,
            outcome="fail",
            reason_code="verification_failed",
            severity_class="semantic",
            feedback="The deliverables have multiple independently repairable gaps.",
            checks=[
                Check(
                    name="syntax_competitors.csv",
                    passed=False,
                    detail=(
                        "reason_code=csv_schema_mismatch; "
                        "CSV row 3 has 8 columns (expected 7)."
                    ),
                ),
                Check(
                    name="market_recommendations",
                    passed=False,
                    detail=(
                        "reason_code=missing_market_specific_recommendations; "
                        "market-specific recommendations are required."
                    ),
                ),
            ],
        )

        decision = await controller.record_failure(
            task_id="task-1",
            run_id="run-1",
            subtask_id="competitive-research",
            result=_result(),
            verification=verification,
        )

        assert {blocker.code for blocker in decision.blockers} == {
            "csv_schema_mismatch",
            "missing_market_specific_recommendations",
        }
        assert decision.handler == CorrectionHandler.SCHEMA_REPAIR
        assert decision.actions[0].arguments["diagnostics"] == [{
            "target": "competitors.csv",
            "row_number": 3,
            "actual_columns": 8,
            "expected_columns": 7,
        }]

    async def test_same_repair_lane_tracks_no_progress_across_reason_changes(
        self,
        correction_runtime,
    ):
        _database, _memory, controller, _events = correction_runtime

        first = await controller.record_failure(
            task_id="task-1",
            run_id="run-1",
            subtask_id="environmental-scan",
            result=_result(),
            verification=VerificationResult(
                tier=1,
                passed=False,
                outcome="fail",
                reason_code="aggregate_scan_not_zonal",
            ),
        )
        second = await controller.record_failure(
            task_id="task-1",
            run_id="run-1",
            subtask_id="environmental-scan",
            result=_result(),
            verification=VerificationResult(
                tier=1,
                passed=False,
                outcome="fail",
                reason_code="missing_leading_indicators_and_geographic_granularity",
            ),
        )
        third = await controller.record_failure(
            task_id="task-1",
            run_id="run-1",
            subtask_id="environmental-scan",
            result=_result(),
            verification=VerificationResult(
                tier=1,
                passed=False,
                outcome="fail",
                reason_code="missing_leading_indicators_and_geographic_granularity",
            ),
        )

        assert first.handler == second.handler == CorrectionHandler.CONTRACT_REPAIR
        assert first.cycle_id != second.cycle_id
        assert second.no_progress_count == 1
        assert third.stop_for_no_progress is True

    async def test_subtask_wide_attempt_budget_spans_changing_blockers(
        self,
        correction_runtime,
    ):
        _database, memory, _controller, events = correction_runtime
        controller = CorrectionController(
            memory,
            lambda event_type, task_id, data: events.append(
                (event_type, task_id, data),
            ),
            max_no_progress_attempts=5,
            max_total_attempts_per_subtask=2,
        )

        first = await controller.record_failure(
            task_id="task-1",
            run_id="run-1",
            subtask_id="risk-register",
            result=_result(),
            verification=VerificationResult(
                tier=1,
                passed=False,
                outcome="fail",
                reason_code="csv_schema_mismatch",
                feedback="CSV row 2 has 4 columns (expected 3).",
                checks=[
                    Check(
                        name="syntax_risk-register.csv",
                        passed=False,
                        detail="CSV row 2 has 4 columns (expected 3).",
                    ),
                ],
            ),
        )
        second = await controller.record_failure(
            task_id="task-1",
            run_id="run-1",
            subtask_id="risk-register",
            result=_result(),
            verification=VerificationResult(
                tier=1,
                passed=False,
                outcome="partial_verified",
                reason_code="missing_structured_leading_indicators",
                feedback="A required structured field remains missing.",
            ),
        )

        assert first.state == CorrectionState.PLANNED
        assert second.total_attempt_count == 2
        assert second.stop_for_attempt_budget is True
        assert second.state == CorrectionState.TERMINAL

    async def test_open_ended_budget_label_routes_to_checkpoint_continuation(
        self,
        correction_runtime,
    ):
        _database, _memory, controller, _events = correction_runtime
        verification = VerificationResult(
            tier=1,
            passed=False,
            outcome="fail",
            reason_code="budget_exhausted_incomplete",
            severity_class="infra",
            feedback="Tool iteration limit reached while preserving draft output.",
        )

        decision = await controller.record_failure(
            task_id="task-1",
            run_id="run-1",
            subtask_id="collect-evidence",
            result=_result(),
            verification=verification,
        )

        assert decision.repairability == Repairability.CONDITIONAL
        assert decision.handler == CorrectionHandler.CHECKPOINT_CONTINUE
        assert decision.actions[0].action_type == "continue_from_partial_checkpoint"

    async def test_output_policy_failure_routes_to_allowed_path_repair(
        self,
        correction_runtime,
    ):
        _database, _memory, controller, _events = correction_runtime
        verification = VerificationResult(
            tier=1,
            passed=False,
            outcome="fail",
            reason_code="forbidden_canonical_write",
            severity_class="hard_invariant",
            feedback="Output must be written through the declared staging path.",
            metadata={"missing_targets": ["deliverables/summary.md"]},
        )

        decision = await controller.record_failure(
            task_id="task-1",
            run_id="run-1",
            subtask_id="publish-summary",
            result=_result(),
            verification=verification,
        )

        assert decision.repairability == Repairability.AUTOMATIC
        assert decision.handler == CorrectionHandler.OUTPUT_REROUTE
        assert decision.actions[0].action_type == "reroute_output_to_allowed_path"

    async def test_verifier_specific_missing_field_routes_to_artifact_repair(
        self,
        correction_runtime,
    ):
        _database, _memory, controller, _events = correction_runtime
        verification = VerificationResult(
            tier=1,
            passed=False,
            outcome="fail",
            reason_code="missing_required_register_entry",
            severity_class="semantic",
            feedback="One required output field is absent.",
        )

        decision = await controller.record_failure(
            task_id="task-1",
            run_id="run-1",
            subtask_id="build-register",
            result=_result(),
            verification=verification,
        )

        assert decision.repairability == Repairability.AUTOMATIC
        assert decision.handler == CorrectionHandler.CONTRACT_REPAIR
        assert decision.actions[0].action_type == "repair_structured_output_contract"

    async def test_integrity_failure_is_not_auto_healed(self, correction_runtime):
        _database, _memory, controller, events = correction_runtime
        verification = VerificationResult(
            tier=1,
            passed=False,
            outcome="fail",
            reason_code="artifact_seal_invalid",
            severity_class="hard_invariant",
            feedback="A sealed artifact changed without an authorized reseal.",
        )

        decision = await controller.record_failure(
            task_id="task-1",
            run_id="run-1",
            subtask_id="publish",
            result=_result(),
            verification=verification,
        )

        assert decision.repairability == Repairability.TERMINAL
        assert decision.handler == CorrectionHandler.NONE
        assert decision.state == CorrectionState.TERMINAL
        assert any(event_type == "correction_terminal" for event_type, _, _ in events)

    async def test_workspace_escape_remains_terminal(self, correction_runtime):
        _database, _memory, controller, _events = correction_runtime
        verification = VerificationResult(
            tier=1,
            passed=False,
            outcome="fail",
            reason_code="path_policy_violation",
            severity_class="hard_invariant",
            feedback="Output attempted a path traversal outside workspace.",
        )

        decision = await controller.record_failure(
            task_id="task-1",
            run_id="run-1",
            subtask_id="publish",
            result=_result(),
            verification=verification,
        )

        assert decision.repairability == Repairability.TERMINAL
        assert decision.handler == CorrectionHandler.NONE

    async def test_structured_no_progress_survives_controller_restart(
        self,
        correction_runtime,
    ):
        _database, memory, controller, _events = correction_runtime
        verification = VerificationResult(
            tier=1,
            passed=False,
            outcome="fail",
            reason_code="tool_runtime_retryable",
            severity_class="infra",
            feedback="first wording",
            metadata={"missing_targets": ["https://members.example.test/"]},
        )
        first = await controller.record_failure(
            task_id="task-1",
            run_id="run-1",
            subtask_id="scope-company",
            result=_result(),
            verification=verification,
        )
        restarted = CorrectionController(memory, lambda *_args: None, max_no_progress_attempts=2)
        verification.feedback = "different wording for exactly the same blocker"
        second = await restarted.record_failure(
            task_id="task-1",
            run_id="run-2",
            subtask_id="scope-company",
            result=_result(),
            verification=verification,
        )
        third = await restarted.record_failure(
            task_id="task-1",
            run_id="run-2",
            subtask_id="scope-company",
            result=_result(),
            verification=verification,
        )

        assert first.cycle_id == second.cycle_id == third.cycle_id
        assert second.no_progress_count == 1
        assert third.stop_for_no_progress is True
        assert third.state == CorrectionState.TERMINAL

    async def test_success_resolves_open_cycles(self, correction_runtime):
        database, memory, controller, events = correction_runtime
        verification = VerificationResult(
            tier=1,
            passed=False,
            outcome="fail",
            reason_code="tool_runtime_retryable",
            severity_class="infra",
        )
        decision = await controller.record_failure(
            task_id="task-1",
            run_id="run-1",
            subtask_id="research",
            result=_result(),
            verification=verification,
        )

        await controller.resolve_subtask(task_id="task-1", subtask_id="research")

        cycle = await memory.get_correction_cycle(cycle_id=decision.cycle_id)
        assert cycle is not None
        assert cycle["state"] == "resolved"
        assert cycle["resolved_at"]
        attempts = await database.query(
            "SELECT state, outcome FROM correction_attempts WHERE correction_id=?",
            (decision.cycle_id,),
        )
        actions = await database.query(
            "SELECT state FROM correction_actions WHERE correction_id=?",
            (decision.cycle_id,),
        )
        assert attempts[-1]["state"] == "resolved"
        assert attempts[-1]["outcome"] == "resolved"
        assert actions[-1]["state"] == "completed"
        assert any(event_type == "correction_resolved" for event_type, _, _ in events)

    async def test_budget_reason_aliases_share_checkpoint_cycle(
        self,
        correction_runtime,
    ):
        _database, _memory, controller, _events = correction_runtime
        first = await controller.record_failure(
            task_id="task-1",
            run_id="run-1",
            subtask_id="risk-register",
            result=_result(),
            verification=VerificationResult(
                tier=2,
                passed=False,
                outcome="fail",
                reason_code="iteration_budget_exceeded",
                severity_class="infra",
                feedback="partial register remains",
                metadata={"missing_targets": ["RISK-006", "RISK-007"]},
            ),
        )
        second = await controller.record_failure(
            task_id="task-1",
            run_id="run-2",
            subtask_id="risk-register",
            result=_result(),
            verification=VerificationResult(
                tier=2,
                passed=False,
                outcome="fail",
                reason_code="tool_budget_exhausted",
                severity_class="infra",
                feedback="worded differently",
                metadata={"missing_targets": ["RISK-006"]},
            ),
        )

        assert first.cycle_id == second.cycle_id
        assert first.handler == CorrectionHandler.CHECKPOINT_CONTINUE
        assert second.handler == CorrectionHandler.CHECKPOINT_CONTINUE
        assert second.progress_made is True
        assert second.actions[0].action_type == "continue_from_partial_checkpoint"
