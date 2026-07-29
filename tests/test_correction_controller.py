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
        assert "edit the existing structured output in place" in (
            decision.actions[0].arguments["guardrails"]
        )

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
