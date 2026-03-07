"""Orchestrator validity policy tests."""

from __future__ import annotations

import hashlib
import json
from unittest.mock import AsyncMock

import pytest

from loom.config import Config, ExecutionConfig
from loom.engine.orchestrator import (
    Orchestrator,
    SubtaskResult,
    SubtaskResultStatus,
    ToolCallRecord,
)
from loom.engine.verification import VerificationResult
from loom.events.types import (
    ARTIFACT_SEAL_VALIDATION,
    CLAIMS_PRUNED,
    RUN_VALIDITY_SCORECARD,
    SUBTASK_POLICY_RECONCILED,
    SYNTHESIS_INPUT_GATE_DECISION,
)
from loom.processes.schema import PhaseTemplate, ProcessDefinition
from loom.state.task_state import Plan, Subtask, SubtaskStatus, TaskStatus
from loom.tools.registry import ToolResult
from tests.orchestrator.conftest import (
    _make_config,
    _make_event_bus,
    _make_mock_memory,
    _make_mock_prompts,
    _make_mock_router,
    _make_mock_tools,
    _make_state_manager,
    _make_task,
)


class TestOrchestratorValidityPolicy:
    @pytest.mark.asyncio
    async def test_resume_reconciles_policy_and_prevents_silent_tier_downgrade(self, tmp_path):
        bus = _make_event_bus()
        events = []
        bus.subscribe_all(lambda event: events.append(event))
        state_manager = _make_state_manager(tmp_path)
        process = ProcessDefinition(
            name="resume-validity",
            validity_contract={
                "enabled": True,
                "claim_extraction": {"enabled": False},
                "final_gate": {
                    "enforce_verified_context_only": False,
                    "synthesis_min_verification_tier": 3,
                    "critical_claim_support_ratio": 1.0,
                },
            },
            phases=[
                PhaseTemplate(
                    id="final",
                    description="Final synthesis",
                    is_synthesis=True,
                    model_tier=2,
                    verification_tier=1,
                    acceptance_criteria="Ground every final claim",
                ),
            ],
        )
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=bus,
            config=_make_config(),
            process=process,
        )
        observed: dict[str, int] = {}

        async def _run(_task, dispatched_subtask, **_kwargs):
            observed["model_tier"] = int(dispatched_subtask.model_tier)
            observed["verification_tier"] = int(dispatched_subtask.verification_tier)
            return (
                SubtaskResult(status="success", summary="ok"),
                VerificationResult(tier=3, passed=True, outcome="pass"),
            )

        orch._runner.run = AsyncMock(side_effect=_run)
        task = _make_task(goal="Resume run")
        task.status = TaskStatus.FAILED
        task.plan = Plan(
            subtasks=[
                Subtask(
                    id="final",
                    description="Synthesize final output",
                    phase_id="final",
                    is_synthesis=True,
                    model_tier=1,
                    verification_tier=1,
                    acceptance_criteria="",
                    validity_contract_snapshot={},
                    validity_contract_hash="",
                ),
            ],
        )
        state_manager.create(task)

        resumed = state_manager.load(task.id)
        result = await orch.execute_task(resumed, reuse_existing_plan=True)
        reconciled_subtask = result.get_subtask("final")

        assert result.status == TaskStatus.COMPLETED
        assert observed["model_tier"] == 2
        assert observed["verification_tier"] == 3
        assert reconciled_subtask is not None
        assert reconciled_subtask.acceptance_criteria == "Ground every final claim"
        assert reconciled_subtask.validity_contract_hash
        restored = state_manager.load(task.id).get_subtask("final")
        assert restored is not None
        assert restored.verification_tier == 3
        assert restored.model_tier == 2
        assert restored.validity_contract_hash
        assert any(
            event.event_type == SUBTASK_POLICY_RECONCILED
            for event in events
        )

    @pytest.mark.asyncio
    async def test_regression_cowork_3df9d4dd_guardrail_chain_blocks_invalid_final_synthesis(
        self,
        tmp_path,
    ):
        bus = _make_event_bus()
        events = []
        bus.subscribe_all(lambda event: events.append(event))
        state_manager = _make_state_manager(tmp_path)
        process = ProcessDefinition(
            name="finance-guardrail",
            tags=["finance"],
            phases=[
                PhaseTemplate(
                    id="final",
                    description="Final synthesis",
                    is_synthesis=True,
                    model_tier=2,
                    verification_tier=1,
                ),
            ],
        )
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=bus,
            config=Config(execution=ExecutionConfig(max_subtask_retries=0)),
            process=process,
        )
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(status="success", summary="should never run"),
            VerificationResult(tier=3, passed=True, outcome="pass"),
        ))

        task = _make_task(goal="Resume regression guard")
        task.status = TaskStatus.FAILED
        task.metadata["claim_graph"] = {
            "supported_by_subtask": {},
            "unresolved_by_subtask": {
                "analysis": [
                    {
                        "claim_id": "CLM-UNRESOLVED",
                        "text": "Unresolved critical claim",
                        "status": "insufficient_evidence",
                    },
                ],
            },
        }
        task.plan = Plan(
            subtasks=[
                Subtask(
                    id="final",
                    description="Finalize recommendation",
                    phase_id="final",
                    is_synthesis=True,
                    is_critical_path=True,
                    model_tier=1,
                    verification_tier=1,
                    validity_contract_snapshot={},
                    validity_contract_hash="",
                    max_retries=0,
                ),
            ],
        )
        state_manager.create(task)

        resumed = state_manager.load(task.id)
        result = await orch.execute_task(resumed, reuse_existing_plan=True)

        assert result.status == TaskStatus.FAILED
        orch._runner.run.assert_not_awaited()
        gate_events = [
            event for event in events
            if event.event_type == SYNTHESIS_INPUT_GATE_DECISION
        ]
        assert gate_events
        assert gate_events[-1].data.get("passed") is False
        assert gate_events[-1].data.get("unresolved_claim_count") == 1
        assert any(
            event.event_type == SUBTASK_POLICY_RECONCILED
            for event in events
        )

    @pytest.mark.asyncio
    async def test_synthesis_input_gate_blocks_when_only_unresolved_claims_exist(self, tmp_path):
        bus = _make_event_bus()
        events = []
        bus.subscribe_all(lambda event: events.append(event))
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=bus,
            config=_make_config(),
        )
        task = _make_task(goal="Gate blocked")
        task.metadata["claim_graph"] = {
            "supported_by_subtask": {},
            "unresolved_by_subtask": {
                "analyze": [
                    {
                        "claim_id": "CLM-UNRESOLVED",
                        "text": "Unresolved critical claim",
                        "status": "insufficient_evidence",
                    },
                ],
            },
        }
        subtask = Subtask(
            id="synth",
            description="Synthesize",
            is_synthesis=True,
            verification_tier=1,
            validity_contract_snapshot={
                "enabled": True,
                "claim_extraction": {"enabled": True},
                "final_gate": {
                    "enforce_verified_context_only": True,
                    "synthesis_min_verification_tier": 2,
                    "critical_claim_support_ratio": 1.0,
                },
            },
        )
        task.plan = Plan(subtasks=[subtask])
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(status="success", summary="unexpected"),
            VerificationResult(tier=2, passed=True, outcome="pass"),
        ))

        _, result, verification = await orch._dispatch_subtask(task, subtask, {})

        assert result.status == SubtaskResultStatus.FAILED
        assert verification.passed is False
        assert verification.reason_code == "coverage_below_threshold"
        orch._runner.run.assert_not_awaited()
        gate_events = [
            event for event in events
            if event.event_type == SYNTHESIS_INPUT_GATE_DECISION
        ]
        assert gate_events
        assert gate_events[-1].data.get("passed") is False

    @pytest.mark.asyncio
    async def test_synthesis_input_gate_injects_supported_claims_only(self, tmp_path):
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        task = _make_task(goal="Gate context")
        task.metadata["claim_graph"] = {
            "supported_by_subtask": {
                "analyze": [
                    {
                        "claim_id": "CLM-SUPPORTED",
                        "text": "Supported fact from upstream analysis",
                        "status": "supported",
                    },
                ],
            },
            "unresolved_by_subtask": {
                "analyze": [
                    {
                        "claim_id": "CLM-UNRESOLVED",
                        "text": "Unresolved rumor from draft analysis",
                        "status": "insufficient_evidence",
                    },
                ],
            },
        }
        subtask = Subtask(
            id="synth",
            description="Synthesize",
            is_synthesis=True,
            validity_contract_snapshot={
                "enabled": True,
                "claim_extraction": {"enabled": True},
                "final_gate": {
                    "enforce_verified_context_only": True,
                    "synthesis_min_verification_tier": 2,
                    "critical_claim_support_ratio": 1.0,
                },
            },
        )
        task.plan = Plan(subtasks=[subtask])
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(status="success", summary="ok"),
            VerificationResult(tier=2, passed=True, outcome="pass"),
        ))

        await orch._dispatch_subtask(task, subtask, {})
        retry_context = str(orch._runner.run.await_args.kwargs.get("retry_context", ""))

        assert "Supported fact from upstream analysis" in retry_context
        assert "Unresolved rumor from draft analysis" not in retry_context

    @pytest.mark.asyncio
    async def test_required_fact_checker_contract_enforces_failure_when_missing(self, tmp_path):
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        task = _make_task(goal="Fact checker required")
        subtask = Subtask(
            id="synth",
            description="Synthesize",
            is_synthesis=True,
            validity_contract_snapshot={
                "enabled": True,
                "claim_extraction": {"enabled": False},
                "require_fact_checker_for_synthesis": True,
                "final_gate": {
                    "enforce_verified_context_only": False,
                    "synthesis_min_verification_tier": 2,
                    "critical_claim_support_ratio": 1.0,
                },
            },
        )
        task.plan = Plan(subtasks=[subtask])
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(
                status="success",
                summary="Synthesized without grounding",
                tool_calls=[],
            ),
            VerificationResult(tier=2, passed=True, outcome="pass"),
        ))

        _, result, verification = await orch._dispatch_subtask(task, subtask, {})

        assert result.status == SubtaskResultStatus.FAILED
        assert verification.passed is False
        assert verification.reason_code == "required_verifier_missing"
        assert "fact grounding" in str(verification.feedback).lower()

    @pytest.mark.asyncio
    async def test_required_fact_checker_contract_requires_verdicts_when_claim_extraction_enabled(
        self,
        tmp_path,
    ):
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        task = _make_task(goal="Fact checker verdicts required")
        subtask = Subtask(
            id="synth",
            description="Synthesize",
            is_synthesis=True,
            validity_contract_snapshot={
                "enabled": True,
                "claim_extraction": {"enabled": True},
                "require_fact_checker_for_synthesis": True,
                "final_gate": {
                    "enforce_verified_context_only": False,
                    "synthesis_min_verification_tier": 2,
                    "critical_claim_support_ratio": 1.0,
                },
            },
        )
        task.plan = Plan(subtasks=[subtask])
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(
                status="success",
                summary="Synthesized with empty fact checker",
                tool_calls=[
                    ToolCallRecord(
                        tool="fact_checker",
                        args={"claims": ["A claim"]},
                        result=ToolResult.ok("ok", data={"verdicts": []}),
                    ),
                ],
            ),
            VerificationResult(tier=2, passed=True, outcome="pass"),
        ))

        _, result, verification = await orch._dispatch_subtask(task, subtask, {})

        assert result.status == SubtaskResultStatus.FAILED
        assert verification.passed is False
        assert verification.reason_code == "required_verifier_empty"
        assert verification.metadata.get("required_verifier_empty") is True

    @pytest.mark.asyncio
    async def test_synthesis_gate_rejects_orphan_critical_numeric_claims(self, tmp_path):
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        task = _make_task(goal="Numeric lineage")
        subtask = Subtask(
            id="synth",
            description="Synthesize",
            is_synthesis=True,
            validity_contract_snapshot={
                "enabled": True,
                "claim_extraction": {"enabled": True},
                "final_gate": {
                    "enforce_verified_context_only": False,
                    "synthesis_min_verification_tier": 2,
                    "critical_claim_support_ratio": 1.0,
                },
            },
        )
        task.plan = Plan(subtasks=[subtask])
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(
                status="success",
                summary="Numeric recommendation complete",
                tool_calls=[
                    ToolCallRecord(
                        tool="fact_checker",
                        args={"claims": ["EPS grows 20%"]},
                        result=ToolResult.ok("grounded"),
                    ),
                ],
            ),
            VerificationResult(
                tier=2,
                passed=True,
                outcome="pass",
                metadata={
                    "claim_lifecycle": [
                        {
                            "claim_id": "CLM-NUM-1",
                            "text": "EPS grows 20% in FY2027.",
                            "claim_type": "numeric",
                            "criticality": "critical",
                            "status": "supported",
                            "reason_code": "claim_supported",
                            "evidence_refs": [],
                            "lifecycle": ["extracted", "supported"],
                        },
                    ],
                },
            ),
        ))

        _, result, verification = await orch._dispatch_subtask(task, subtask, {})

        assert result.status == SubtaskResultStatus.FAILED
        assert verification.passed is False
        assert verification.reason_code == "claim_insufficient_evidence"
        assert verification.metadata.get("orphan_critical_numeric_claims")

    @pytest.mark.asyncio
    async def test_synthesis_temporal_gate_rejects_stale_supported_claims(self, tmp_path):
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        task = _make_task(goal="Temporal stale check")
        subtask = Subtask(
            id="synth",
            description="Synthesize",
            is_synthesis=True,
            validity_contract_snapshot={
                "enabled": True,
                "claim_extraction": {"enabled": True},
                "final_gate": {
                    "enforce_verified_context_only": False,
                    "synthesis_min_verification_tier": 2,
                    "critical_claim_support_ratio": 1.0,
                    "temporal_consistency": {
                        "enabled": True,
                        "require_as_of_alignment": False,
                        "enforce_cross_claim_date_conflict_check": False,
                        "max_source_age_days": 30,
                        "as_of": "2026-03-05",
                    },
                },
            },
        )
        task.plan = Plan(subtasks=[subtask])
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(status="success", summary="Temporal synthesis"),
            VerificationResult(
                tier=2,
                passed=True,
                outcome="pass",
                metadata={
                    "claim_lifecycle": [
                        {
                            "claim_id": "CLM-DATE-1",
                            "text": "Revenue is stable.",
                            "claim_type": "date",
                            "criticality": "critical",
                            "status": "supported",
                            "reason_code": "claim_supported",
                            "as_of": "2025-01-01",
                            "evidence_refs": ["EV-1"],
                            "lifecycle": ["extracted", "supported"],
                        },
                    ],
                },
            ),
        ))

        _, result, verification = await orch._dispatch_subtask(task, subtask, {})

        assert result.status == SubtaskResultStatus.FAILED
        assert verification.passed is False
        assert verification.reason_code == "claim_stale_source"
        assert verification.metadata.get("temporal_consistency", {}).get("stale_claim_count") == 1

    @pytest.mark.asyncio
    async def test_synthesis_temporal_gate_rejects_as_of_conflicts(self, tmp_path):
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        task = _make_task(goal="Temporal conflict check")
        subtask = Subtask(
            id="synth",
            description="Synthesize",
            is_synthesis=True,
            validity_contract_snapshot={
                "enabled": True,
                "claim_extraction": {"enabled": True},
                "final_gate": {
                    "enforce_verified_context_only": False,
                    "synthesis_min_verification_tier": 2,
                    "critical_claim_support_ratio": 1.0,
                    "temporal_consistency": {
                        "enabled": True,
                        "require_as_of_alignment": True,
                        "enforce_cross_claim_date_conflict_check": True,
                        "max_source_age_days": 0,
                        "as_of": "2026-03-05",
                    },
                },
            },
        )
        task.plan = Plan(subtasks=[subtask])
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(status="success", summary="Temporal synthesis"),
            VerificationResult(
                tier=2,
                passed=True,
                outcome="pass",
                metadata={
                    "claim_lifecycle": [
                        {
                            "claim_id": "CLM-DATE-A",
                            "text": "Guidance as of 2026-01-15 indicates growth.",
                            "claim_type": "date",
                            "criticality": "critical",
                            "status": "supported",
                            "reason_code": "claim_supported",
                            "as_of": "2026-01-15",
                            "evidence_refs": ["EV-A"],
                            "lifecycle": ["extracted", "supported"],
                        },
                        {
                            "claim_id": "CLM-DATE-B",
                            "text": "Guidance as of 2026-02-20 indicates growth.",
                            "claim_type": "date",
                            "criticality": "critical",
                            "status": "supported",
                            "reason_code": "claim_supported",
                            "as_of": "2026-02-20",
                            "evidence_refs": ["EV-B"],
                            "lifecycle": ["extracted", "supported"],
                        },
                    ],
                },
            ),
        ))

        _, result, verification = await orch._dispatch_subtask(task, subtask, {})

        assert result.status == SubtaskResultStatus.FAILED
        assert verification.passed is False
        assert verification.reason_code == "temporal_conflict"
        assert verification.metadata.get("temporal_consistency", {}).get("conflict_count", 0) >= 1

    @pytest.mark.asyncio
    async def test_intermediate_claim_pruning_is_non_fatal_when_thresholds_hold(self, tmp_path):
        bus = _make_event_bus()
        events = []
        bus.subscribe_all(lambda event: events.append(event))
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=bus,
            config=_make_config(),
        )
        task = _make_task(goal="Intermediate prune")
        subtask = Subtask(
            id="analysis",
            description="Analyze evidence",
            is_synthesis=False,
            validity_contract_snapshot={
                "enabled": True,
                "claim_extraction": {"enabled": True},
                "min_supported_ratio": 0.4,
                "prune_mode": "rewrite_uncertainty",
            },
        )
        task.plan = Plan(subtasks=[subtask])
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(
                status="failed",
                summary="Contains unsupported claim",
            ),
            VerificationResult(
                tier=2,
                passed=False,
                outcome="fail",
                reason_code="claim_insufficient_evidence",
                feedback="Unsupported claims remain.",
                metadata={
                    "claim_lifecycle": [
                        {
                            "claim_id": "CLM-SUPPORTED",
                            "text": "Supported claim",
                            "claim_type": "qualitative",
                            "criticality": "important",
                            "status": "supported",
                            "reason_code": "claim_supported",
                            "evidence_refs": [],
                            "lifecycle": ["extracted", "supported"],
                        },
                        {
                            "claim_id": "CLM-UNSUPPORTED",
                            "text": "Unsupported claim",
                            "claim_type": "qualitative",
                            "criticality": "important",
                            "status": "insufficient_evidence",
                            "reason_code": "claim_insufficient_evidence",
                            "evidence_refs": [],
                            "lifecycle": ["extracted", "insufficient_evidence"],
                        },
                    ],
                },
            ),
        ))

        _, result, verification = await orch._dispatch_subtask(task, subtask, {})

        assert result.status == SubtaskResultStatus.SUCCESS
        assert verification.passed is True
        assert verification.reason_code == "claim_pruned"
        assert verification.metadata.get("claim_pruned") is True
        assert verification.metadata.get("claim_pruned_count") == 1
        assert any(
            event.event_type == CLAIMS_PRUNED
            for event in events
        )

    @pytest.mark.asyncio
    async def test_intermediate_claim_pruning_fails_when_post_prune_thresholds_fail(self, tmp_path):
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        task = _make_task(goal="Intermediate prune thresholds")
        subtask = Subtask(
            id="analysis",
            description="Analyze evidence",
            is_synthesis=False,
            validity_contract_snapshot={
                "enabled": True,
                "claim_extraction": {"enabled": True},
                "min_supported_ratio": 0.8,
                "max_unverified_ratio": 0.2,
                "max_contradicted_count": 0,
            },
        )
        task.plan = Plan(subtasks=[subtask])
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(
                status="failed",
                summary="Contains unsupported claim",
            ),
            VerificationResult(
                tier=2,
                passed=False,
                outcome="fail",
                reason_code="claim_insufficient_evidence",
                feedback="Unsupported claims remain.",
                metadata={
                    "claim_lifecycle": [
                        {
                            "claim_id": "CLM-UNSUPPORTED",
                            "text": "Unsupported claim",
                            "claim_type": "qualitative",
                            "criticality": "important",
                            "status": "insufficient_evidence",
                            "reason_code": "claim_insufficient_evidence",
                            "evidence_refs": [],
                            "lifecycle": ["extracted", "insufficient_evidence"],
                        },
                    ],
                },
            ),
        ))

        _, result, verification = await orch._dispatch_subtask(task, subtask, {})

        assert result.status == SubtaskResultStatus.FAILED
        assert verification.passed is False
        assert verification.reason_code == "coverage_below_threshold"
        assert verification.metadata.get("post_prune_gate_passed") is False

    @pytest.mark.asyncio
    async def test_claim_artifact_persistence_links_supported_claim_to_write_path_provenance(
        self,
        tmp_path,
    ):
        memory = _make_mock_memory()
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=memory,
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        task = _make_task(goal="Persist claim evidence")
        subtask = Subtask(
            id="analysis",
            description="Analyze",
            phase_id="analysis",
            validity_contract_snapshot={
                "enabled": True,
                "claim_extraction": {"enabled": True},
            },
        )
        task.plan = Plan(subtasks=[subtask])
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(
                status="success",
                summary="ok",
                tool_calls=[
                    ToolCallRecord(
                        tool="write_file",
                        args={"path": "analysis.md", "content": "Evidence-backed analysis"},
                        result=ToolResult.ok("ok"),
                    ),
                ],
                evidence_records=[],
            ),
            VerificationResult(
                tier=2,
                passed=True,
                outcome="pass",
                metadata={
                    "claim_lifecycle": [
                        {
                            "claim_id": "CLM-001",
                            "text": "Revenue increased year over year.",
                            "claim_type": "numeric",
                            "criticality": "critical",
                            "status": "supported",
                            "reason_code": "claim_supported",
                            "evidence_refs": ["analysis.md"],
                            "lifecycle": ["extracted", "supported"],
                        },
                    ],
                },
            ),
        ))

        await orch._dispatch_subtask(task, subtask, {})

        memory.insert_artifact_claims.assert_awaited()
        memory.insert_claim_verification_results.assert_awaited()
        memory.insert_claim_evidence_links.assert_awaited()
        links = memory.insert_claim_evidence_links.await_args.kwargs["links"]
        assert links
        assert links[0]["claim_id"] == "CLM-001"
        assert str(links[0]["evidence_id"]).startswith("EV-WRITE-")

    def test_persist_subtask_evidence_captures_write_path_provenance(self, tmp_path):
        state_manager = _make_state_manager(tmp_path)
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        task = _make_task(goal="Evidence provenance")
        state_manager.create(task)
        content = "Final synthesis artifact"
        tool_call = ToolCallRecord(
            tool="write_file",
            args={"path": "reports/final.md", "content": content},
            result=ToolResult.ok("wrote reports/final.md"),
            timestamp="2026-03-05T12:00:00",
        )

        orch._persist_subtask_evidence(
            task.id,
            "s1",
            [],
            tool_calls=[tool_call],
        )
        records = state_manager.load_evidence_records(task.id)
        write_records = [
            record for record in records
            if str(record.get("tool", "")).strip().lower() == "write_file"
        ]

        assert write_records
        record = write_records[0]
        assert record.get("artifact_workspace_relpath") == "reports/final.md"
        assert record.get("artifact_size_bytes") == len(content.encode("utf-8"))
        assert record.get("artifact_sha256") == hashlib.sha256(
            content.encode("utf-8"),
        ).hexdigest()

    @pytest.mark.asyncio
    async def test_synthesis_gate_blocks_on_artifact_seal_mismatch(self, tmp_path):
        bus = _make_event_bus()
        events = []
        bus.subscribe_all(lambda event: events.append(event))
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=bus,
            config=_make_config(),
        )
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        artifact = workspace / "reports" / "final.md"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("tampered", encoding="utf-8")

        task = _make_task(goal="Seal gate", workspace=str(workspace))
        task.metadata["artifact_seals"] = {
            "reports/final.md": {
                "sha256": hashlib.sha256(b"original").hexdigest(),
            },
        }
        subtask = Subtask(
            id="synth",
            description="Synthesize",
            is_synthesis=True,
            validity_contract_snapshot={
                "enabled": True,
                "claim_extraction": {"enabled": False},
                "final_gate": {
                    "enforce_verified_context_only": False,
                    "synthesis_min_verification_tier": 2,
                    "critical_claim_support_ratio": 1.0,
                },
            },
        )
        task.plan = Plan(subtasks=[subtask])
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(status="success", summary="unexpected"),
            VerificationResult(tier=2, passed=True, outcome="pass"),
        ))

        _, result, verification = await orch._dispatch_subtask(task, subtask, {})

        assert result.status == SubtaskResultStatus.FAILED
        assert verification.passed is False
        assert verification.reason_code == "artifact_seal_invalid"
        orch._runner.run.assert_not_awaited()
        seal_events = [
            event for event in events
            if event.event_type == ARTIFACT_SEAL_VALIDATION
        ]
        assert seal_events
        assert seal_events[-1].data.get("passed") is False
        assert seal_events[-1].data.get("mismatch_count", 0) >= 1

    @pytest.mark.asyncio
    async def test_synthesis_gate_blocks_on_stale_seals_from_internal_changes(self, tmp_path):
        state_manager = _make_state_manager(tmp_path)
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        artifact = workspace / "reports" / "final.md"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("original", encoding="utf-8")

        task = _make_task(goal="Seal refresh", workspace=str(workspace))
        task.metadata["artifact_seals"] = {
            "reports/final.md": {
                "path": "reports/final.md",
                "sha256": hashlib.sha256(b"original").hexdigest(),
                "sealed_at": "2026-03-05T12:00:00",
                "subtask_id": "seed",
            },
        }
        changelog = orch._get_changelog(task)
        assert changelog is not None
        changelog.record_before_write("reports/final.md", subtask_id="synth")
        artifact.write_text("updated-by-process", encoding="utf-8")

        subtask = Subtask(
            id="synth",
            description="Synthesize",
            is_synthesis=True,
            validity_contract_snapshot={
                "enabled": True,
                "claim_extraction": {"enabled": False},
                "final_gate": {
                    "enforce_verified_context_only": False,
                    "synthesis_min_verification_tier": 2,
                    "critical_claim_support_ratio": 1.0,
                },
            },
        )
        task.plan = Plan(subtasks=[subtask])
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(status="success", summary="ok"),
            VerificationResult(tier=2, passed=True, outcome="pass"),
        ))

        _, result, verification = await orch._dispatch_subtask(task, subtask, {})

        assert result.status == SubtaskResultStatus.FAILED
        assert verification.passed is False
        assert verification.reason_code == "artifact_seal_invalid"
        orch._runner.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_synthesis_success_adds_provenance_footer_and_updates_scorecard(self, tmp_path):
        state_manager = _make_state_manager(tmp_path)
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        task = _make_task(goal="Synthesis provenance", workspace=str(workspace))
        subtask = Subtask(
            id="synth",
            description="Final synthesis",
            is_synthesis=True,
        )
        task.plan = Plan(subtasks=[subtask])
        result = SubtaskResult(
            status="success",
            summary="Final synthesis complete.",
            tool_calls=[
                ToolCallRecord(
                    tool="write_file",
                    args={"path": "reports/final.md", "content": "final output"},
                    result=ToolResult.ok("ok"),
                ),
            ],
            evidence_records=[],
        )
        verification = VerificationResult(
            tier=2,
            passed=True,
            outcome="pass",
            reason_code="claim_supported",
            metadata={
                "claim_status_counts": {
                    "extracted": 1,
                    "supported": 1,
                    "contradicted": 0,
                    "insufficient_evidence": 0,
                    "pruned": 0,
                    "unresolved": 0,
                    "critical_total": 1,
                    "critical_supported": 1,
                    "critical_contradicted": 0,
                },
                "claim_reason_codes": ["claim_supported"],
            },
        )

        await orch._handle_success(task, subtask, result, verification)

        stored = task.get_subtask("synth")
        assert stored is not None
        assert "VALIDITY_PROVENANCE_FOOTER:" in stored.summary
        scorecard = task.metadata.get("validity_scorecard", {}).get("run", {})
        assert scorecard.get("counts", {}).get("supported") == 1
        assert scorecard.get("verification_report_path") == "validity-scorecard.json"
        seals = task.metadata.get("artifact_seals", {})
        assert "reports/final.md" in seals

    def test_finalize_emits_run_validity_scorecard_and_writes_report(self, tmp_path):
        bus = _make_event_bus()
        events = []
        bus.subscribe_all(lambda event: events.append(event))
        state_manager = _make_state_manager(tmp_path)
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=bus,
            config=_make_config(),
        )
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        task = _make_task(goal="Finalize scorecard", workspace=str(workspace))
        task.plan = Plan(
            subtasks=[
                Subtask(
                    id="done",
                    description="done",
                    status=SubtaskStatus.COMPLETED,
                    summary="ok",
                ),
            ],
        )
        task.metadata["validity_scorecard"] = {
            "subtask_metrics": {
                "done": {
                    "counts": {
                        "extracted": 2,
                        "supported": 2,
                        "contradicted": 0,
                        "insufficient_evidence": 0,
                        "pruned": 0,
                        "unresolved": 0,
                        "critical_total": 1,
                        "critical_supported": 1,
                        "critical_contradicted": 0,
                    },
                    "reason_codes": ["claim_supported"],
                },
            },
        }

        finalized = orch._finalize_task(task)

        assert finalized.status == TaskStatus.COMPLETED
        assert any(event.event_type == RUN_VALIDITY_SCORECARD for event in events)
        report_path = workspace / "validity-scorecard.json"
        assert report_path.exists()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload.get("summary", {}).get("counts", {}).get("supported") == 2
