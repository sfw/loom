"""Tests for verification profile auto-classification."""

from __future__ import annotations

from loom.engine.orchestrator import profile as orchestrator_profile
from loom.processes.schema import ProcessDefinition, VerificationPolicyContract
from loom.state.task_state import Subtask, Task


def _task(goal: str) -> Task:
    return Task(id="task-profile", goal=goal, workspace="/tmp")


def test_resolve_verification_profile_prefers_coding_signals() -> None:
    task = _task("Fix failing tests and lint errors")
    subtask = Subtask(
        id="s1",
        description="Run pytest and patch failing modules",
        acceptance_criteria="All tests pass and lint clean",
    )
    result = orchestrator_profile.resolve_verification_profile(
        task=task,
        subtask=subtask,
        process=None,
        tool_calls=[],
    )
    assert result.profile == "coding"
    assert result.confidence >= 0.6


def test_resolve_verification_profile_uses_hybrid_when_signals_sparse() -> None:
    task = _task("Do the thing")
    subtask = Subtask(
        id="s2",
        description="General ad hoc task",
    )
    result = orchestrator_profile.resolve_verification_profile(
        task=task,
        subtask=subtask,
        process=None,
        tool_calls=[],
    )
    assert result.profile == "hybrid"
    assert result.fallback_profile == "hybrid"


def test_resolve_verification_profile_finance_process_biases_research() -> None:
    process = ProcessDefinition(name="finance", tags=["finance"])
    task = _task("Produce final investment memo")
    subtask = Subtask(
        id="s3",
        description="Synthesize recommendation with sources",
        is_synthesis=True,
    )
    result = orchestrator_profile.resolve_verification_profile(
        task=task,
        subtask=subtask,
        process=process,
        tool_calls=[],
    )
    assert result.profile == "research"
    assert result.confidence >= 0.65


def test_resolve_verification_profile_uses_dev_policy_signal_for_build_process() -> None:
    process = ProcessDefinition(
        name="build-process",
        tags=["adhoc", "generated", "build"],
        verification_policy=VerificationPolicyContract(
            mode="static_first",
            static_checks={"tool_success_policy": "development_balanced"},
            semantic_checks=[
                {
                    "name": "optional_browser_verification",
                    "capability": "browser_runtime",
                    "helper": "browser_assert",
                    "optional": True,
                },
                {
                    "name": "required_service_probe",
                    "capability": "service_runtime",
                    "helper": "serve_static",
                },
            ],
            outcome_policy={"optional_capabilities": ["browser_runtime"]},
        ),
    )
    task = _task("Do the thing")
    subtask = Subtask(
        id="s4",
        description="General ad hoc task",
    )
    result = orchestrator_profile.resolve_verification_profile(
        task=task,
        subtask=subtask,
        process=process,
        tool_calls=[],
    )
    assert result.profile == "coding"
    assert result.confidence >= 0.6
    assert "policy:development_balanced" in result.reason_codes
    assert "policy:dev_required_capabilities" in result.reason_codes
