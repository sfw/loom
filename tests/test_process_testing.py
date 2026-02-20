"""Tests for process test-case selection and execution helpers."""

from __future__ import annotations

import pytest

from loom.processes.schema import (
    PhaseTemplate,
    ProcessDefinition,
    ProcessTestAcceptance,
    ProcessTestCase,
)
from loom.processes.testing import (
    run_process_case_deterministic,
    select_process_test_cases,
)


class TestSelectProcessTestCases:
    def test_default_case_when_manifest_missing(self):
        process = ProcessDefinition(
            name="demo",
            phases=[
                PhaseTemplate(
                    id="phase-a",
                    description="Phase A",
                    deliverables=["out.md — output"],
                ),
            ],
        )
        selected = select_process_test_cases(process)
        assert len(selected) == 1
        case = selected[0]
        assert case.id == "smoke"
        assert case.mode == "deterministic"
        assert "phase-a" in case.acceptance.phases_must_include
        assert "out.md" in case.acceptance.deliverables_must_exist

    def test_case_filter_by_id(self):
        process = ProcessDefinition(
            name="demo",
            tests=[
                ProcessTestCase(
                    id="deterministic",
                    mode="deterministic",
                    goal="Run deterministic case",
                ),
                ProcessTestCase(
                    id="live",
                    mode="live",
                    goal="Run live case",
                ),
            ],
        )
        selected = select_process_test_cases(
            process,
            include_live=True,
            case_id="live",
        )
        assert len(selected) == 1
        assert selected[0].id == "live"

    def test_live_cases_filtered_by_default(self):
        process = ProcessDefinition(
            name="demo",
            tests=[
                ProcessTestCase(
                    id="live",
                    mode="live",
                    goal="Run live case",
                ),
            ],
        )
        selected = select_process_test_cases(process)
        assert selected == []

    def test_missing_case_id_raises(self):
        process = ProcessDefinition(
            name="demo",
            tests=[
                ProcessTestCase(
                    id="smoke",
                    mode="deterministic",
                    goal="Run smoke case",
                ),
            ],
        )
        with pytest.raises(ValueError, match="Process test case not found"):
            select_process_test_cases(process, case_id="unknown")


class TestRunProcessCaseDeterministic:
    @pytest.mark.asyncio
    @pytest.mark.process_integration
    async def test_runs_and_writes_expected_deliverables(self, tmp_path):
        process = ProcessDefinition(
            name="deterministic-demo",
            phase_mode="strict",
            phases=[
                PhaseTemplate(
                    id="phase-a",
                    description="Phase A",
                    depends_on=[],
                    deliverables=["out.md — markdown output"],
                ),
                PhaseTemplate(
                    id="phase-b",
                    description="Phase B",
                    depends_on=["phase-a"],
                    deliverables=["metrics.csv — csv output"],
                ),
            ],
        )
        case = ProcessTestCase(
            id="smoke",
            mode="deterministic",
            goal="Run deterministic process",
            acceptance=ProcessTestAcceptance(
                phases_must_include=["phase-a", "phase-b"],
                deliverables_must_exist=["out.md", "metrics.csv"],
            ),
        )

        workspace = tmp_path / "workspace"
        result = await run_process_case_deterministic(
            process,
            case,
            workspace=workspace,
        )
        assert result.passed, "\n".join([result.message, *result.details])
        assert (workspace / "out.md").exists()
        assert (workspace / "metrics.csv").exists()

    @pytest.mark.asyncio
    @pytest.mark.process_integration
    async def test_fails_when_required_tool_missing(self, tmp_path):
        process = ProcessDefinition(
            name="deterministic-demo",
            phase_mode="strict",
            phases=[
                PhaseTemplate(
                    id="phase-a",
                    description="Phase A",
                    depends_on=[],
                    deliverables=["out.md — markdown output"],
                ),
            ],
        )
        case = ProcessTestCase(
            id="smoke",
            mode="deterministic",
            goal="Run deterministic process",
            requires_tools=["tool_does_not_exist"],
            acceptance=ProcessTestAcceptance(
                phases_must_include=["phase-a"],
                deliverables_must_exist=["out.md"],
            ),
        )

        result = await run_process_case_deterministic(
            process,
            case,
            workspace=tmp_path / "workspace",
        )
        assert not result.passed
        assert result.task_status == "tooling_error"
        assert any("tool_does_not_exist" in detail for detail in result.details)

    @pytest.mark.asyncio
    @pytest.mark.process_integration
    async def test_forbidden_pattern_trips_acceptance(self, tmp_path):
        process = ProcessDefinition(
            name="deterministic-demo",
            phase_mode="strict",
            phases=[
                PhaseTemplate(
                    id="phase-a",
                    description="Phase A",
                    depends_on=[],
                    deliverables=["out.md — markdown output"],
                ),
            ],
        )
        case = ProcessTestCase(
            id="smoke",
            mode="deterministic",
            goal="Run deterministic process",
            acceptance=ProcessTestAcceptance(
                phases_must_include=["phase-a"],
                deliverables_must_exist=["out.md"],
                verification_forbidden_patterns=["Completed\\s+phase-a"],
            ),
        )

        result = await run_process_case_deterministic(
            process,
            case,
            workspace=tmp_path / "workspace",
        )
        assert not result.passed
        assert any(
            "Forbidden verification pattern matched" in detail
            for detail in result.details
        )
