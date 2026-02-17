"""Tests for the three-tier verification gates."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from loom.config import VerificationConfig
from loom.engine.verification import (
    Check,
    DeterministicVerifier,
    LLMVerifier,
    VerificationGates,
    VerificationResult,
    VotingVerifier,
)
from loom.models.base import ModelResponse, TokenUsage
from loom.models.router import ModelRouter, ResponseValidator
from loom.prompts.assembler import PromptAssembler
from loom.state.task_state import Subtask
from loom.tools.registry import ToolResult

# --- Helpers ---


class MockToolCallRecord:
    """Lightweight stand-in for ToolCallRecord in verification tests."""

    def __init__(self, tool: str, args: dict, result: ToolResult):
        self.tool = tool
        self.args = args
        self.result = result


def _make_subtask(
    subtask_id: str = "s1",
    description: str = "Test subtask",
    acceptance_criteria: str = "",
) -> Subtask:
    return Subtask(
        id=subtask_id,
        description=description,
        acceptance_criteria=acceptance_criteria,
    )


# --- Check and VerificationResult ---


class TestDataStructures:
    def test_check(self):
        c = Check(name="test", passed=True)
        assert c.passed
        assert c.detail is None

    def test_check_with_detail(self):
        c = Check(name="test", passed=False, detail="Something wrong")
        assert not c.passed
        assert c.detail == "Something wrong"

    def test_verification_result_defaults(self):
        r = VerificationResult(tier=1, passed=True)
        assert r.confidence == 1.0
        assert r.checks == []
        assert r.feedback is None


# --- DeterministicVerifier ---


class TestDeterministicVerifier:
    @pytest.mark.asyncio
    async def test_passes_with_no_tool_calls(self):
        v = DeterministicVerifier()
        result = await v.verify(_make_subtask(), "output", [], None)
        assert result.passed
        assert result.tier == 1

    @pytest.mark.asyncio
    async def test_passes_with_successful_tool_calls(self):
        v = DeterministicVerifier()
        tc = MockToolCallRecord(
            tool="write_file",
            args={"path": "test.py"},
            result=ToolResult.ok("ok", files_changed=[]),
        )
        result = await v.verify(_make_subtask(), "output", [tc], None)
        assert result.passed

    @pytest.mark.asyncio
    async def test_fails_with_failed_tool_calls(self):
        v = DeterministicVerifier()
        tc = MockToolCallRecord(
            tool="write_file",
            args={"path": "test.py"},
            result=ToolResult.fail("Permission denied"),
        )
        result = await v.verify(_make_subtask(), "output", [tc], None)
        assert not result.passed
        assert result.feedback is not None
        assert "Permission denied" in result.feedback

    @pytest.mark.asyncio
    async def test_web_tool_transient_failure_is_advisory(self):
        v = DeterministicVerifier()
        tc = MockToolCallRecord(
            tool="web_fetch",
            args={"url": "https://example.com"},
            result=ToolResult.fail("HTTP 403: https://example.com"),
        )
        result = await v.verify(_make_subtask(), "output", [tc], None)
        assert result.passed
        assert any(
            c.name == "tool_web_fetch_advisory" and c.passed
            for c in result.checks
        )

    @pytest.mark.asyncio
    async def test_web_tool_safety_failure_still_blocks(self):
        v = DeterministicVerifier()
        tc = MockToolCallRecord(
            tool="web_fetch",
            args={"url": "http://localhost:8080"},
            result=ToolResult.fail(
                "Blocked host: localhost (private/internal network)",
            ),
        )
        result = await v.verify(_make_subtask(), "output", [tc], None)
        assert not result.passed
        assert any(
            c.name == "tool_web_fetch_success" and not c.passed
            for c in result.checks
        )

    @pytest.mark.asyncio
    async def test_checks_file_nonempty(self, tmp_path):
        # Create a non-empty file
        test_file = tmp_path / "out.py"
        test_file.write_text("print('hello')")

        v = DeterministicVerifier()
        tc = MockToolCallRecord(
            tool="write_file",
            args={"path": "out.py"},
            result=ToolResult.ok("ok", files_changed=["out.py"]),
        )
        result = await v.verify(_make_subtask(), "output", [tc], tmp_path)
        assert result.passed

    @pytest.mark.asyncio
    async def test_fails_on_empty_file(self, tmp_path):
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        v = DeterministicVerifier()
        tc = MockToolCallRecord(
            tool="write_file",
            args={"path": "empty.py"},
            result=ToolResult.ok("ok", files_changed=["empty.py"]),
        )
        result = await v.verify(_make_subtask(), "output", [tc], tmp_path)
        assert not result.passed

    @pytest.mark.asyncio
    async def test_python_syntax_check_passes(self, tmp_path):
        test_file = tmp_path / "good.py"
        test_file.write_text("x = 1\nprint(x)\n")

        v = DeterministicVerifier()
        tc = MockToolCallRecord(
            tool="write_file",
            args={"path": "good.py"},
            result=ToolResult.ok("ok", files_changed=["good.py"]),
        )
        result = await v.verify(_make_subtask(), "output", [tc], tmp_path)
        assert result.passed

    @pytest.mark.asyncio
    async def test_python_syntax_check_fails(self, tmp_path):
        test_file = tmp_path / "bad.py"
        test_file.write_text("def foo(\n")  # Invalid syntax

        v = DeterministicVerifier()
        tc = MockToolCallRecord(
            tool="write_file",
            args={"path": "bad.py"},
            result=ToolResult.ok("ok", files_changed=["bad.py"]),
        )
        result = await v.verify(_make_subtask(), "output", [tc], tmp_path)
        assert not result.passed
        failed_checks = [c for c in result.checks if not c.passed]
        assert any("syntax" in c.name for c in failed_checks)

    @pytest.mark.asyncio
    async def test_json_syntax_check_passes(self, tmp_path):
        test_file = tmp_path / "data.json"
        test_file.write_text('{"key": "value"}')

        v = DeterministicVerifier()
        tc = MockToolCallRecord(
            tool="write_file",
            args={"path": "data.json"},
            result=ToolResult.ok("ok", files_changed=["data.json"]),
        )
        result = await v.verify(_make_subtask(), "output", [tc], tmp_path)
        assert result.passed

    @pytest.mark.asyncio
    async def test_json_syntax_check_fails(self, tmp_path):
        test_file = tmp_path / "bad.json"
        test_file.write_text("{not valid json")

        v = DeterministicVerifier()
        tc = MockToolCallRecord(
            tool="write_file",
            args={"path": "bad.json"},
            result=ToolResult.ok("ok", files_changed=["bad.json"]),
        )
        result = await v.verify(_make_subtask(), "output", [tc], tmp_path)
        assert not result.passed


# --- LLMVerifier ---


class TestLLMVerifier:
    @pytest.mark.asyncio
    async def test_passes_when_model_says_pass(self):
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.complete = AsyncMock(return_value=ModelResponse(
            text=json.dumps({"passed": True, "issues": [], "confidence": 0.9}),
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        ))
        router.select = MagicMock(return_value=model)

        prompts = MagicMock(spec=PromptAssembler)
        prompts.build_verifier_prompt = MagicMock(return_value="Verify this")

        v = LLMVerifier(router, prompts, ResponseValidator())
        result = await v.verify(_make_subtask(), "output", [], None)
        assert result.passed
        assert result.tier == 2
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_fails_when_model_says_fail(self):
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.complete = AsyncMock(return_value=ModelResponse(
            text=json.dumps({
                "passed": False,
                "issues": ["Missing error handling"],
                "confidence": 0.8,
                "suggestion": "Add try/except blocks",
            }),
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        ))
        router.select = MagicMock(return_value=model)

        prompts = MagicMock(spec=PromptAssembler)
        prompts.build_verifier_prompt = MagicMock(return_value="Verify this")

        v = LLMVerifier(router, prompts, ResponseValidator())
        result = await v.verify(_make_subtask(), "output", [], None)
        assert not result.passed
        assert result.feedback == "Add try/except blocks"

    @pytest.mark.asyncio
    async def test_fails_safe_on_no_verifier(self):
        router = MagicMock(spec=ModelRouter)
        router.select = MagicMock(side_effect=Exception("No model"))

        prompts = MagicMock(spec=PromptAssembler)
        v = LLMVerifier(router, prompts, ResponseValidator())
        result = await v.verify(_make_subtask(), "output", [], None)
        assert result.passed
        assert result.tier == 0
        assert result.confidence == 0.5
        assert "Verification skipped" in result.feedback


# --- VotingVerifier ---


class TestVotingVerifier:
    @pytest.mark.asyncio
    async def test_passes_on_majority(self):
        llm = AsyncMock(spec=LLMVerifier)
        llm.verify = AsyncMock(side_effect=[
            VerificationResult(tier=2, passed=True),
            VerificationResult(tier=2, passed=True),
            VerificationResult(tier=2, passed=False),
        ])

        v = VotingVerifier(llm, vote_count=3)
        result = await v.verify(_make_subtask(), "output", [], None)
        assert result.passed
        assert result.tier == 3
        assert result.confidence == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_fails_on_minority(self):
        llm = AsyncMock(spec=LLMVerifier)
        llm.verify = AsyncMock(side_effect=[
            VerificationResult(tier=2, passed=False),
            VerificationResult(tier=2, passed=False),
            VerificationResult(tier=2, passed=True),
        ])

        v = VotingVerifier(llm, vote_count=3)
        result = await v.verify(_make_subtask(), "output", [], None)
        assert not result.passed
        assert result.feedback is not None


# --- VerificationGates ---


class TestVerificationGates:
    @pytest.mark.asyncio
    async def test_tier1_only(self):
        config = VerificationConfig(tier1_enabled=True, tier2_enabled=False)
        router = MagicMock(spec=ModelRouter)
        prompts = MagicMock(spec=PromptAssembler)

        gates = VerificationGates(router, prompts, config)
        result = await gates.verify(_make_subtask(), "output", [], None, tier=1)
        assert result.passed

    @pytest.mark.asyncio
    async def test_tier1_failure_blocks_tier2(self, tmp_path):
        config = VerificationConfig(tier1_enabled=True, tier2_enabled=True)
        router = MagicMock(spec=ModelRouter)
        prompts = MagicMock(spec=PromptAssembler)

        gates = VerificationGates(router, prompts, config)

        # Create a failed tool call
        tc = MockToolCallRecord(
            tool="write_file",
            args={"path": "test.py"},
            result=ToolResult.fail("Error writing"),
        )
        result = await gates.verify(_make_subtask(), "output", [tc], None, tier=2)
        assert not result.passed
        assert result.tier == 1  # Failed at tier 1, never reached tier 2


# --- DeterministicVerifier with process definitions ---


def _make_process(
    deliverables: dict[str, list[str]] | None = None,
    regex_rules: list | None = None,
):
    """Build a mock ProcessDefinition for verification tests."""
    from loom.processes.schema import (
        PhaseTemplate,
        ProcessDefinition,
        VerificationRule,
    )

    phases = []
    if deliverables:
        for phase_id, files in deliverables.items():
            phases.append(PhaseTemplate(
                id=phase_id,
                description=f"Phase {phase_id}",
                deliverables=[f"{f} — desc" for f in files],
            ))

    rules = []
    if regex_rules:
        for r in regex_rules:
            rules.append(VerificationRule(**r))

    return ProcessDefinition(
        name="test-proc",
        phases=phases,
        verification_rules=rules,
    )


class TestDeterministicVerifierDeliverables:
    """Tests for phase-scoped deliverable checking."""

    @pytest.mark.asyncio
    async def test_deliverables_found_in_workspace(self, tmp_path):
        """Only the active phase deliverables should be enforced."""
        (tmp_path / "report.md").write_text("Report content")

        process = _make_process(deliverables={
            "research": ["report.md"],
            "analysis": ["data.csv"],
        })
        v = DeterministicVerifier(process=process)
        result = await v.verify(
            _make_subtask(subtask_id="research"),
            "output",
            [],
            tmp_path,
        )
        assert result.passed
        check_names = {c.name for c in result.checks}
        assert "deliverable_report.md" in check_names
        assert "deliverable_data.csv" not in check_names

    @pytest.mark.asyncio
    async def test_deliverables_found_in_tool_calls(self, tmp_path):
        """Deliverables reported in tool call files_changed should pass."""
        process = _make_process(deliverables={
            "phase1": ["output.txt"],
        })
        tc = MockToolCallRecord(
            tool="write_file",
            args={"path": "output.txt"},
            result=ToolResult.ok("ok", files_changed=["output.txt"]),
        )
        v = DeterministicVerifier(process=process)
        result = await v.verify(
            _make_subtask(subtask_id="phase1"),
            "output",
            [tc],
            tmp_path,
        )
        assert result.passed

    @pytest.mark.asyncio
    async def test_deliverables_missing_fails(self, tmp_path):
        """Missing deliverables should cause verification to fail."""
        process = _make_process(deliverables={
            "phase1": ["missing.txt"],
        })
        v = DeterministicVerifier(process=process)
        result = await v.verify(
            _make_subtask(subtask_id="phase1"),
            "output",
            [],
            tmp_path,
        )
        assert not result.passed
        failed = [c for c in result.checks if not c.passed]
        assert any("missing.txt" in (c.detail or "") for c in failed)

    @pytest.mark.asyncio
    async def test_deliverables_do_not_flatten_across_phases(self, tmp_path):
        """Non-active phase deliverables should not fail the current phase."""
        (tmp_path / "file_a.md").write_text("content")
        process = _make_process(deliverables={
            "phase_a": ["file_a.md"],
            "phase_b": ["file_b.md"],
        })
        v = DeterministicVerifier(process=process)
        result = await v.verify(
            _make_subtask(subtask_id="phase_a"),
            "output", [], tmp_path,
        )
        assert result.passed
        check_names = {c.name: c.passed for c in result.checks}
        assert check_names.get("deliverable_file_a.md") is True
        assert "deliverable_file_b.md" not in check_names

    @pytest.mark.asyncio
    async def test_unmatched_subtask_id_skips_multi_phase_deliverables(self, tmp_path):
        """Unmapped subtask IDs should not enforce unrelated phase outputs."""
        process = _make_process(deliverables={
            "phase_a": ["file_a.md"],
            "phase_b": ["file_b.md"],
        })
        v = DeterministicVerifier(process=process)
        result = await v.verify(
            _make_subtask(subtask_id="planner-generated-id"),
            "output",
            [],
            tmp_path,
        )
        assert result.passed
        assert not any(c.name.startswith("deliverable_") for c in result.checks)


class TestDeterministicVerifierRegexRules:
    """Tests for regex rule severity handling."""

    @pytest.mark.asyncio
    async def test_error_severity_rule_fails_on_match(self):
        """Error-severity regex rules should fail verification when matched."""
        process = _make_process(regex_rules=[{
            "name": "no-tbd",
            "description": "No TBD markers allowed",
            "check": "TBD",
            "severity": "error",
            "type": "regex",
            "target": "output",
        }])
        v = DeterministicVerifier(process=process)
        result = await v.verify(
            _make_subtask(), "This has TBD in it", [], None,
        )
        assert not result.passed

    @pytest.mark.asyncio
    async def test_warning_severity_rule_passes_on_match(self):
        """Warning-severity regex rules should NOT fail verification when matched."""
        process = _make_process(regex_rules=[{
            "name": "no-todo",
            "description": "TODO comments found",
            "check": "TODO",
            "severity": "warning",
            "type": "regex",
            "target": "output",
        }])
        v = DeterministicVerifier(process=process)
        result = await v.verify(
            _make_subtask(), "There is a TODO here", [], None,
        )
        # Warning rules record the match but don't fail
        assert result.passed
        assert any("no-todo" in c.name for c in result.checks)
        # Detail should still record that it matched
        matched_check = [c for c in result.checks if "no-todo" in c.name][0]
        assert matched_check.detail is not None

    @pytest.mark.asyncio
    async def test_error_severity_rule_passes_on_no_match(self):
        """Error-severity regex rules should pass when pattern is NOT found."""
        process = _make_process(regex_rules=[{
            "name": "no-tbd",
            "description": "No TBD markers allowed",
            "check": "TBD",
            "severity": "error",
            "type": "regex",
            "target": "output",
        }])
        v = DeterministicVerifier(process=process)
        result = await v.verify(
            _make_subtask(), "Clean output with no markers", [], None,
        )
        assert result.passed
