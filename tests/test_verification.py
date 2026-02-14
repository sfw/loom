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
        assert not result.passed
        assert result.confidence == 0.0
        assert "No verifier model configured" in result.feedback


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
