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
from loom.events.bus import EventBus
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
    depends_on: list[str] | None = None,
    is_synthesis: bool = False,
) -> Subtask:
    return Subtask(
        id=subtask_id,
        description=description,
        acceptance_criteria=acceptance_criteria,
        depends_on=list(depends_on or []),
        is_synthesis=bool(is_synthesis),
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
        assert r.severity_class == "semantic"

    def test_verification_result_infers_severity_class_from_reason_code(self):
        r = VerificationResult(
            tier=2,
            passed=False,
            outcome="fail",
            reason_code="parse_inconclusive",
        )
        assert r.severity_class == "inconclusive"

    def test_verification_result_respects_explicit_severity_class(self):
        r = VerificationResult(
            tier=2,
            passed=False,
            outcome="fail",
            reason_code="llm_semantic_failed",
            severity_class="infra",
        )
        assert r.severity_class == "infra"


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
    async def test_safety_integrity_policy_downgrades_non_safety_tool_failures(self):
        process = _make_process(static_checks={
            "tool_success_policy": "safety_integrity_only",
        })
        v = DeterministicVerifier(process=process)
        tc = MockToolCallRecord(
            tool="edit_file",
            args={"path": "pricing-access-grid.csv"},
            result=ToolResult.fail(
                "edit[0]: old_str appears 2 times in pricing-access-grid.csv",
            ),
        )
        result = await v.verify(_make_subtask(), "output", [tc], None)
        assert result.passed
        assert any(
            c.name == "tool_edit_file_advisory" and c.passed
            for c in result.checks
        )

    @pytest.mark.asyncio
    async def test_safety_integrity_policy_keeps_safety_failures_hard(self):
        process = _make_process(static_checks={
            "tool_success_policy": "safety_integrity_only",
        })
        v = DeterministicVerifier(process=process)
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
    async def test_web_tool_404_failure_is_advisory(self):
        v = DeterministicVerifier()
        tc = MockToolCallRecord(
            tool="web_fetch",
            args={"url": "https://example.com/missing"},
            result=ToolResult.fail("HTTP 404: https://example.com/missing"),
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
    async def test_web_tool_response_too_large_is_advisory(self):
        v = DeterministicVerifier()
        tc = MockToolCallRecord(
            tool="web_fetch",
            args={"url": "https://example.com/huge"},
            result=ToolResult.fail(
                "Response too large (11260600 bytes). Max: 2097152.",
            ),
        )
        result = await v.verify(_make_subtask(), "output", [tc], None)
        assert result.passed
        assert any(
            c.name == "tool_web_fetch_advisory" and c.passed
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
    class _FakeCompactor:
        async def compact(self, text: str, *, max_chars: int, label: str = "") -> str:
            value = str(text or "").strip()
            if len(value) <= max_chars:
                return value
            words = value.split()
            if len(words) <= 1:
                return f"[compacted {len(value)} chars]"
            compacted = ""
            for word in words:
                candidate = f"{compacted} {word}".strip()
                if compacted and len(candidate) > max_chars:
                    break
                compacted = candidate
            return compacted or value

    class _NoopCompactor:
        async def compact(self, text: str, *, max_chars: int, label: str = "") -> str:
            return str(text or "")

    @pytest.mark.asyncio
    async def test_passes_when_model_says_pass(self):
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(return_value=ModelResponse(
            text=json.dumps({"passed": True, "issues": [], "confidence": 0.9}),
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        ))
        router.select = MagicMock(return_value=model)

        prompts = MagicMock(spec=PromptAssembler)
        prompts.build_verifier_prompt = MagicMock(return_value="Verify this")

        v = LLMVerifier(router, prompts, ResponseValidator())
        v._compactor = self._FakeCompactor()
        result = await v.verify(_make_subtask(), "output", [], None)
        assert result.passed
        assert result.tier == 2
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_fails_when_model_says_fail(self):
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
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
        v._compactor = self._FakeCompactor()
        result = await v.verify(_make_subtask(), "output", [], None)
        assert not result.passed
        assert result.feedback == "Add try/except blocks"

    @pytest.mark.asyncio
    async def test_verifier_preserves_structured_severity_class(self):
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(return_value=ModelResponse(
            text=json.dumps({
                "passed": False,
                "issues": ["Verifier backend unavailable"],
                "confidence": 0.2,
                "outcome": "fail",
                "reason_code": "infra_verifier_error",
                "severity_class": "infra",
            }),
            usage=TokenUsage(input_tokens=20, output_tokens=18, total_tokens=38),
        ))
        router.select = MagicMock(return_value=model)
        prompts = MagicMock(spec=PromptAssembler)
        prompts.build_verifier_prompt = MagicMock(return_value="Verify this")

        verifier = LLMVerifier(router, prompts, ResponseValidator())
        verifier._compactor = self._FakeCompactor()
        result = await verifier.verify(_make_subtask(), "output", [], None)
        assert not result.passed
        assert result.reason_code == "infra_verifier_error"
        assert result.severity_class == "infra"

    @pytest.mark.asyncio
    async def test_verifier_parses_json_wrapped_in_text(self):
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(return_value=ModelResponse(
            text=(
                "Assessment complete.\n"
                '{"passed": true, "issues": [], "confidence": 0.74, '
                '"suggestion": "Looks good"}\n'
                "End."
            ),
            usage=TokenUsage(input_tokens=40, output_tokens=30, total_tokens=70),
        ))
        router.select = MagicMock(return_value=model)

        prompts = MagicMock(spec=PromptAssembler)
        prompts.build_verifier_prompt = MagicMock(return_value="Verify this")

        v = LLMVerifier(router, prompts, ResponseValidator())
        v._compactor = self._FakeCompactor()
        result = await v.verify(_make_subtask(), "output", [], None)
        assert result.passed
        assert result.confidence == 0.74
        assert result.feedback == "Looks good"

    @pytest.mark.asyncio
    async def test_verifier_uses_feedback_field(self):
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(return_value=ModelResponse(
            text=json.dumps({
                "passed": False,
                "issues": ["Needs citations"],
                "confidence": 0.6,
                "feedback": "Add source links for each claim",
            }),
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        ))
        router.select = MagicMock(return_value=model)

        prompts = MagicMock(spec=PromptAssembler)
        prompts.build_verifier_prompt = MagicMock(return_value="Verify this")

        v = LLMVerifier(router, prompts, ResponseValidator())
        v._compactor = self._FakeCompactor()
        result = await v.verify(_make_subtask(), "output", [], None)
        assert not result.passed
        assert result.feedback == "Add source links for each claim"

    @pytest.mark.asyncio
    async def test_verifier_parses_yaml_style_assessment(self):
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(return_value=ModelResponse(
            text=(
                "passed: false\n"
                "confidence: 62%\n"
                "feedback: Missing direct citations\n"
                "issues:\n"
                "  - No source links for pricing claims\n"
                "  - Regulatory claims are unverified\n"
            ),
            usage=TokenUsage(input_tokens=60, output_tokens=40, total_tokens=100),
        ))
        router.select = MagicMock(return_value=model)

        prompts = MagicMock(spec=PromptAssembler)
        prompts.build_verifier_prompt = MagicMock(return_value="Verify this")

        v = LLMVerifier(router, prompts, ResponseValidator())
        v._compactor = self._FakeCompactor()
        result = await v.verify(_make_subtask(), "output", [], None)
        assert not result.passed
        assert result.confidence == pytest.approx(0.62, rel=1e-3)
        assert "Missing direct citations" in (result.feedback or "")
        assert result.checks
        assert "No source links for pricing claims" in (result.checks[0].detail or "")

    @pytest.mark.asyncio
    async def test_verifier_parses_plain_text_verdict(self):
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(return_value=ModelResponse(
            text=(
                "Assessment: failed.\n"
                "Confidence: 0.71\n"
                "Reason: acceptance criteria were not met due to missing output file."
            ),
            usage=TokenUsage(input_tokens=30, output_tokens=20, total_tokens=50),
        ))
        router.select = MagicMock(return_value=model)

        prompts = MagicMock(spec=PromptAssembler)
        prompts.build_verifier_prompt = MagicMock(return_value="Verify this")

        v = LLMVerifier(router, prompts, ResponseValidator())
        v._compactor = self._FakeCompactor()
        result = await v.verify(_make_subtask(), "output", [], None)
        assert not result.passed
        assert result.confidence == pytest.approx(0.71, rel=1e-3)
        assert "acceptance criteria were not met" in (result.feedback or "").lower()

    @pytest.mark.asyncio
    async def test_verifier_inconclusive_when_no_structured_signal(self):
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(return_value=ModelResponse(
            text="I cannot determine the result from the available information.",
            usage=TokenUsage(input_tokens=30, output_tokens=20, total_tokens=50),
        ))
        router.select = MagicMock(return_value=model)

        prompts = MagicMock(spec=PromptAssembler)
        prompts.build_verifier_prompt = MagicMock(return_value="Verify this")

        v = LLMVerifier(router, prompts, ResponseValidator())
        v._compactor = self._FakeCompactor()
        result = await v.verify(_make_subtask(), "output", [], None)
        assert not result.passed
        assert result.feedback == "Verification inconclusive: could not parse verifier output."

    @pytest.mark.asyncio
    async def test_verifier_compact_text_keeps_oversize_compactor_output(self):
        router = MagicMock(spec=ModelRouter)
        prompts = MagicMock(spec=PromptAssembler)

        v = LLMVerifier(router, prompts, ResponseValidator())
        v._compactor = self._NoopCompactor()

        value = "B" * 8000
        compacted = await v._compact_text(
            value,
            max_chars=180,
            label="verifier oversize guard",
        )

        assert compacted == value

    @pytest.mark.asyncio
    async def test_fails_safe_on_no_verifier(self):
        router = MagicMock(spec=ModelRouter)
        router.select = MagicMock(side_effect=Exception("No model"))

        prompts = MagicMock(spec=PromptAssembler)
        v = LLMVerifier(router, prompts, ResponseValidator())
        v._compactor = self._FakeCompactor()
        result = await v.verify(_make_subtask(), "output", [], None)
        assert result.passed
        assert result.tier == 0
        assert result.confidence == 0.5
        assert "Verification skipped" in result.feedback

    @pytest.mark.asyncio
    async def test_compacts_large_tool_args_in_verifier_prompt(self):
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(return_value=ModelResponse(
            text=json.dumps({"passed": True, "issues": [], "confidence": 0.8}),
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        ))
        router.select = MagicMock(return_value=model)

        prompts = MagicMock(spec=PromptAssembler)
        prompts.build_verifier_prompt = MagicMock(
            side_effect=lambda **kw: kw["tool_calls_formatted"]
        )

        large_markdown = "# Section\n" + ("x" * 20_000)
        tool_calls = [
            MockToolCallRecord(
                tool="document_write",
                args={"path": "report.md", "content": large_markdown},
                result=ToolResult.ok("wrote file", files_changed=["report.md"]),
            )
        ]

        v = LLMVerifier(router, prompts, ResponseValidator())
        v._compactor = self._FakeCompactor()
        result = await v.verify(_make_subtask(), "output", tool_calls, None)

        assert result.passed
        sent_prompt = model.complete.await_args.args[0][0]["content"]
        assert "x" * 1000 not in sent_prompt
        assert len(sent_prompt) < 2000

    @pytest.mark.asyncio
    async def test_retries_verifier_with_compact_prompt_after_exception(self):
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(side_effect=[
            RuntimeError("token limit exceeded"),
            ModelResponse(
                text=json.dumps({"passed": True, "issues": [], "confidence": 0.7}),
                usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
            ),
        ])
        router.select = MagicMock(return_value=model)

        prompts = MagicMock(spec=PromptAssembler)
        prompts.build_verifier_prompt = MagicMock(
            side_effect=lambda **kw: f"VERIFY\n{kw['tool_calls_formatted']}"
        )

        large_content = "y" * 15_000
        tool_calls = [
            MockToolCallRecord(
                tool="document_write",
                args={"path": "brief.md", "content": large_content},
                result=ToolResult.ok("done"),
            )
            for _ in range(30)
        ]

        v = LLMVerifier(router, prompts, ResponseValidator())
        v._compactor = self._FakeCompactor()
        result = await v.verify(_make_subtask(), "output", tool_calls, None)

        assert result.passed
        assert model.complete.await_count == 2
        first_prompt = model.complete.await_args_list[0].args[0][0]["content"]
        second_prompt = model.complete.await_args_list[1].args[0][0]["content"]
        assert len(second_prompt) <= len(first_prompt)

    @pytest.mark.asyncio
    async def test_verifier_exception_feedback_includes_error_detail(self):
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(side_effect=RuntimeError("token budget exceeded"))
        router.select = MagicMock(return_value=model)

        prompts = MagicMock(spec=PromptAssembler)
        prompts.build_verifier_prompt = MagicMock(return_value="Verify this")

        v = LLMVerifier(router, prompts, ResponseValidator())
        v._compactor = self._FakeCompactor()
        result = await v.verify(_make_subtask(), "output", [], None)
        assert not result.passed
        assert "token budget exceeded" in (result.feedback or "")

    @pytest.mark.asyncio
    async def test_llm_tool_call_format_marks_web_404_as_advisory(self):
        router = MagicMock(spec=ModelRouter)
        prompts = MagicMock(spec=PromptAssembler)
        v = LLMVerifier(router, prompts, ResponseValidator())
        v._compactor = self._FakeCompactor()

        tool_calls = [
            MockToolCallRecord(
                tool="web_fetch",
                args={"url": "https://example.com/missing"},
                result=ToolResult.fail("HTTP 404: https://example.com/missing"),
            ),
        ]
        formatted = await v._format_tool_calls_for_prompt(tool_calls)

        assert "-> ADVISORY_FAILURE:" in formatted

    @pytest.mark.asyncio
    async def test_llm_tool_call_format_includes_read_output_excerpt(self):
        router = MagicMock(spec=ModelRouter)
        prompts = MagicMock(spec=PromptAssembler)
        v = LLMVerifier(router, prompts, ResponseValidator())
        v._compactor = self._FakeCompactor()

        tool_calls = [
            MockToolCallRecord(
                tool="read_file",
                args={"path": "slogan-shortlist.md"},
                result=ToolResult.ok("Top line: Your dentist sees more than cavities."),
            ),
        ]
        formatted = await v._format_tool_calls_for_prompt(tool_calls)

        assert "output_excerpt:" in formatted
        assert "Top line: Your dentist sees more than cavities." in formatted

    @pytest.mark.asyncio
    async def test_llm_tool_call_format_includes_document_write_excerpt(self):
        router = MagicMock(spec=ModelRouter)
        prompts = MagicMock(spec=PromptAssembler)
        v = LLMVerifier(router, prompts, ResponseValidator())
        v._compactor = self._FakeCompactor()

        tool_calls = [
            MockToolCallRecord(
                tool="document_write",
                args={
                    "path": "slogan-shortlist.md",
                    "title": "Shortlist",
                    "sections": [
                        {
                            "heading": "Winner",
                            "body": "Your dentist sees more than cavities.",
                        },
                    ],
                },
                result=ToolResult.ok(
                    "Created slogan-shortlist.md (~120 words)",
                    files_changed=["slogan-shortlist.md"],
                ),
            ),
        ]
        formatted = await v._format_tool_calls_for_prompt(tool_calls)

        assert "output_excerpt:" in formatted
        assert "Your dentist sees more than cavities." in formatted

    @pytest.mark.asyncio
    async def test_verifier_caps_result_summary_before_prompt(self):
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(return_value=ModelResponse(
            text=json.dumps({"passed": True, "issues": [], "confidence": 0.8}),
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        ))
        router.select = MagicMock(return_value=model)

        prompts = MagicMock(spec=PromptAssembler)
        prompts.build_verifier_prompt = MagicMock(
            side_effect=lambda **kw: kw["result_summary"]
        )

        long_summary = "A" * 12_000
        v = LLMVerifier(router, prompts, ResponseValidator())
        v._compactor = self._FakeCompactor()
        result = await v.verify(_make_subtask(), long_summary, [], None)

        assert result.passed
        sent = model.complete.await_args.args[0][0]["content"]
        assert len(sent) < len(long_summary)

    @pytest.mark.asyncio
    async def test_verifier_appends_advisory_evidence_context(self):
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(return_value=ModelResponse(
            text=json.dumps({"passed": True, "issues": [], "confidence": 0.8}),
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        ))
        router.select = MagicMock(return_value=model)

        prompts = MagicMock(spec=PromptAssembler)
        prompts.build_verifier_prompt = MagicMock(return_value="Verify this")

        v = LLMVerifier(router, prompts, ResponseValidator())
        v._compactor = self._FakeCompactor()
        result = await v.verify(
            _make_subtask(subtask_id="gather-evidence"),
            "output",
            tool_calls=[],
            workspace=None,
            evidence_tool_calls=[
                MockToolCallRecord(
                    tool="web_fetch",
                    args={"url": "https://example.com/texas"},
                    result=ToolResult.ok("Texas source details"),
                ),
            ],
            evidence_records=[{
                "evidence_id": "EV-EXISTING",
                "market": "Alberta",
                "dimension": "economic",
                "source_url": "https://example.com/alberta",
            }],
        )

        assert result.passed
        sent = model.complete.await_args.args[0][0]["content"]
        assert "EVIDENCE CONTEXT SNAPSHOT (advisory, non-binding):" in sent
        assert "observed_tools:" in sent
        assert "sample_records:" in sent

    @pytest.mark.asyncio
    async def test_verifier_extracts_file_based_evidence_records(self):
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(return_value=ModelResponse(
            text=json.dumps({"passed": True, "issues": [], "confidence": 0.8}),
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        ))
        router.select = MagicMock(return_value=model)

        prompts = MagicMock(spec=PromptAssembler)
        prompts.build_verifier_prompt = MagicMock(return_value="Verify this")

        v = LLMVerifier(router, prompts, ResponseValidator())
        v._compactor = self._FakeCompactor()
        result = await v.verify(
            _make_subtask(subtask_id="shortlist-and-polish"),
            "output",
            tool_calls=[],
            workspace=None,
            evidence_tool_calls=[
                MockToolCallRecord(
                    tool="read_file",
                    args={"path": "slogan-shortlist.md"},
                    result=ToolResult.ok(
                        "Top lines:\n- Your dentist sees more than cavities."
                    ),
                ),
                MockToolCallRecord(
                    tool="document_write",
                    args={
                        "path": "shortlist-scorecard.md",
                        "content": "Rank 1: Your dentist sees more than cavities.",
                    },
                    result=ToolResult.ok("Created shortlist-scorecard.md"),
                ),
            ],
        )

        assert result.passed
        sent = model.complete.await_args.args[0][0]["content"]
        assert "EVIDENCE CONTEXT SNAPSHOT (advisory, non-binding):" in sent
        assert "observed_tools:" in sent
        assert "read_file" in sent
        assert "document_write" in sent

    @pytest.mark.asyncio
    async def test_verifier_appends_changed_artifact_content_snapshot(self, tmp_path):
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(return_value=ModelResponse(
            text=json.dumps({"passed": True, "issues": [], "confidence": 0.8}),
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        ))
        router.select = MagicMock(return_value=model)

        prompts = MagicMock(spec=PromptAssembler)
        prompts.build_verifier_prompt = MagicMock(return_value="Verify this")

        (tmp_path / "slogan-shortlist.md").write_text(
            "## Winners\n- Your dentist sees more than cavities.\n"
        )
        tool_calls = [
            MockToolCallRecord(
                tool="write_file",
                args={"path": "slogan-shortlist.md"},
                result=ToolResult.ok(
                    "Wrote slogan-shortlist.md",
                    files_changed=["slogan-shortlist.md"],
                ),
            ),
        ]

        v = LLMVerifier(router, prompts, ResponseValidator())
        v._compactor = self._FakeCompactor()
        result = await v.verify(
            _make_subtask(subtask_id="shortlist-and-polish"),
            "output",
            tool_calls=tool_calls,
            workspace=tmp_path,
        )

        assert result.passed
        sent = model.complete.await_args.args[0][0]["content"]
        assert "ARTIFACT CONTENT SNAPSHOT (for semantic review):" in sent
        assert "Your dentist sees more than cavities." in sent

    @pytest.mark.asyncio
    async def test_verifier_injects_only_phase_scoped_rules(self):
        from loom.processes.schema import ProcessDefinition, VerificationRule

        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(return_value=ModelResponse(
            text=json.dumps({"passed": True, "issues": [], "confidence": 0.8}),
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        ))
        router.select = MagicMock(return_value=model)

        process = ProcessDefinition(
            name="phase-scoped",
            verification_rules=[
                VerificationRule(
                    name="rule-a",
                    description="A only",
                    check="Check A",
                    type="llm",
                    applies_to_phases=["phase-a"],
                ),
                VerificationRule(
                    name="rule-b",
                    description="B only",
                    check="Check B",
                    type="llm",
                    applies_to_phases=["phase-b"],
                ),
            ],
        )
        prompts = PromptAssembler(process=process)
        v = LLMVerifier(
            router,
            prompts,
            ResponseValidator(),
            verification_config=VerificationConfig(phase_scope_default="current_phase"),
        )
        v._compactor = self._FakeCompactor()
        result = await v.verify(
            _make_subtask(subtask_id="phase-a"),
            "output",
            [],
            None,
            task_id="task-1",
        )

        assert result.passed
        sent = model.complete.await_args.args[0][0]["content"]
        assert "rule-a" in sent
        assert "rule-b" not in sent

    @pytest.mark.asyncio
    async def test_verifier_preserves_model_metadata_without_static_policy_override(self):
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(return_value=ModelResponse(
            text=json.dumps({
                "passed": True,
                "confidence": 0.82,
                "issues": [],
                "metadata": {
                    "primary_unconfirmed_count": 1,
                    "remediation_mode": "confirm_or_prune",
                },
            }),
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        ))
        router.select = MagicMock(return_value=model)
        prompts = MagicMock(spec=PromptAssembler)
        prompts.process = None
        prompts.build_verifier_prompt = MagicMock(return_value="Verify this")
        v = LLMVerifier(router, prompts, ResponseValidator())
        v._compactor = self._FakeCompactor()

        result = await v.verify(_make_subtask(), "output", [], None)
        assert result.passed
        assert result.outcome == "pass"
        assert result.metadata.get("primary_unconfirmed_count") == 1
        assert result.metadata.get("remediation_mode") == "confirm_or_prune"

    @pytest.mark.asyncio
    async def test_verifier_respects_model_declared_failure_and_reason_code(self):
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(return_value=ModelResponse(
            text=json.dumps({
                "passed": False,
                "outcome": "fail",
                "reason_code": "policy_remediation_required",
                "severity_class": "semantic",
                "confidence": 0.45,
                "feedback": "Insufficient evidence linkage",
                "issues": ["claim-to-evidence map missing"],
                "metadata": {"remediation_mode": "targeted_remediation"},
            }),
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        ))
        router.select = MagicMock(return_value=model)
        prompts = MagicMock(spec=PromptAssembler)
        prompts.process = None
        prompts.build_verifier_prompt = MagicMock(return_value="Verify this")
        v = LLMVerifier(router, prompts, ResponseValidator())
        v._compactor = self._FakeCompactor()

        result = await v.verify(_make_subtask(), "output", [], None)
        assert not result.passed
        assert result.reason_code == "policy_remediation_required"
        assert result.metadata.get("remediation_mode") == "targeted_remediation"

    @pytest.mark.asyncio
    async def test_verifier_normalizes_uncertainty_metadata_fields(self):
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(return_value=ModelResponse(
            text=json.dumps({
                "passed": False,
                "outcome": "fail",
                "reason_code": "policy_remediation_required",
                "severity_class": "semantic",
                "confidence": 0.45,
                "feedback": "Supporting claims still need evidence links.",
                "issues": ["missing evidence linkage"],
                "metadata": {
                    "remediation_required": "true",
                    "remediation_mode": "Queue_Follow_Up",
                    "missing_targets": "T1, T2",
                    "unverified_claim_count": "3",
                    "verified_claim_count": "7",
                    "supporting_ratio": "55%",
                },
            }),
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        ))
        router.select = MagicMock(return_value=model)
        prompts = MagicMock(spec=PromptAssembler)
        prompts.process = None
        prompts.build_verifier_prompt = MagicMock(return_value="Verify this")
        verifier = LLMVerifier(router, prompts, ResponseValidator())
        verifier._compactor = self._FakeCompactor()

        result = await verifier.verify(_make_subtask(), "output", [], None)
        assert not result.passed
        assert result.metadata.get("remediation_required") is True
        assert result.metadata.get("remediation_mode") == "queue_follow_up"
        assert result.metadata.get("missing_targets") == ["T1", "T2"]
        assert result.metadata.get("unverified_claim_count") == 3
        assert result.metadata.get("verified_claim_count") == 7
        assert result.metadata.get("supporting_ratio") == pytest.approx(0.55)


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

    @pytest.mark.asyncio
    async def test_tier2_parse_inconclusive_falls_back_to_tier1_warning(self):
        config = VerificationConfig(tier1_enabled=True, tier2_enabled=True)
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(return_value=ModelResponse(
            text="I cannot determine the result from the available information.",
            usage=TokenUsage(input_tokens=20, output_tokens=15, total_tokens=35),
        ))
        router.select = MagicMock(return_value=model)
        prompts = PromptAssembler()

        gates = VerificationGates(router, prompts, config)
        result = await gates.verify(
            _make_subtask(subtask_id="phase-a"),
            "subtask summary",
            [],
            None,
            tier=2,
            task_id="task-parse-fallback",
        )

        assert result.passed
        assert result.outcome == "pass_with_warnings"
        assert result.reason_code == "infra_verifier_error"
        assert result.severity_class == "infra"
        assert result.metadata.get("fallback") == "tier1_due_to_tier2_parse_inconclusive"

    @pytest.mark.asyncio
    async def test_placeholder_claim_contradiction_returns_inconclusive_retry_path(self, tmp_path):
        process = _make_process(deliverables={"phase-a": ["report.md"]})
        (tmp_path / "report.md").write_text("Final output with no unresolved placeholders.")
        config = VerificationConfig(
            tier1_enabled=True,
            tier2_enabled=True,
            contradiction_guard_enabled=True,
        )
        router = MagicMock(spec=ModelRouter)
        prompts = PromptAssembler(process=process)
        event_bus = EventBus()
        events = []
        event_bus.subscribe_all(lambda event: events.append(event))

        gates = VerificationGates(
            router,
            prompts,
            config,
            process=process,
            event_bus=event_bus,
        )
        gates._tier2.verify = AsyncMock(return_value=VerificationResult(
            tier=2,
            passed=False,
            outcome="fail",
            reason_code="incomplete_deliverable_placeholder",
            severity_class="semantic",
            feedback="Deliverable contains [MISSING] placeholder markers.",
            metadata={"issues": ["[MISSING] placeholder present"]},
        ))

        tool_call = MockToolCallRecord(
            tool="write_file",
            args={"path": "report.md"},
            result=ToolResult.ok("ok", files_changed=["report.md"]),
        )
        result = await gates.verify(
            _make_subtask(subtask_id="phase-a"),
            "subtask summary",
            [tool_call],
            tmp_path,
            tier=2,
            task_id="task-contradiction",
        )

        assert not result.passed
        assert result.reason_code == "parse_inconclusive"
        assert result.severity_class == "inconclusive"
        assert result.metadata.get("contradiction_detected") is True
        scan = result.metadata.get("deterministic_placeholder_scan", {})
        assert scan.get("scanned_file_count") == 1
        assert scan.get("matched_file_count") == 0
        assert scan.get("coverage_sufficient") is True
        assert scan.get("scan_mode") in {"targeted_only", "targeted_plus_fallback"}
        event_types = [event.event_type for event in events]
        assert "verification_contradiction_detected" in event_types
        contradiction_event = next(
            event for event in events
            if event.event_type == "verification_contradiction_detected"
        )
        assert contradiction_event.data.get("coverage_sufficient") is True
        assert contradiction_event.data.get("contradiction_downgrade_count") == 1
        assert contradiction_event.data.get("contradiction_detected_no_downgrade_count") == 0

    @pytest.mark.asyncio
    async def test_marker_in_changed_file_blocks_downgrade(self, tmp_path):
        process = _make_process(deliverables={"phase-a": ["report.md"]})
        (tmp_path / "report.md").write_text("Final deliverable.")
        (tmp_path / "scratch-notes.md").write_text("TODO: unresolved bullet.")
        config = VerificationConfig(
            tier1_enabled=True,
            tier2_enabled=True,
            contradiction_guard_enabled=True,
        )
        router = MagicMock(spec=ModelRouter)
        prompts = PromptAssembler(process=process)
        gates = VerificationGates(router, prompts, config, process=process)
        gates._tier2.verify = AsyncMock(return_value=VerificationResult(
            tier=2,
            passed=False,
            outcome="fail",
            reason_code="incomplete_deliverable_placeholder",
            severity_class="semantic",
            feedback="Deliverable still contains TODO markers.",
            metadata={"issues": ["TODO placeholder found"]},
        ))

        tool_calls = [
            MockToolCallRecord(
                tool="write_file",
                args={"path": "report.md"},
                result=ToolResult.ok("ok", files_changed=["report.md"]),
            ),
            MockToolCallRecord(
                tool="write_file",
                args={"path": "scratch-notes.md"},
                result=ToolResult.ok("ok", files_changed=["scratch-notes.md"]),
            ),
        ]
        result = await gates.verify(
            _make_subtask(subtask_id="phase-a"),
            "summary",
            tool_calls,
            tmp_path,
            tier=2,
            task_id="task-nondeliverable-marker",
        )

        assert not result.passed
        assert result.reason_code == "incomplete_deliverable_placeholder"
        assert result.metadata.get("contradiction_detected") is False
        scan = result.metadata.get("deterministic_placeholder_scan", {})
        assert "scratch-notes.md" in scan.get("matched_files", [])
        assert scan.get("matched_file_count") == 1

    @pytest.mark.asyncio
    async def test_placeholder_marker_in_prior_attempt_candidate_blocks_downgrade(self, tmp_path):
        process = _make_process(deliverables={"phase-a": ["report.md"]})
        (tmp_path / "report.md").write_text("Final deliverable.")
        (tmp_path / "prior-draft.md").write_text("PLACEHOLDER remains in prior attempt.")
        config = VerificationConfig(
            tier1_enabled=True,
            tier2_enabled=True,
            contradiction_guard_enabled=True,
        )
        router = MagicMock(spec=ModelRouter)
        prompts = PromptAssembler(process=process)
        gates = VerificationGates(router, prompts, config, process=process)
        gates._tier2.verify = AsyncMock(return_value=VerificationResult(
            tier=2,
            passed=False,
            outcome="fail",
            reason_code="incomplete_deliverable_placeholder",
            severity_class="semantic",
            feedback="Placeholder marker reported by verifier.",
            metadata={"issues": ["placeholder marker present"]},
        ))

        tool_calls = [
            MockToolCallRecord(
                tool="write_file",
                args={"path": "report.md"},
                result=ToolResult.ok("ok", files_changed=["report.md"]),
            ),
        ]
        prior_tool_calls = [
            MockToolCallRecord(
                tool="write_file",
                args={"path": "prior-draft.md"},
                result=ToolResult.ok("ok", files_changed=["prior-draft.md"]),
            ),
        ]
        result = await gates.verify(
            _make_subtask(subtask_id="phase-a"),
            "summary",
            tool_calls,
            tmp_path,
            evidence_tool_calls=prior_tool_calls,
            tier=2,
            task_id="task-prior-marker",
        )

        assert not result.passed
        assert result.reason_code == "incomplete_deliverable_placeholder"
        scan = result.metadata.get("deterministic_placeholder_scan", {})
        assert "prior-draft.md" in scan.get("matched_files", [])
        assert scan.get("matched_file_count") == 1

    @pytest.mark.asyncio
    async def test_clean_scan_with_insufficient_coverage_preserves_fail(self, tmp_path):
        process = _make_process(deliverables={"phase-a": ["report.md", "appendix.md"]})
        (tmp_path / "report.md").write_text("Final report")
        (tmp_path / "appendix.md").write_text("Appendix notes")
        config = VerificationConfig(
            tier1_enabled=True,
            tier2_enabled=True,
            contradiction_guard_enabled=True,
            contradiction_guard_strict_coverage=True,
            contradiction_scan_min_files_for_sufficiency=3,
        )
        router = MagicMock(spec=ModelRouter)
        prompts = PromptAssembler(process=process)
        gates = VerificationGates(router, prompts, config, process=process)
        gates._tier2.verify = AsyncMock(return_value=VerificationResult(
            tier=2,
            passed=False,
            outcome="fail",
            reason_code="incomplete_deliverable_placeholder",
            severity_class="semantic",
            feedback="Verifier reported placeholder marker.",
            metadata={"issues": ["placeholder marker present"]},
        ))

        result = await gates.verify(
            _make_subtask(subtask_id="phase-a"),
            "summary",
            [],
            tmp_path,
            tier=2,
            task_id="task-insufficient-coverage",
        )

        assert not result.passed
        assert result.reason_code == "incomplete_deliverable_placeholder"
        assert result.metadata.get("contradiction_detected") is False
        assert result.metadata.get("contradiction_detected_no_downgrade") is True
        scan = result.metadata.get("deterministic_placeholder_scan", {})
        assert scan.get("coverage_sufficient") is False
        assert "minimum_file_coverage_not_met" in scan.get("coverage_insufficient_reason", "")

    @pytest.mark.asyncio
    async def test_cap_exhaustion_marks_insufficient_coverage_and_no_downgrade(self, tmp_path):
        process = _make_process(deliverables={"phase-a": ["a.md", "b.md"]})
        (tmp_path / "a.md").write_text("File A clean")
        (tmp_path / "b.md").write_text("File B clean")
        config = VerificationConfig(
            tier1_enabled=True,
            tier2_enabled=True,
            contradiction_guard_enabled=True,
            contradiction_guard_strict_coverage=True,
            contradiction_scan_max_files=1,
            contradiction_scan_min_files_for_sufficiency=2,
        )
        router = MagicMock(spec=ModelRouter)
        prompts = PromptAssembler(process=process)
        event_bus = EventBus()
        events = []
        event_bus.subscribe_all(lambda event: events.append(event))
        gates = VerificationGates(
            router,
            prompts,
            config,
            process=process,
            event_bus=event_bus,
        )
        gates._tier2.verify = AsyncMock(return_value=VerificationResult(
            tier=2,
            passed=False,
            outcome="fail",
            reason_code="incomplete_deliverable_placeholder",
            severity_class="semantic",
            feedback="Verifier reported placeholder marker.",
            metadata={"issues": ["placeholder marker present"]},
        ))

        result = await gates.verify(
            _make_subtask(subtask_id="phase-a"),
            "summary",
            [],
            tmp_path,
            tier=2,
            task_id="task-cap-exhaustion",
        )

        assert not result.passed
        assert result.reason_code == "incomplete_deliverable_placeholder"
        scan = result.metadata.get("deterministic_placeholder_scan", {})
        assert scan.get("cap_exhausted") is True
        assert scan.get("cap_exhaustion_reason") == "max_files_reached"
        assert scan.get("coverage_sufficient") is False
        assert result.metadata.get("contradiction_detected_no_downgrade") is True
        contradiction_event = next(
            event for event in events
            if event.event_type == "verification_contradiction_detected"
        )
        assert contradiction_event.data.get("contradiction_detected_no_downgrade_count") == 1
        assert contradiction_event.data.get("cap_exhaustion_count") == 1

    @pytest.mark.asyncio
    async def test_large_binary_and_symlink_candidates_are_skipped(self, tmp_path):
        process = _make_process(
            deliverables={"phase-a": ["report.md", "large.md", "binary.md", "linked.md"]},
        )
        (tmp_path / "report.md").write_text("Clean final report")
        (tmp_path / "large.md").write_text("A" * 400)
        (tmp_path / "binary.md").write_bytes(b"\x00\x01\x02binary")
        try:
            (tmp_path / "linked.md").symlink_to(tmp_path / "report.md")
        except OSError:
            pytest.skip("Symlink not supported in this environment")
        config = VerificationConfig(
            tier1_enabled=True,
            tier2_enabled=True,
            contradiction_guard_enabled=True,
            contradiction_guard_strict_coverage=True,
            contradiction_scan_max_file_bytes=128,
            contradiction_scan_min_files_for_sufficiency=2,
        )
        router = MagicMock(spec=ModelRouter)
        prompts = PromptAssembler(process=process)
        gates = VerificationGates(router, prompts, config, process=process)
        gates._tier2.verify = AsyncMock(return_value=VerificationResult(
            tier=2,
            passed=False,
            outcome="fail",
            reason_code="incomplete_deliverable_placeholder",
            severity_class="semantic",
            feedback="Verifier reported placeholder marker.",
            metadata={"issues": ["placeholder marker present"]},
        ))

        result = await gates.verify(
            _make_subtask(subtask_id="phase-a"),
            "summary",
            [],
            tmp_path,
            tier=2,
            task_id="task-skip-behavior",
        )

        assert not result.passed
        assert result.reason_code == "incomplete_deliverable_placeholder"
        scan = result.metadata.get("deterministic_placeholder_scan", {})
        assert scan.get("scanned_file_count") == 1
        assert scan.get("skipped_large_file_count", 0) >= 1
        assert scan.get("skipped_binary_file_count", 0) >= 1
        assert scan.get("skipped_symlink_count", 0) >= 1
        assert scan.get("coverage_sufficient") is False

    @pytest.mark.asyncio
    async def test_policy_engine_merges_tier1_warnings_with_tier2_pass(self):
        process = _make_process(regex_rules=[{
            "name": "no-todo",
            "description": "No TODO markers",
            "check": "TODO",
            "severity": "error",
            "type": "regex",
            "target": "output",
        }])
        config = VerificationConfig(
            tier1_enabled=True,
            tier2_enabled=True,
            policy_engine_enabled=True,
            regex_default_advisory=True,
        )
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(return_value=ModelResponse(
            text=json.dumps({"passed": True, "issues": [], "confidence": 0.91}),
            usage=TokenUsage(input_tokens=40, output_tokens=25, total_tokens=65),
        ))
        router.select = MagicMock(return_value=model)
        prompts = PromptAssembler(process=process)

        gates = VerificationGates(router, prompts, config, process=process)
        result = await gates.verify(
            _make_subtask(subtask_id="s1"),
            "Contains TODO marker",
            [],
            None,
            tier=2,
            task_id="task-1",
        )
        assert result.passed
        assert result.outcome == "pass_with_warnings"

    @pytest.mark.asyncio
    async def test_shadow_compare_emits_false_negative_candidate_for_legacy_regex(self):
        process = _make_process(regex_rules=[{
            "name": "no-todo",
            "description": "No TODO markers",
            "check": "TODO",
            "severity": "error",
            "type": "regex",
            "target": "output",
        }])
        config = VerificationConfig(
            tier1_enabled=True,
            tier2_enabled=True,
            policy_engine_enabled=True,
            regex_default_advisory=True,
            shadow_compare_enabled=True,
        )
        router = MagicMock(spec=ModelRouter)
        model = AsyncMock()
        model.roles = ["verifier", "extractor"]
        model.complete = AsyncMock(return_value=ModelResponse(
            text=json.dumps({"passed": True, "issues": [], "confidence": 0.88}),
            usage=TokenUsage(input_tokens=40, output_tokens=25, total_tokens=65),
        ))
        router.select = MagicMock(return_value=model)
        prompts = PromptAssembler(process=process)
        event_bus = EventBus()
        events = []
        event_bus.subscribe_all(lambda event: events.append(event))

        gates = VerificationGates(
            router,
            prompts,
            config,
            process=process,
            event_bus=event_bus,
        )
        result = await gates.verify(
            _make_subtask(subtask_id="s1"),
            "Contains TODO marker",
            [],
            None,
            tier=2,
            task_id="task-1",
        )
        assert result.passed
        assert result.outcome == "pass_with_warnings"
        event_types = [event.event_type for event in events]
        assert "verification_shadow_diff" in event_types
        assert "verification_false_negative_candidate" in event_types

    def test_classify_shadow_diff_reason_diff(self):
        classification = VerificationGates.classify_shadow_diff(
            VerificationResult(
                tier=2,
                passed=False,
                outcome="fail",
                reason_code="unconfirmed_critical_path",
            ),
            VerificationResult(
                tier=2,
                passed=False,
                outcome="fail",
                reason_code="recommendation_unconfirmed",
            ),
        )
        assert classification == "both_fail_reason_diff"


# --- DeterministicVerifier with process definitions ---


def _make_process(
    deliverables: dict[str, list[str]] | None = None,
    regex_rules: list | None = None,
    static_checks: dict[str, object] | None = None,
):
    """Build a mock ProcessDefinition for verification tests."""
    from loom.processes.schema import (
        PhaseTemplate,
        ProcessDefinition,
        VerificationPolicyContract,
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

    policy = VerificationPolicyContract(
        static_checks=dict(static_checks or {}),
    )

    return ProcessDefinition(
        name="test-proc",
        phases=phases,
        verification_rules=rules,
        verification_policy=policy,
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

    @pytest.mark.asyncio
    async def test_terminal_synthesis_fails_when_upstream_deliverables_missing(
        self,
        tmp_path,
    ):
        process = _make_process(deliverables={
            "prep": ["prep.md"],
            "synth": ["final.md"],
        })
        (tmp_path / "final.md").write_text("Final report", encoding="utf-8")

        verifier = DeterministicVerifier(process=process)
        result = await verifier.verify(
            _make_subtask(
                subtask_id="synth",
                depends_on=["prep"],
                is_synthesis=True,
            ),
            "Composed final output.",
            [],
            tmp_path,
        )

        assert not result.passed
        failed = {c.name: c for c in result.checks if not c.passed}
        assert "synthesis_input_ready_prep" in failed
        assert "prep.md" in (failed["synthesis_input_ready_prep"].detail or "")

    @pytest.mark.asyncio
    async def test_terminal_synthesis_fails_when_upstream_inputs_not_integrated(
        self,
        tmp_path,
    ):
        process = _make_process(deliverables={
            "prep": ["prep.md"],
            "synth": ["final.md"],
        })
        (tmp_path / "prep.md").write_text("Prepared evidence", encoding="utf-8")
        (tmp_path / "final.md").write_text("Standalone summary", encoding="utf-8")

        verifier = DeterministicVerifier(process=process)
        result = await verifier.verify(
            _make_subtask(
                subtask_id="synth",
                depends_on=["prep"],
                is_synthesis=True,
            ),
            "Composed final output.",
            [],
            tmp_path,
        )

        assert not result.passed
        failed = {c.name: c for c in result.checks if not c.passed}
        assert "synthesis_input_integrated_prep" in failed

    @pytest.mark.asyncio
    async def test_terminal_synthesis_passes_when_upstream_inputs_integrated(
        self,
        tmp_path,
    ):
        process = _make_process(deliverables={
            "prep": ["prep.md"],
            "synth": ["final.md"],
        })
        (tmp_path / "prep.md").write_text("Prepared evidence", encoding="utf-8")
        (tmp_path / "final.md").write_text(
            "Final report built from prep.md findings.",
            encoding="utf-8",
        )

        verifier = DeterministicVerifier(process=process)
        result = await verifier.verify(
            _make_subtask(
                subtask_id="synth",
                depends_on=["prep"],
                is_synthesis=True,
            ),
            "Integrated prep.md into final output.",
            [],
            tmp_path,
        )

        assert result.passed
        names = {c.name for c in result.checks}
        assert "synthesis_input_ready_prep" in names
        assert "synthesis_input_integrated_prep" in names


class TestDeterministicVerifierSemanticAgnostic:
    @pytest.mark.asyncio
    async def test_does_not_run_domain_semantic_coverage_checks(self, tmp_path):
        (tmp_path / "market-condition-scorecard.csv").write_text(
            "market_id,geography,dimension,evidence_id,source_url\n"
            "AB-PESTLE-POL-001,Alberta,Political,EV-111,https://example.com/ab\n"
            "AB-PESTLE-ECO-001,Alberta,Economic,EV-112,\n",
            encoding="utf-8",
        )
        process = _make_process(deliverables={
            "environmental-scan": ["market-condition-scorecard.csv"],
        })
        verifier = DeterministicVerifier(process=process)
        subtask = _make_subtask(
            subtask_id="environmental-scan",
            description=(
                "Perform environmental scans for priority markets and include "
                "evidence-backed implications."
            ),
            acceptance_criteria=(
                "For each priority market, include a concise PESTLE-style readout "
                "with evidence-backed implications."
            ),
        )

        result = await verifier.verify(
            subtask,
            "Generated environmental scan deliverables.",
            [],
            tmp_path,
        )

        assert result.passed
        assert not any(
            c.name.startswith("market_evidence_")
            or c.name.startswith("market_dimension_coverage_")
            or c.name.startswith("sources_present_")
            or c.name.startswith("claim_evidence_map_")
            for c in result.checks
        )


class TestDeterministicVerifierRegexRules:
    """Tests for regex rule severity handling."""

    @pytest.mark.asyncio
    async def test_error_severity_rule_is_advisory_by_default(self):
        """Regex rules are advisory by default unless explicitly hard."""
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
        assert result.passed
        assert result.outcome == "pass_with_warnings"

    @pytest.mark.asyncio
    async def test_hard_enforced_regex_rule_fails_on_match(self):
        process = _make_process(regex_rules=[{
            "name": "no-tbd-hard",
            "description": "No TBD markers allowed",
            "check": "TBD",
            "severity": "error",
            "type": "regex",
            "target": "output",
            "enforcement": "hard",
        }])
        v = DeterministicVerifier(process=process)
        result = await v.verify(
            _make_subtask(), "This has TBD in it", [], None,
        )
        assert not result.passed
        assert result.outcome == "fail"

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
