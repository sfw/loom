"""Focused tests for extracted verifier prompting helpers."""

from __future__ import annotations

from types import SimpleNamespace

from loom.engine.verification import prompting as verification_prompting
from loom.state.task_state import Subtask


def test_build_verifier_prompt_delegates_to_prompt_assembler() -> None:
    prompt_assembler = SimpleNamespace(
        build_verifier_prompt=lambda **kwargs: (
            f"{kwargs['subtask'].id}|{kwargs['phase_scope_default']}|"
            f"{kwargs['tool_calls_formatted']}"
        ),
    )
    subtask = Subtask(id="s1", description="Verify output")

    prompt = verification_prompting.build_verifier_prompt(
        prompt_assembler,
        subtask=subtask,
        result_summary="summary",
        tool_calls_formatted="tools",
        llm_rules=[],
        phase_scope_default="current_phase",
    )

    assert prompt == "s1|current_phase|tools"


def test_expected_keys_and_metadata_fields_from_process() -> None:
    process = SimpleNamespace(
        verifier_required_response_fields=lambda: ["confidence", "issues"],
        verifier_metadata_fields=lambda: ["missing_targets", "remediation_mode"],
    )
    prompts = SimpleNamespace(process=process)

    keys = verification_prompting.expected_verifier_response_keys(prompts)
    fields = verification_prompting.verifier_metadata_fields(prompts)

    assert keys[0] == "passed"
    assert "confidence" in keys
    assert "issues" in keys
    assert fields == ["missing_targets", "remediation_mode"]


def test_build_repair_prompt_includes_optional_metadata_hint() -> None:
    prompt = verification_prompting.build_repair_prompt(
        expected_keys=["passed", "confidence", "issues"],
        raw_text="not valid json",
        metadata_fields=["missing_targets"],
    )

    assert "\"passed\"" in prompt
    assert "missing_targets" in prompt
    assert "RAW OUTPUT" in prompt
