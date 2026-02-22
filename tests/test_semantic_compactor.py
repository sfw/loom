"""Tests for semantic compactor model invocation behavior."""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from loom.engine.semantic_compactor import SemanticCompactor
from loom.models.base import ModelConnectionError, ModelResponse


class _ConfiguredTempModel:
    """Fake model asserting compactor uses model-configured temperature."""

    name = "configured-temp-model"
    tier = 1
    roles = ["extractor"]
    configured_temperature = 0.65

    def __init__(self) -> None:
        self.temperatures: list[float | None] = []

    async def complete(
        self,
        messages,
        tools=None,
        temperature=None,
        max_tokens=None,
        response_format=None,
    ):
        del messages, tools, max_tokens, response_format
        self.temperatures.append(temperature)
        if temperature != self.configured_temperature:
            raise ModelConnectionError(
                "unexpected temperature sent to compactor model"
            )
        return ModelResponse(
            text=json.dumps({"compressed_text": "compact output"}),
        )

    async def health_check(self) -> bool:
        return True


class _TempMismatchModel:
    """Fake model that always rejects the configured temperature value."""

    name = "temp-mismatch-model"
    tier = 1
    roles = ["extractor"]
    configured_temperature = 0.1

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        messages,
        tools=None,
        temperature=None,
        max_tokens=None,
        response_format=None,
    ):
        del messages, tools, temperature, max_tokens, response_format
        self.calls += 1
        raise ModelConnectionError(
            "Model server returned HTTP 400: "
            '{"error":{"message":"invalid temperature: only 1 is allowed for this model"}}',
        )

    async def health_check(self) -> bool:
        return True


class _RouterSelectStub:
    """Simple router stub implementing select(tier, role)."""

    def __init__(self, mapping: dict[str, object]):
        self._mapping = dict(mapping)
        self.calls: list[tuple[int, str]] = []

    def select(self, tier: int = 1, role: str = "executor"):
        self.calls.append((tier, role))
        model = self._mapping.get(role)
        if model is None:
            raise RuntimeError(f"no model for role {role}")
        return model


class _RoleModel:
    """Minimal fake model for role-routing tests."""

    def __init__(self, *, name: str, roles: list[str], output: str = "compact output"):
        self.name = name
        self.tier = 1
        self.roles = list(roles)
        self.configured_temperature = 0.0
        self._output = output

    async def complete(
        self,
        messages,
        tools=None,
        temperature=None,
        max_tokens=None,
        response_format=None,
    ):
        del messages, tools, temperature, max_tokens, response_format
        return ModelResponse(
            text=json.dumps({"compressed_text": self._output}),
        )

    async def health_check(self) -> bool:
        return True


class _CountingModel:
    """Fake model that counts complete() invocations."""

    name = "counting-model"
    tier = 1
    roles = ["extractor"]
    configured_temperature = 1.0

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        messages,
        tools=None,
        temperature=None,
        max_tokens=None,
        response_format=None,
    ):
        del messages, tools, temperature, max_tokens, response_format
        self.calls += 1
        return ModelResponse(
            text=json.dumps({"compressed_text": f"compact output {self.calls}"}),
        )

    async def health_check(self) -> bool:
        return True


class _RoundTargetCaptureModel:
    """Fake model recording requested hard limit chars across rounds."""

    name = "round-target-capture-model"
    tier = 1
    roles = ["extractor"]
    configured_temperature = 1.0

    def __init__(self) -> None:
        self.calls = 0
        self.target_chars: list[int] = []
        self.hard_limits: list[int] = []
        self.max_tokens_seen: list[int | None] = []

    async def complete(
        self,
        messages,
        tools=None,
        temperature=None,
        max_tokens=None,
        response_format=None,
    ):
        del tools, temperature, response_format
        self.calls += 1
        self.max_tokens_seen.append(max_tokens)
        user = ""
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                user = str(message.get("content", ""))
                break
        target_match = re.search(r"Target character budget:\s*<=\s*(\d+)\.", user)
        hard_match = re.search(r"Hard character limit:\s*<=\s*(\d+)\.", user)
        if target_match is None:
            raise AssertionError("missing target budget constraint in compactor prompt")
        if hard_match is None:
            raise AssertionError("missing hard limit constraint in compactor prompt")
        self.target_chars.append(int(target_match.group(1)))
        self.hard_limits.append(int(hard_match.group(1)))

        if self.calls == 1:
            return ModelResponse(
                text=json.dumps({"compressed_text": "y" * 900}),
            )
        return ModelResponse(
            text=json.dumps({"compressed_text": "z" * 150}),
        )

    async def health_check(self) -> bool:
        return True


class _LimitCaptureModel:
    """Fake model capturing hard limit and max token budget."""

    name = "limit-capture-model"
    tier = 1
    roles = ["extractor"]
    configured_temperature = 1.0
    configured_max_tokens = 1000

    def __init__(self) -> None:
        self.target_chars: list[int] = []
        self.hard_limits: list[int] = []
        self.max_tokens_seen: list[int | None] = []

    async def complete(
        self,
        messages,
        tools=None,
        temperature=None,
        max_tokens=None,
        response_format=None,
    ):
        del tools, temperature, response_format
        user = ""
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                user = str(message.get("content", ""))
                break

        target_match = re.search(r"Target character budget:\s*<=\s*(\d+)\.", user)
        hard_limit_match = re.search(r"Hard character limit:\s*<=\s*(\d+)\.", user)
        if target_match is None:
            raise AssertionError("missing target budget constraint in compactor prompt")
        if hard_limit_match is None:
            raise AssertionError("missing hard limit constraint in compactor prompt")

        self.target_chars.append(int(target_match.group(1)))
        self.hard_limits.append(int(hard_limit_match.group(1)))
        self.max_tokens_seen.append(max_tokens)

        return ModelResponse(
            text=json.dumps({"compressed_text": "ok"}),
        )

    async def health_check(self) -> bool:
        return True


class _PromptCaptureModel:
    """Fake model capturing compactor prompt messages."""

    name = "prompt-capture-model"
    tier = 1
    roles = ["extractor"]
    configured_temperature = 1.0

    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def complete(
        self,
        messages,
        tools=None,
        temperature=None,
        max_tokens=None,
        response_format=None,
    ):
        del tools, temperature, max_tokens, response_format
        self.messages = list(messages)
        return ModelResponse(
            text=json.dumps({"compressed_text": "compact output"}),
        )

    async def health_check(self) -> bool:
        return True


class _SequenceModel:
    """Fake model returning a deterministic sequence of response texts."""

    name = "sequence-model"
    tier = 1
    roles = ["compactor"]
    configured_temperature = 1.0

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.last_response_format = None

    async def complete(
        self,
        messages,
        tools=None,
        temperature=None,
        max_tokens=None,
        response_format=None,
    ):
        del messages, tools, temperature, max_tokens
        self.calls += 1
        self.last_response_format = response_format
        if self._responses:
            return ModelResponse(text=self._responses.pop(0))
        return ModelResponse(text=json.dumps({"compressed_text": "fallback"}))

    async def health_check(self) -> bool:
        return True


class _BlockingModel:
    """Fake model that blocks the first call to test in-flight dedup."""

    name = "blocking-model"
    tier = 1
    roles = ["compactor"]
    configured_temperature = 1.0

    def __init__(self) -> None:
        self.calls = 0
        self._gate = asyncio.Event()

    def release(self) -> None:
        self._gate.set()

    async def complete(
        self,
        messages,
        tools=None,
        temperature=None,
        max_tokens=None,
        response_format=None,
    ):
        del messages, tools, temperature, max_tokens, response_format
        self.calls += 1
        if self.calls == 1:
            await self._gate.wait()
        return ModelResponse(
            text=json.dumps({"compressed_text": "blocked compact output"}),
        )

    async def health_check(self) -> bool:
        return True


class _LengthThenJsonModel:
    """Fake model that truncates first, then returns valid JSON."""

    name = "length-then-json-model"
    tier = 1
    roles = ["compactor"]
    configured_temperature = 1.0
    configured_max_tokens = 8192

    def __init__(self) -> None:
        self.calls = 0
        self.max_tokens_seen: list[int | None] = []
        self.user_prompts: list[str] = []

    async def complete(
        self,
        messages,
        tools=None,
        temperature=None,
        max_tokens=None,
        response_format=None,
    ):
        del tools, temperature, response_format
        self.calls += 1
        self.max_tokens_seen.append(max_tokens)
        user = ""
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                user = str(message.get("content", ""))
                break
        self.user_prompts.append(user)

        if self.calls == 1:
            return ModelResponse(
                text="TRUNCATED BEFORE JSON_OBJECT",
                finish_reason="length",
            )
        return ModelResponse(
            text=json.dumps({"compressed_text": "repaired compact output"}),
            finish_reason="stop",
        )

    async def health_check(self) -> bool:
        return True


class _PartialLengthJsonModel:
    """Fake model returning truncated JSON with recoverable compressed_text."""

    name = "partial-length-json-model"
    tier = 1
    roles = ["compactor"]
    configured_temperature = 1.0

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        messages,
        tools=None,
        temperature=None,
        max_tokens=None,
        response_format=None,
    ):
        del messages, tools, temperature, max_tokens, response_format
        self.calls += 1
        return ModelResponse(
            text="{\"compressed_text\":\"" + ("x" * 900),
            finish_reason="length",
        )

    async def health_check(self) -> bool:
        return True


class _ExceedsThenShortModel:
    """Fake model that exceeds hard limit first, then succeeds."""

    name = "exceeds-then-short-model"
    tier = 1
    roles = ["compactor"]
    configured_temperature = 1.0

    def __init__(self) -> None:
        self.calls = 0
        self.user_prompts: list[str] = []

    async def complete(
        self,
        messages,
        tools=None,
        temperature=None,
        max_tokens=None,
        response_format=None,
    ):
        del tools, temperature, max_tokens, response_format
        self.calls += 1
        user = ""
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                user = str(message.get("content", ""))
                break
        self.user_prompts.append(user)

        if self.calls == 1:
            return ModelResponse(
                text=json.dumps({"compressed_text": "DRAFT_ONE " * 600}),
                finish_reason="stop",
            )
        return ModelResponse(
            text=json.dumps({"compressed_text": "short compact output"}),
            finish_reason="stop",
        )

    async def health_check(self) -> bool:
        return True


class _AlwaysOvershootModel:
    """Fake model that always returns valid JSON above the hard limit."""

    name = "always-overshoot-model"
    tier = 1
    roles = ["compactor"]
    configured_temperature = 1.0

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        messages,
        tools=None,
        temperature=None,
        max_tokens=None,
        response_format=None,
    ):
        del messages, tools, temperature, max_tokens, response_format
        self.calls += 1
        return ModelResponse(
            text=json.dumps({"compressed_text": "OVERSHOOT " * 120}),
            finish_reason="stop",
        )

    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_semantic_compactor_uses_model_configured_temperature():
    model = _ConfiguredTempModel()
    compactor = SemanticCompactor(model=model)

    result = await compactor.compact(
        "x" * 4000,
        max_chars=300,
        label="read_file tool output",
    )

    assert result == "compact output"
    assert model.temperatures and model.temperatures[0] == 0.65


@pytest.mark.asyncio
async def test_semantic_compactor_does_not_retry_temperature_validation_loop():
    model = _TempMismatchModel()
    compactor = SemanticCompactor(model=model)

    result = await compactor.compact(
        "x" * 3200,
        max_chars=260,
        label="tool output",
    )

    # Falls back to whitespace compaction without infinite retries.
    assert isinstance(result, str)
    assert len(result) > 260
    assert model.calls == 1


@pytest.mark.asyncio
async def test_semantic_compactor_does_not_cross_roles_by_default():
    router = _RouterSelectStub({
        "verifier": _RoleModel(name="verifier", roles=["verifier"]),
        "executor": _RoleModel(name="executor", roles=["executor"]),
    })
    compactor = SemanticCompactor(
        model_router=router,  # type: ignore[arg-type]
        role="extractor",
        tier=1,
    )

    source = "X " * 1800
    result = await compactor.compact(
        source,
        max_chars=240,
        label="strict role compaction",
    )

    # Default behavior is strict role selection: no verifier/executor fallback.
    assert router.calls == [(1, "extractor")]
    # Without an extractor model, compactor falls back to whitespace compaction.
    assert len(result) > 240
    assert "..." not in result


@pytest.mark.asyncio
async def test_semantic_compactor_prefers_compactor_role_by_default():
    router = _RouterSelectStub({
        "compactor": _RoleModel(name="compactor", roles=["compactor"]),
    })
    compactor = SemanticCompactor(
        model_router=router,  # type: ignore[arg-type]
        tier=1,
    )

    result = await compactor.compact(
        "x" * 3600,
        max_chars=300,
        label="compactor role",
    )

    assert result == "compact output"
    assert router.calls == [(1, "compactor")]


@pytest.mark.asyncio
async def test_semantic_compactor_allows_explicit_cross_role_fallback():
    router = _RouterSelectStub({
        "executor": _RoleModel(name="executor", roles=["executor"], output="compact output"),
    })
    compactor = SemanticCompactor(
        model_router=router,  # type: ignore[arg-type]
        role="extractor",
        tier=1,
        allow_role_fallback=True,
    )

    result = await compactor.compact(
        "x" * 3600,
        max_chars=300,
        label="fallback role compaction",
    )

    assert result == "compact output"
    assert router.calls == [(1, "extractor"), (1, "verifier"), (1, "executor")]


@pytest.mark.asyncio
async def test_semantic_compactor_emits_model_event_hook_payloads():
    model = _ConfiguredTempModel()
    events: list[dict] = []

    compactor = SemanticCompactor(
        model=model,
        model_event_hook=lambda payload: events.append(dict(payload)),
    )

    result = await compactor.compact(
        "x" * 3200,
        max_chars=280,
        label="hooked compaction",
    )

    assert result == "compact output"
    phases = [event.get("phase") for event in events]
    assert "start" in phases
    assert "done" in phases
    assert "validation" in phases

    start_events = [event for event in events if event.get("phase") == "start"]
    assert start_events
    start = start_events[0]
    assert start.get("compactor_requested_max_chars") == 280
    assert isinstance(start.get("compactor_hard_limit_chars"), int)
    assert isinstance(start.get("compactor_target_chars"), int)
    assert isinstance(start.get("compactor_max_tokens"), int)

    done_events = [event for event in events if event.get("phase") == "done"]
    assert done_events
    assert done_events[0].get("compactor_response_chars") == done_events[0].get("response_chars")
    assert any(
        "compressed_text" in str(event.get("response_preview", ""))
        for event in done_events
    )

    validation_events = [event for event in events if event.get("phase") == "validation"]
    assert validation_events
    assert "compactor_invalid_reason" in validation_events[0]
    assert "compactor_response_finish_reason" in validation_events[0]


@pytest.mark.asyncio
async def test_semantic_compactor_cache_reused_across_labels():
    model = _CountingModel()
    compactor = SemanticCompactor(model=model)

    source = "x" * 3600
    first = await compactor.compact(
        source,
        max_chars=300,
        label="read_file tool output",
    )
    second = await compactor.compact(
        source,
        max_chars=300,
        label="web_fetch tool output",
    )

    assert model.calls == 1
    assert first == second == "compact output 1"


@pytest.mark.asyncio
async def test_semantic_compactor_reduction_targets_requested_budget():
    model = _RoundTargetCaptureModel()
    compactor = SemanticCompactor(model=model)

    result = await compactor.compact(
        "x" * 1000,
        max_chars=200,
        label="target-capture",
    )

    assert result == "z" * 150
    assert model.target_chars[:2] == [150, 140]
    assert model.hard_limits[:2] == [200, 200]


@pytest.mark.asyncio
async def test_semantic_compactor_target_and_budget_link_to_hard_limit():
    model = _LimitCaptureModel()
    compactor = SemanticCompactor(
        model=model,
        response_tokens_floor=0,
        response_tokens_ratio=1.0,
        response_tokens_buffer=0,
        target_chars_ratio=0.75,
    )

    result = await compactor.compact(
        "x" * 5000,
        max_chars=2000,
        label="hard-limit-link",
    )

    assert result == "ok"
    assert model.target_chars[:1] == [750]
    assert model.hard_limits[:1] == [1000]
    expected_budget = compactor._compactor_response_max_tokens(1000, model)
    assert model.max_tokens_seen[:1] == [expected_budget]
    assert isinstance(expected_budget, int) and expected_budget < 1000


@pytest.mark.asyncio
async def test_semantic_compactor_prompt_forbids_meta_commentary():
    model = _PromptCaptureModel()
    compactor = SemanticCompactor(model=model)

    result = await compactor.compact(
        "x" * 3200,
        max_chars=260,
        label="prompt-guard",
    )

    assert result == "compact output"
    user_messages = [
        str(message.get("content", ""))
        for message in model.messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    assert user_messages
    prompt = user_messages[0]
    assert "Target character budget:" in prompt
    assert "Hard character limit:" in prompt
    assert "Output ONLY transformed source content." in prompt
    assert "Do NOT include meta commentary about the task" in prompt
    assert "Respond with exactly one JSON object" in prompt


@pytest.mark.asyncio
async def test_semantic_compactor_retries_once_for_invalid_json_then_succeeds():
    model = _SequenceModel([
        "not json",
        json.dumps({"compressed_text": "good compact output"}),
    ])
    compactor = SemanticCompactor(model=model)

    result = await compactor.compact(
        "x" * 3200,
        max_chars=260,
        label="json-retry",
    )

    assert result == "good compact output"
    assert model.calls == 2


@pytest.mark.asyncio
async def test_semantic_compactor_validation_retry_is_bounded():
    model = _SequenceModel([
        "not json",
        "still not json",
    ])
    compactor = SemanticCompactor(model=model)

    source = "x " * 1800
    result = await compactor.compact(
        source,
        max_chars=220,
        label="bounded-retry",
    )

    # Falls back after one retry (two attempts total) without truncation.
    assert model.calls == 2
    assert len(result) > 220
    assert "...[truncated]..." not in result


@pytest.mark.asyncio
async def test_semantic_compactor_inflight_dedup_for_same_key():
    model = _BlockingModel()
    compactor = SemanticCompactor(model=model)

    async def _run_one():
        return await compactor.compact(
            "x" * 3600,
            max_chars=300,
            label="inflight",
        )

    first_task = asyncio.create_task(_run_one())
    # Let the first task enter model.complete() before launching the second.
    await asyncio.sleep(0)
    second_task = asyncio.create_task(_run_one())
    await asyncio.sleep(0)
    model.release()

    first, second = await asyncio.gather(first_task, second_task)
    assert first == second == "blocked compact output"
    assert model.calls == 1


@pytest.mark.asyncio
async def test_semantic_compactor_fast_path_skips_model_when_normalization_fits():
    model = _CountingModel()
    compactor = SemanticCompactor(model=model)

    source = ("line 1   \nline 2   \nline 3   \n") * 2
    normalized = "line 1\nline 2\nline 3\nline 1\nline 2\nline 3"
    result = await compactor.compact(
        source,
        max_chars=len(normalized),
        label="fast-path",
    )

    assert result == normalized
    assert model.calls == 0


@pytest.mark.asyncio
async def test_semantic_compactor_retry_grows_tokens_on_truncated_invalid_json():
    model = _LengthThenJsonModel()
    compactor = SemanticCompactor(
        model=model,
        max_reduction_rounds=1,
        response_tokens_floor=120,
        response_tokens_ratio=0.25,
        response_tokens_buffer=0,
        token_headroom=16,
        target_chars_ratio=0.8,
    )

    result = await compactor.compact(
        "x" * 5000,
        max_chars=800,
        label="length-retry",
    )

    assert result == "repaired compact output"
    assert model.calls == 2
    assert model.max_tokens_seen[1] is not None
    assert model.max_tokens_seen[0] is not None
    assert model.max_tokens_seen[1] > model.max_tokens_seen[0]
    assert "Retry guidance:" in model.user_prompts[1]
    assert "truncated before valid JSON" in model.user_prompts[1]


@pytest.mark.asyncio
async def test_semantic_compactor_recovers_partial_truncated_json_without_hard_cap():
    model = _PartialLengthJsonModel()
    compactor = SemanticCompactor(
        model=model,
        max_reduction_rounds=1,
        response_tokens_floor=120,
        response_tokens_ratio=0.25,
        response_tokens_buffer=0,
        token_headroom=16,
        target_chars_ratio=0.8,
    )

    result = await compactor.compact(
        "x" * 5000,
        max_chars=800,
        label="partial-length",
    )

    assert model.calls == 2
    assert len(result) > 800
    assert "...[truncated]..." not in result


@pytest.mark.asyncio
async def test_semantic_compactor_retry_uses_prior_draft_on_overshoot():
    model = _ExceedsThenShortModel()
    compactor = SemanticCompactor(
        model=model,
        response_tokens_floor=0,
        response_tokens_ratio=0.5,
        response_tokens_buffer=64,
        target_chars_ratio=0.8,
    )

    result = await compactor.compact(
        "x" * 6000,
        max_chars=2400,
        label="overshoot-retry",
    )

    assert result == "short compact output"
    assert model.calls == 2
    assert "DRAFT_ONE" in model.user_prompts[1]
    assert "exceeded the hard character limit" in model.user_prompts[1]


@pytest.mark.asyncio
async def test_semantic_compactor_keeps_overshoot_after_retry_and_flags_warning():
    model = _AlwaysOvershootModel()
    events: list[dict] = []
    compactor = SemanticCompactor(
        model=model,
        max_reduction_rounds=1,
        model_event_hook=lambda payload: events.append(dict(payload)),
    )

    result = await compactor.compact(
        "x" * 5000,
        max_chars=400,
        label="overshoot-warning",
    )

    assert model.calls == 2
    assert len(result) > 400
    validation_events = [event for event in events if event.get("phase") == "validation"]
    assert len(validation_events) >= 2
    first = validation_events[0]
    final = validation_events[-1]
    assert first.get("compactor_invalid_reason") == "output_exceeds_target"
    assert first.get("compactor_warning") is False
    assert int(first.get("compactor_target_delta_chars", 0)) > 0
    assert final.get("compactor_output_valid") is True
    assert final.get("compactor_invalid_reason") == ""
    assert final.get("compactor_warning") is True
    assert final.get("compactor_warning_reason") == "output_exceeds_target"
    assert int(final.get("compactor_warning_delta_chars", 0)) > 0
