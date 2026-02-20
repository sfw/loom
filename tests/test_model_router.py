"""Tests for model router and providers."""

from __future__ import annotations

import pytest

from loom.config import Config, ModelConfig
from loom.models.base import (
    ModelNotAvailableError,
    ModelProvider,
    ModelResponse,
    TokenUsage,
    ToolCall,
)
from loom.models.router import ModelRouter, ResponseValidator, _create_provider

# --- Mock Provider ---

class MockProvider(ModelProvider):
    """A mock model provider for testing."""

    def __init__(
        self,
        name: str = "mock",
        tier: int = 1,
        roles: list[str] | None = None,
        response_text: str = "Mock response",
        tool_calls: list[ToolCall] | None = None,
    ):
        self._name = name
        self._tier = tier
        self._roles = roles or ["executor"]
        self._response_text = response_text
        self._tool_calls = tool_calls
        self.call_count = 0
        self.last_messages = None

    async def complete(self, messages, tools=None, **kwargs) -> ModelResponse:
        self.call_count += 1
        self.last_messages = messages
        return ModelResponse(
            text=self._response_text,
            tool_calls=self._tool_calls,
            usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            model=self._name,
            latency_ms=42,
        )

    async def health_check(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return self._name

    @property
    def tier(self) -> int:
        return self._tier

    @property
    def roles(self) -> list[str]:
        return self._roles


class UnhealthyProvider(MockProvider):
    async def health_check(self) -> bool:
        return False


# --- Router Tests ---

class TestModelRouter:
    """Test model selection and routing."""

    def test_select_by_role(self):
        router = ModelRouter({
            "exec": MockProvider("exec", tier=1, roles=["executor"]),
            "plan": MockProvider("plan", tier=2, roles=["planner"]),
        })
        assert router.select(role="executor").name == "exec"
        assert router.select(role="planner").name == "plan"

    def test_select_no_model_for_role_raises(self):
        router = ModelRouter({
            "exec": MockProvider("exec", roles=["executor"]),
        })
        with pytest.raises(ModelNotAvailableError, match="No model configured for role"):
            router.select(role="planner")

    def test_select_by_tier_prefers_lowest_above_minimum(self):
        router = ModelRouter({
            "small": MockProvider("small", tier=1, roles=["executor"]),
            "medium": MockProvider("medium", tier=2, roles=["executor"]),
            "large": MockProvider("large", tier=3, roles=["executor"]),
        })
        # tier=2: should select medium (lowest at or above 2)
        assert router.select(tier=2, role="executor").name == "medium"

    def test_select_tier_exact_match(self):
        router = ModelRouter({
            "t1": MockProvider("t1", tier=1, roles=["executor"]),
            "t2": MockProvider("t2", tier=2, roles=["executor"]),
            "t3": MockProvider("t3", tier=3, roles=["executor"]),
        })
        assert router.select(tier=1, role="executor").name == "t1"
        assert router.select(tier=3, role="executor").name == "t3"

    def test_select_falls_back_to_higher_tier(self):
        router = ModelRouter({
            "large": MockProvider("large", tier=3, roles=["executor"]),
        })
        # Only tier 3 available, but tier 1 requested — should fall back
        assert router.select(tier=1, role="executor").name == "large"

    def test_select_falls_back_when_below_tier(self):
        router = ModelRouter({
            "small": MockProvider("small", tier=1, roles=["executor"]),
        })
        # Only tier 1, but tier 3 requested — still use it (best available)
        result = router.select(tier=3, role="executor")
        assert result.name == "small"

    def test_list_providers(self):
        router = ModelRouter({
            "a": MockProvider("model-a", tier=1, roles=["executor"]),
            "b": MockProvider("model-b", tier=2, roles=["planner"]),
        })
        providers = router.list_providers()
        assert len(providers) == 2
        names = {p["name"] for p in providers}
        assert "a" in names
        assert "b" in names

    async def test_health_check(self):
        router = ModelRouter({
            "healthy": MockProvider("h"),
            "unhealthy": UnhealthyProvider("u"),
        })
        health = await router.health()
        assert health["healthy"] is True
        assert health["unhealthy"] is False

    def test_add_provider(self):
        router = ModelRouter({})
        router.add_provider("new", MockProvider("new", roles=["verifier"]))
        assert router.select(role="verifier").name == "new"

    def test_from_config(self):
        config = Config(models={
            "test": ModelConfig(
                provider="ollama",
                base_url="http://localhost:11434",
                model="qwen3:8b",
                roles=["executor"],
            ),
        })
        router = ModelRouter.from_config(config)
        providers = router.list_providers()
        assert len(providers) == 1
        assert providers[0]["model"] == "test"

    def test_empty_router(self):
        router = ModelRouter()
        assert router.list_providers() == []
        with pytest.raises(ModelNotAvailableError):
            router.select(role="executor")


# --- Provider Creation ---

class TestCreateProvider:
    def test_create_ollama(self):
        config = ModelConfig(
            provider="ollama",
            base_url="http://localhost:11434",
            model="qwen3:8b",
            roles=["executor"],
        )
        provider = _create_provider("test", config)
        assert provider.name == "test"

    def test_create_openai_compatible(self):
        config = ModelConfig(
            provider="openai_compatible",
            base_url="http://localhost:1234/v1",
            model="test-model",
            roles=["planner"],
        )
        provider = _create_provider("test", config)
        assert provider.name == "test"

    def test_create_unknown_raises(self):
        config = ModelConfig(
            provider="unknown",
            base_url="http://localhost:1234",
            model="test",
        )
        with pytest.raises(ValueError, match="Unknown provider"):
            _create_provider("test", config)


# --- Response Validator ---

class TestResponseValidator:
    def test_valid_tool_calls(self):
        validator = ResponseValidator()
        response = ModelResponse(
            text="",
            tool_calls=[ToolCall(id="1", name="read_file", arguments={"path": "/tmp/x"})],
        )
        tools = [{"name": "read_file"}, {"name": "write_file"}]
        result = validator.validate_tool_calls(response, tools)
        assert result.valid

    def test_invalid_tool_name(self):
        validator = ResponseValidator()
        response = ModelResponse(
            text="",
            tool_calls=[ToolCall(id="1", name="hallucinated_tool", arguments={})],
        )
        tools = [{"name": "read_file"}]
        result = validator.validate_tool_calls(response, tools)
        assert not result.valid
        assert "Unknown tool" in result.error

    def test_missing_required_tool_argument(self):
        validator = ResponseValidator()
        response = ModelResponse(
            text="",
            tool_calls=[ToolCall(id="1", name="document_write", arguments={"title": "Report"})],
        )
        tools = [
            {
                "name": "document_write",
                "parameters": {
                    "type": "object",
                    "required": ["path"],
                    "properties": {
                        "path": {"type": "string"},
                        "title": {"type": "string"},
                    },
                },
            },
        ]
        result = validator.validate_tool_calls(response, tools)
        assert not result.valid
        assert "Missing required arguments for document_write: path" in result.error
        assert "required tool arguments" in result.suggestion

    def test_blank_required_tool_argument(self):
        validator = ResponseValidator()
        response = ModelResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="1",
                    name="write_file",
                    arguments={"path": "  ", "content": "x"},
                )
            ],
        )
        tools = [
            {
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "required": ["path", "content"],
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        ]
        result = validator.validate_tool_calls(response, tools)
        assert not result.valid
        assert "Missing required arguments for write_file: path" in result.error

    def test_no_tool_calls_is_valid(self):
        validator = ResponseValidator()
        response = ModelResponse(text="Just text")
        result = validator.validate_tool_calls(response, [])
        assert result.valid

    def test_valid_json_response(self):
        validator = ResponseValidator()
        response = ModelResponse(text='{"subtasks": []}')
        result = validator.validate_json_response(response, expected_keys=["subtasks"])
        assert result.valid
        assert result.parsed == {"subtasks": []}

    def test_invalid_json_response(self):
        validator = ResponseValidator()
        response = ModelResponse(text="This is not JSON")
        result = validator.validate_json_response(response)
        assert not result.valid
        assert "Invalid JSON" in result.error

    def test_json_with_markdown_fences(self):
        validator = ResponseValidator()
        response = ModelResponse(text='```json\n{"key": "value"}\n```')
        result = validator.validate_json_response(response)
        assert result.valid
        assert result.parsed == {"key": "value"}

    def test_json_embedded_in_explanatory_text(self):
        validator = ResponseValidator()
        response = ModelResponse(
            text='I checked the task.\n{"passed": true, "confidence": 0.9}\nDone.'
        )
        result = validator.validate_json_response(response, expected_keys=["passed"])
        assert result.valid
        assert result.parsed == {"passed": True, "confidence": 0.9}

    def test_json_prefers_candidate_with_expected_keys(self):
        validator = ResponseValidator()
        response = ModelResponse(
            text='Example schema: {"foo": 1}\nFinal: {"passed": false, "issues": []}'
        )
        result = validator.validate_json_response(response, expected_keys=["passed"])
        assert result.valid
        assert result.parsed == {"passed": False, "issues": []}

    def test_json_missing_keys(self):
        validator = ResponseValidator()
        response = ModelResponse(text='{"other": "value"}')
        result = validator.validate_json_response(response, expected_keys=["subtasks"])
        assert not result.valid
        assert "Missing required keys" in result.error


# --- ModelResponse ---

class TestModelResponse:
    def test_has_tool_calls_true(self):
        r = ModelResponse(text="", tool_calls=[ToolCall(id="1", name="t", arguments={})])
        assert r.has_tool_calls()

    def test_has_tool_calls_false(self):
        r = ModelResponse(text="text")
        assert not r.has_tool_calls()

    def test_has_tool_calls_empty_list(self):
        r = ModelResponse(text="", tool_calls=[])
        assert not r.has_tool_calls()
