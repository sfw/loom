# Spec 04: Model Router

## Overview

The model router selects which LLM to use for each invocation based on the role (planner, executor, extractor, verifier), the subtask's difficulty tier, and runtime performance history. It abstracts over multiple inference backends (MLX/LM Studio, Ollama, OpenAI-compatible APIs) through a unified interface.

## The Thinking vs. Acting Problem

Local models behave differently depending on their training. Reasoning-optimized models (trained with RL for chain-of-thought) resist producing deterministic structured output. Tool-calling models produce reliable JSON but lack deep reasoning. The router must assign the right model to the right role.

### Role Definitions

| Role | Needs | Best Model Type |
|------|-------|-----------------|
| `planner` | Complex reasoning, task decomposition, architectural thinking | Thinking mode (M2.1 with `<think>` tags, or Qwen3 `/think`) |
| `executor` | Reliable tool calling, structured JSON output, follow instructions | Acting/instruct mode (M2.1 instruct, Qwen3 `/no_think`) |
| `extractor` | Structured output from unstructured data, fast | Smallest viable (Qwen3 8B `/no_think`) |
| `verifier` | Independent assessment, fresh perspective | Different model instance than executor |
| `replanner` | Same as planner but with different context | Thinking mode, tier 3 |

## Model Pool Configuration

Models are defined in `loom.toml` and loaded at startup:

```toml
[models.primary]
provider = "openai_compatible"  # MLX server or LM Studio
base_url = "http://localhost:1234/v1"
model = "minimax-m2.1"
max_tokens = 4096
temperature = 0.1
roles = ["planner", "executor", "replanner"]
tier = 3

[models.utility]
provider = "ollama"
base_url = "http://localhost:11434"
model = "qwen3:8b"
max_tokens = 2048
temperature = 0.0
roles = ["extractor", "verifier", "executor"]
tier = 1

[models.medium]
# Optional: Qwen3 14B or 32B for mid-tier tasks
provider = "ollama"
base_url = "http://localhost:11434"
model = "qwen3:14b"
max_tokens = 4096
temperature = 0.1
roles = ["executor", "verifier"]
tier = 2
```

## Abstract Model Interface

All model providers implement this interface:

```python
# models/base.py

@dataclass
class ModelResponse:
    text: str                          # Full text response
    tool_calls: list[ToolCall] | None  # Parsed tool calls, if any
    raw: str                           # Raw response for logging
    usage: TokenUsage                  # Input/output token counts
    model: str                         # Which model was used
    latency_ms: int                    # Response time

    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

@dataclass
class ToolCall:
    id: str                            # Tool call ID (for matching results)
    name: str                          # Tool name
    arguments: dict                    # Parsed arguments

@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int

class ModelProvider(ABC):
    """Abstract base class for all model providers."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> ModelResponse:
        """Send a completion request and return structured response."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the model is available and responding."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        ...

    @property
    @abstractmethod
    def tier(self) -> int:
        """Model capability tier (1=fast/small, 2=medium, 3=large)."""
        ...
```

## Provider Implementations

### OpenAI-Compatible Provider (MLX, LM Studio)

```python
# models/openai_provider.py

class OpenAICompatibleProvider(ModelProvider):
    """
    Connects to any OpenAI-compatible API endpoint.
    Used for MLX server, LM Studio, vLLM, etc.
    """

    def __init__(self, config: ModelConfig):
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(300.0),  # Long timeout for large models
        )
        self._model = config.model
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature

    async def complete(self, messages, tools=None, **kwargs) -> ModelResponse:
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "temperature": kwargs.get("temperature", self._temperature),
        }
        if tools:
            payload["tools"] = self._format_tools(tools)

        start = time.monotonic()
        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        latency = int((time.monotonic() - start) * 1000)

        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]

        # Parse tool calls if present
        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"]["arguments"]),
                )
                for tc in message["tool_calls"]
            ]

        return ModelResponse(
            text=message.get("content", ""),
            tool_calls=tool_calls,
            raw=json.dumps(message),
            usage=TokenUsage(**data.get("usage", {})),
            model=self._model,
            latency_ms=latency,
        )
```

### Ollama Provider

```python
# models/ollama_provider.py

class OllamaProvider(ModelProvider):
    """
    Connects to Ollama's API. Handles Ollama-specific tool calling format
    and the /think /no_think mode switching for Qwen3.
    """

    async def complete(self, messages, tools=None, **kwargs) -> ModelResponse:
        # Ollama uses /api/chat endpoint
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self._temperature),
                "num_predict": kwargs.get("max_tokens", self._max_tokens),
            },
        }
        if tools:
            payload["tools"] = self._format_ollama_tools(tools)

        response = await self._client.post("/api/chat", json=payload)
        ...
```

## Model Router

```python
# models/router.py

class ModelRouter:
    """
    Routes model requests to the appropriate provider based on role and tier.

    Selection logic:
    1. Filter models by role compatibility
    2. Filter by minimum tier requirement
    3. Select the lowest-tier model that meets requirements (prefer cheap)
    4. Fall back to higher tier if lower tier is unavailable
    """

    def __init__(self, providers: dict[str, ModelProvider], config: Config):
        self._providers = providers  # name -> provider
        self._config = config
        self._role_map: dict[str, list[str]] = {}  # role -> [provider names]
        self._build_role_map()

    def select(self, tier: int = 1, role: str = "executor") -> ModelProvider:
        """
        Select the best model for the given role and minimum tier.

        Priority:
        1. Exact tier match with role capability
        2. Next tier up with role capability
        3. Any model with role capability
        4. Raise ModelNotAvailable
        """
        candidates = self._role_map.get(role, [])
        if not candidates:
            raise ModelNotAvailable(f"No model configured for role: {role}")

        # Sort by tier (prefer lowest that meets minimum)
        sorted_candidates = sorted(
            candidates,
            key=lambda name: abs(self._providers[name].tier - tier),
        )

        for name in sorted_candidates:
            provider = self._providers[name]
            if provider.tier >= tier:
                return provider

        # Fall back to highest-tier available
        return self._providers[sorted_candidates[-1]]

    async def health(self) -> dict[str, bool]:
        """Check health of all configured models."""
        results = {}
        for name, provider in self._providers.items():
            try:
                results[name] = await provider.health_check()
            except Exception:
                results[name] = False
        return results
```

## Thinking Mode Management

For Qwen3 models that support `/think` and `/no_think` mode switching:

```python
class ThinkingModeManager:
    """
    Manages thinking/acting mode for models that support it.
    Injects mode control into system messages.
    """

    def prepare_messages(
        self,
        messages: list[dict],
        role: str,
    ) -> list[dict]:
        """
        Inject thinking mode control based on the role.
        Planner/replanner roles use thinking mode.
        Executor/extractor/verifier roles use acting mode.
        """
        thinking_roles = {"planner", "replanner"}

        if role in thinking_roles:
            mode_instruction = "/think"
        else:
            mode_instruction = "/no_think"

        # Prepend mode instruction to system message
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = f"{mode_instruction}\n{messages[0]['content']}"
        else:
            messages.insert(0, {"role": "system", "content": mode_instruction})

        return messages
```

## Structured Output Validation

Every model response is validated at the harness level before being accepted:

```python
class ResponseValidator:
    """
    Validates model responses before they're processed.
    Catches malformed JSON, invalid tool calls, etc.
    """

    def validate_tool_calls(self, response: ModelResponse, available_tools: list[dict]) -> ValidationResult:
        """
        Check that:
        1. Tool names match registered tools
        2. Arguments match expected schema
        3. JSON is well-formed
        4. No hallucinated tool names
        """
        ...

    def validate_structured_output(self, response: ModelResponse, expected_schema: dict) -> ValidationResult:
        """
        For responses expected to be JSON (plans, extraction results),
        validate against the expected schema.
        """
        ...
```

## Acceptance Criteria

- [ ] Router correctly selects models by role and tier
- [ ] All three providers (OpenAI-compatible, Ollama, MLX) can complete requests
- [ ] Tool calls are correctly parsed from all provider response formats
- [ ] Thinking mode is correctly injected for Qwen3 models
- [ ] Health checks report model availability
- [ ] Structured output validation catches malformed JSON
- [ ] Invalid tool calls are caught before execution
- [ ] Router falls back gracefully when preferred model is unavailable
- [ ] Token usage is tracked per request
- [ ] Response latency is measured and available for logging
- [ ] Configuration changes to models don't require code changes
