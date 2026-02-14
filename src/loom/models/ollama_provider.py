"""Ollama model provider.

Connects to Ollama's native API with support for tool calling
and thinking mode switching for Qwen3 models.
"""

from __future__ import annotations

import json
import time

import httpx

from collections.abc import AsyncGenerator

from loom.config import ModelConfig
from loom.models.base import ModelProvider, ModelResponse, StreamChunk, TokenUsage, ToolCall


class OllamaProvider(ModelProvider):
    """Provider for Ollama local models."""

    def __init__(self, config: ModelConfig, provider_name: str = "", tier_override: int = 0):
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(300.0),
        )
        self._model = config.model
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        self._provider_name = provider_name or config.model
        self._roles = list(config.roles)
        self._tier = tier_override or self._infer_tier()

    def _infer_tier(self) -> int:
        """Infer tier from model name."""
        model_lower = self._model.lower()
        if any(k in model_lower for k in ["70b", "72b", "large"]):
            return 3
        if any(k in model_lower for k in ["14b", "32b", "medium"]):
            return 2
        return 1

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> ModelResponse:
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self._temperature,
                "num_predict": max_tokens or self._max_tokens,
            },
        }
        if tools:
            payload["tools"] = self._format_ollama_tools(tools)
        if response_format:
            payload["format"] = response_format.get("type", "json")

        start = time.monotonic()
        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        latency = int((time.monotonic() - start) * 1000)

        data = response.json()
        message = data.get("message", {})

        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = []
            for i, tc in enumerate(message["tool_calls"]):
                func = tc.get("function", {})
                args = func.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append(ToolCall(
                    id=f"call_{i}",
                    name=func.get("name", ""),
                    arguments=args,
                ))

        # Ollama token usage
        usage = TokenUsage(
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        )

        return ModelResponse(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            raw=json.dumps(message),
            usage=usage,
            model=self._model,
            latency_ms=latency,
        )

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a completion from Ollama, yielding text tokens as they arrive."""
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature if temperature is not None else self._temperature,
                "num_predict": max_tokens or self._max_tokens,
            },
        }
        if tools:
            payload["tools"] = self._format_ollama_tools(tools)

        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            accumulated_tool_calls: list[dict] = []

            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                message = data.get("message", {})
                text = message.get("content") or ""
                done = data.get("done", False)

                # Collect tool calls (only available in final message)
                if message.get("tool_calls"):
                    accumulated_tool_calls.extend(message["tool_calls"])

                parsed_tools = None
                usage = None

                if done:
                    # Parse accumulated tool calls
                    if accumulated_tool_calls:
                        parsed_tools = []
                        for i, tc in enumerate(accumulated_tool_calls):
                            func = tc.get("function", {})
                            args = func.get("arguments", {})
                            if isinstance(args, str):
                                args = json.loads(args)
                            parsed_tools.append(ToolCall(
                                id=f"call_{i}",
                                name=func.get("name", ""),
                                arguments=args,
                            ))

                    usage = TokenUsage(
                        input_tokens=data.get("prompt_eval_count", 0),
                        output_tokens=data.get("eval_count", 0),
                        total_tokens=(
                            data.get("prompt_eval_count", 0)
                            + data.get("eval_count", 0)
                        ),
                    )

                yield StreamChunk(
                    text=text,
                    done=done,
                    tool_calls=parsed_tools,
                    usage=usage,
                )

                if done:
                    return

    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/api/tags", timeout=5.0)
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def tier(self) -> int:
        return self._tier

    @property
    def roles(self) -> list[str]:
        return self._roles

    @staticmethod
    def _format_ollama_tools(tools: list[dict]) -> list[dict]:
        """Format tools for Ollama's tool calling format."""
        formatted = []
        for tool in tools:
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            })
        return formatted

    async def close(self) -> None:
        await self._client.aclose()


class ThinkingModeManager:
    """Manages thinking/acting mode for models that support it.

    Qwen3 models support /think and /no_think mode switching.
    Planner/replanner roles use thinking mode for deeper reasoning.
    Executor/extractor/verifier roles use acting mode for reliability.
    """

    THINKING_ROLES = frozenset({"planner", "replanner"})

    def prepare_messages(self, messages: list[dict], role: str) -> list[dict]:
        """Inject thinking mode control based on the role."""
        messages = [dict(m) for m in messages]  # shallow copy

        if role in self.THINKING_ROLES:
            mode_instruction = "/think"
        else:
            mode_instruction = "/no_think"

        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{mode_instruction}\n{messages[0]['content']}"
        else:
            messages.insert(0, {"role": "system", "content": mode_instruction})

        return messages
