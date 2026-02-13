"""OpenAI-compatible model provider.

Connects to any OpenAI-compatible API endpoint:
MLX server, LM Studio, vLLM, llama.cpp server, etc.
"""

from __future__ import annotations

import json
import time

import httpx

from loom.config import ModelConfig
from loom.models.base import ModelProvider, ModelResponse, TokenUsage, ToolCall


class OpenAICompatibleProvider(ModelProvider):
    """Provider for OpenAI-compatible API endpoints."""

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
        """Infer tier from model name if not explicitly set."""
        model_lower = self._model.lower()
        if any(k in model_lower for k in ["70b", "72b", "m2.1", "large"]):
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
            "max_tokens": max_tokens or self._max_tokens,
            "temperature": temperature if temperature is not None else self._temperature,
        }
        if tools:
            payload["tools"] = self._format_tools(tools)
        if response_format:
            payload["response_format"] = response_format

        start = time.monotonic()
        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        latency = int((time.monotonic() - start) * 1000)

        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]

        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = []
            for tc in message["tool_calls"]:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    name=tc["function"]["name"],
                    arguments=args,
                ))

        usage_data = data.get("usage", {})
        usage = TokenUsage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return ModelResponse(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            raw=json.dumps(message),
            usage=usage,
            model=self._model,
            latency_ms=latency,
        )

    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/models", timeout=5.0)
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
    def _format_tools(tools: list[dict]) -> list[dict]:
        """Format tools for OpenAI-compatible tool calling."""
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
