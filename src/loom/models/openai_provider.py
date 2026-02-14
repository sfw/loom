"""OpenAI-compatible model provider.

Connects to any OpenAI-compatible API endpoint:
MLX server, LM Studio, vLLM, llama.cpp server, etc.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator

import httpx

from loom.config import ModelConfig
from loom.models.base import (
    ModelConnectionError,
    ModelProvider,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCall,
)


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
        try:
            response = await self._client.post(
                "/chat/completions", json=payload,
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise ModelConnectionError(
                f"Cannot connect to model server at "
                f"{self._client.base_url}: {e}",
                original=e,
            ) from e
        except httpx.TimeoutException as e:
            raise ModelConnectionError(
                f"Model request timed out ({self._model}): {e}",
                original=e,
            ) from e
        except httpx.HTTPStatusError as e:
            raise ModelConnectionError(
                f"Model server returned HTTP "
                f"{e.response.status_code}: "
                f"{e.response.text[:200]}",
                original=e,
            ) from e
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

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a completion from OpenAI-compatible endpoint via SSE."""
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens or self._max_tokens,
            "temperature": temperature if temperature is not None else self._temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = self._format_tools(tools)

        try:
            cm = self._client.stream(
                "POST", "/chat/completions", json=payload,
            )
            response = await cm.__aenter__()
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise ModelConnectionError(
                f"Cannot connect to model server at "
                f"{self._client.base_url}: {e}",
                original=e,
            ) from e
        except httpx.TimeoutException as e:
            raise ModelConnectionError(
                f"Streaming request timed out ({self._model}): {e}",
                original=e,
            ) from e
        except httpx.HTTPStatusError as e:
            raise ModelConnectionError(
                f"Model server returned HTTP "
                f"{e.response.status_code}: "
                f"{e.response.text[:200]}",
                original=e,
            ) from e

        try:
            # Buffer tool call deltas until complete
            tool_call_buffers: dict[int, dict] = {}

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    # Final chunk — assemble tool calls
                    parsed_tools = None
                    if tool_call_buffers:
                        parsed_tools = []
                        for idx in sorted(tool_call_buffers):
                            buf = tool_call_buffers[idx]
                            args_str = buf.get("arguments", "")
                            try:
                                args = (
                                    json.loads(args_str)
                                    if args_str
                                    else {}
                                )
                            except json.JSONDecodeError:
                                args = {}
                            parsed_tools.append(ToolCall(
                                id=buf.get("id", ""),
                                name=buf.get("name", ""),
                                arguments=args,
                            ))
                    yield StreamChunk(
                        text="", done=True,
                        tool_calls=parsed_tools,
                    )
                    return

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choice = data.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                text = delta.get("content") or ""

                # Accumulate tool call deltas
                if delta.get("tool_calls"):
                    for tc_delta in delta["tool_calls"]:
                        idx = tc_delta.get("index", 0)
                        if idx not in tool_call_buffers:
                            tool_call_buffers[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        buf = tool_call_buffers[idx]
                        if tc_delta.get("id"):
                            buf["id"] = tc_delta["id"]
                        func = tc_delta.get("function", {})
                        if func.get("name"):
                            buf["name"] = func["name"]
                        if func.get("arguments"):
                            buf["arguments"] += func["arguments"]

                # Check for usage in the chunk
                usage = None
                if data.get("usage"):
                    u = data["usage"]
                    usage = TokenUsage(
                        input_tokens=u.get(
                            "prompt_tokens", 0,
                        ),
                        output_tokens=u.get(
                            "completion_tokens", 0,
                        ),
                        total_tokens=u.get(
                            "total_tokens", 0,
                        ),
                    )

                if text:
                    yield StreamChunk(
                        text=text, done=False, usage=usage,
                    )
        except httpx.ReadError as e:
            raise ModelConnectionError(
                f"Stream interrupted ({self._model}): {e}",
                original=e,
            ) from e
        finally:
            await cm.__aexit__(None, None, None)

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
