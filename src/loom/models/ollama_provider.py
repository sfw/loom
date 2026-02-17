"""Ollama model provider.

Connects to Ollama's native API with support for tool calling,
multimodal content (images), and thinking mode switching for Qwen3 models.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx

from loom.config import ModelConfig
from loom.content import ThinkingBlock
from loom.content_utils import encode_image_base64, pdf_pages_to_images
from loom.models.base import (
    ModelConnectionError,
    ModelProvider,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCall,
)

logger = logging.getLogger(__name__)


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
        self._capabilities = config.resolved_capabilities

    def _infer_tier(self) -> int:
        """Infer tier from model name."""
        model_lower = self._model.lower()
        if any(k in model_lower for k in ["70b", "72b", "large"]):
            return 3
        if any(k in model_lower for k in ["14b", "32b", "medium"]):
            return 2
        return 1

    @staticmethod
    async def _http_error_body(response: httpx.Response, limit: int = 200) -> str:
        """Safely extract an HTTP error body from normal or streaming responses."""
        try:
            body = await response.aread()
            if body:
                return body.decode("utf-8", errors="replace")[:limit]
        except Exception:
            pass

        try:
            return str(response.text)[:limit]
        except Exception:
            return "<response body unavailable>"

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> ModelResponse:
        ollama_messages = self._build_ollama_messages(messages)
        payload: dict = {
            "model": self._model,
            "messages": ollama_messages,
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
        try:
            response = await self._client.post(
                "/api/chat", json=payload,
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise ModelConnectionError(
                f"Cannot connect to Ollama at "
                f"{self._client.base_url}: {e}",
                original=e,
            ) from e
        except httpx.TimeoutException as e:
            raise ModelConnectionError(
                f"Ollama request timed out ({self._model}): {e}",
                original=e,
            ) from e
        except httpx.HTTPStatusError as e:
            body_text = await self._http_error_body(e.response)
            raise ModelConnectionError(
                f"Ollama returned HTTP {e.response.status_code}: "
                f"{body_text}",
                original=e,
            ) from e
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
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        logger.warning("Malformed tool args from Ollama: %s", str(args)[:200])
                        args = {}
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

        # Capture thinking if present
        thinking_blocks = None
        thinking_text = message.get("thinking", "")
        if thinking_text:
            thinking_blocks = [ThinkingBlock(thinking=thinking_text)]

        return ModelResponse(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            thinking=thinking_blocks,
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
        ollama_messages = self._build_ollama_messages(messages)
        payload: dict = {
            "model": self._model,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "temperature": temperature if temperature is not None else self._temperature,
                "num_predict": max_tokens or self._max_tokens,
            },
        }
        if tools:
            payload["tools"] = self._format_ollama_tools(tools)

        try:
            async with self._client.stream(
                "POST", "/api/chat", json=payload,
            ) as response:
                if response.is_error:
                    body_text = await self._http_error_body(response)
                    raise ModelConnectionError(
                        f"Ollama returned HTTP {response.status_code}: "
                        f"{body_text}",
                    )

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
                                    try:
                                        args = json.loads(args)
                                    except json.JSONDecodeError:
                                        logger.warning(
                                            "Malformed tool args from Ollama stream: %s",
                                            str(args)[:200],
                                        )
                                        args = {}
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
        except httpx.ConnectError as e:
            raise ModelConnectionError(
                f"Cannot connect to Ollama at "
                f"{self._client.base_url}: {e}",
                original=e,
            ) from e
        except httpx.TimeoutException as e:
            raise ModelConnectionError(
                f"Ollama streaming timed out ({self._model}): {e}",
                original=e,
            ) from e
        except httpx.ReadError as e:
            raise ModelConnectionError(
                f"Ollama stream interrupted ({self._model}): {e}",
                original=e,
            ) from e

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

    def _build_ollama_messages(self, messages: list[dict]) -> list[dict]:
        """Convert messages to Ollama format with multimodal support.

        Ollama puts images in a separate 'images' field on the message,
        not in the content. Extracts image content blocks from tool results.
        """
        if not self._capabilities.vision:
            return messages

        result = []
        for msg in messages:
            out = {"role": msg.get("role", "user"), "content": msg.get("content", "")}

            # Copy tool_calls if present
            if msg.get("tool_calls"):
                out["tool_calls"] = msg["tool_calls"]
            if msg.get("tool_call_id"):
                out["tool_call_id"] = msg["tool_call_id"]

            # Extract images from tool result content blocks
            if msg.get("role") == "tool":
                images = self._extract_images_from_tool_result(msg.get("content", ""))
                if images:
                    out["images"] = images

            result.append(out)
        return result

    def _extract_images_from_tool_result(self, content: str) -> list[str]:
        """Extract base64 images from a tool result JSON string."""
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return []

        blocks = parsed.get("content_blocks", [])
        images: list[str] = []
        for block in blocks:
            btype = block.get("type", "")
            if btype == "image":
                source_path = block.get("source_path", "")
                if source_path:
                    data = encode_image_base64(Path(source_path))
                    if data:
                        images.append(data)
            elif btype == "document":
                source_path = block.get("source_path", "")
                pr = block.get("page_range")
                if source_path:
                    page_images = pdf_pages_to_images(
                        Path(source_path),
                        tuple(pr) if pr else None,
                    )
                    images.extend(page_images)
        return images

    async def close(self) -> None:
        await self._client.aclose()
