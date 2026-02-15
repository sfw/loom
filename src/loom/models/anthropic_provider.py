"""Anthropic/Claude model provider.

Supports the Anthropic Messages API for Claude models with native
tool use, streaming, multimodal content, and extended thinking.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import httpx

from loom.config import ModelCapabilities
from loom.content import ThinkingBlock
from loom.content_utils import encode_file_base64, encode_image_base64
from loom.models.base import (
    ModelConnectionError,
    ModelProvider,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCall,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"


class AnthropicProvider(ModelProvider):
    """Model provider for the Anthropic Messages API (Claude).

    Supports:
    - Standard completions with tool use
    - Streaming responses
    - Native tool calling format
    """

    def __init__(
        self,
        name: str = "anthropic",
        model: str = "claude-sonnet-4-5-20250929",
        api_key: str = "",
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        tier: int = 2,
        roles: list[str] | None = None,
        capabilities: ModelCapabilities | None = None,
    ):
        self._name = name
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._tier = tier
        self._roles = roles or ["executor"]
        self._capabilities = capabilities or ModelCapabilities.auto_detect(
            "anthropic", model,
        )
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def tier(self) -> int:
        return self._tier

    @property
    def roles(self) -> list[str]:
        return list(self._roles)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=120.0,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": API_VERSION,
                    "content-type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if the Anthropic API is reachable."""
        try:
            client = await self._get_client()
            # A minimal request to check connectivity
            response = await client.post(
                "/v1/messages",
                json={
                    "model": self._model,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            return response.status_code in (200, 400, 401, 429)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Conversions: Loom format <-> Anthropic format
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Convert Loom/OpenAI-format messages to Anthropic format.

        Returns (system_prompt, messages) where system_prompt is extracted
        from the first message if it has role "system".
        """
        system_prompt = None
        anthropic_messages: list[dict] = []

        for msg in messages:
            role = msg.get("role", "")

            if role == "system":
                system_prompt = msg.get("content", "")
                continue

            if role == "assistant":
                content_blocks: list[dict] = []
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})

                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    arguments = fn.get("arguments", "{}")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": arguments,
                    })

                anthropic_messages.append({
                    "role": "assistant",
                    "content": content_blocks if content_blocks else [{"type": "text", "text": ""}],
                })

            elif role == "tool":
                # Anthropic expects tool results as user messages with tool_result blocks
                tool_call_id = msg.get("tool_call_id", "")
                content = msg.get("content", "")
                tool_content = self._build_tool_result_content(content)

                # Check if the previous message is already a user message with tool results
                if (
                    anthropic_messages
                    and anthropic_messages[-1]["role"] == "user"
                    and isinstance(anthropic_messages[-1]["content"], list)
                    and anthropic_messages[-1]["content"]
                    and anthropic_messages[-1]["content"][0].get("type") == "tool_result"
                ):
                    anthropic_messages[-1]["content"].append({
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": tool_content,
                    })
                else:
                    anthropic_messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": tool_content,
                        }],
                    })

            elif role == "user":
                anthropic_messages.append({
                    "role": "user",
                    "content": msg.get("content", ""),
                })

            else:
                # Unknown role — treat as user message to avoid silent data loss
                import logging
                logging.getLogger(__name__).warning(
                    "Unknown message role '%s', treating as user", role,
                )
                anthropic_messages.append({
                    "role": "user",
                    "content": msg.get("content", ""),
                })

        return system_prompt, anthropic_messages

    def _convert_tools(self, tools: list[dict] | None) -> list[dict] | None:
        """Convert OpenAI-format tool schemas to Anthropic format."""
        if not tools:
            return None

        anthropic_tools = []
        for tool in tools:
            # Tools may already be in Anthropic format (name, description, input_schema)
            # or in OpenAI format (type: function, function: {name, description, parameters})
            if "input_schema" in tool:
                anthropic_tools.append(tool)
            elif "function" in tool:
                fn = tool["function"]
                anthropic_tools.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })
            else:
                # Assume it's a flat tool schema from Loom's registry
                anthropic_tools.append({
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
                })

        return anthropic_tools if anthropic_tools else None

    def _parse_response(self, data: dict) -> ModelResponse:
        """Parse an Anthropic API response into Loom's ModelResponse."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        thinking_blocks: list[ThinkingBlock] = []

        for block in data.get("content", []):
            btype = block.get("type", "")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=block.get("input", {}),
                ))
            elif btype == "thinking":
                thinking_blocks.append(ThinkingBlock(
                    thinking=block.get("thinking", ""),
                    signature=block.get("signature", ""),
                ))
            elif btype == "redacted_thinking":
                thinking_blocks.append(ThinkingBlock(
                    thinking="[redacted]",
                    signature=block.get("data", ""),
                ))

        usage_data = data.get("usage", {})
        in_tok = usage_data.get("input_tokens", 0)
        out_tok = usage_data.get("output_tokens", 0)
        usage = TokenUsage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=in_tok + out_tok,
        )

        return ModelResponse(
            text="\n".join(text_parts) if text_parts else "",
            tool_calls=tool_calls,
            thinking=thinking_blocks if thinking_blocks else None,
            usage=usage,
            raw=data,
        )

    # ------------------------------------------------------------------
    # Multimodal content conversion
    # ------------------------------------------------------------------

    def _build_tool_result_content(self, result_json: str) -> str | list[dict]:
        """Convert a tool result with possible content blocks to Anthropic format."""
        try:
            parsed = json.loads(result_json)
        except (json.JSONDecodeError, TypeError):
            return result_json

        blocks = parsed.get("content_blocks")
        if not blocks:
            return parsed.get("output", result_json)

        caps = self._capabilities
        anthropic_blocks: list[dict] = []
        text_output = parsed.get("output", "")

        for block in blocks:
            btype = block.get("type", "")

            if btype == "text":
                anthropic_blocks.append({"type": "text", "text": block.get("text", "")})

            elif btype == "image" and caps.vision:
                source_path = block.get("source_path", "")
                image_data = encode_image_base64(Path(source_path)) if source_path else None
                if image_data:
                    anthropic_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": block.get("media_type", "image/png"),
                            "data": image_data,
                        },
                    })
                else:
                    anthropic_blocks.append({
                        "type": "text",
                        "text": block.get("text_fallback", ""),
                    })

            elif btype == "document" and caps.native_pdf:
                source_path = block.get("source_path", "")
                doc_data = encode_file_base64(Path(source_path)) if source_path else None
                if doc_data:
                    anthropic_blocks.append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": doc_data,
                        },
                    })
                else:
                    anthropic_blocks.append({
                        "type": "text",
                        "text": block.get("text_fallback", ""),
                    })

            else:
                fb = block.get("text_fallback", "")
                if fb:
                    anthropic_blocks.append({"type": "text", "text": fb})

        if anthropic_blocks:
            # Prepend the text output if it exists and no text block already covers it
            if text_output and not any(
                b.get("type") == "text" and b.get("text") == text_output
                for b in anthropic_blocks
            ):
                anthropic_blocks.insert(0, {"type": "text", "text": text_output})
            return anthropic_blocks

        return text_output

    # ------------------------------------------------------------------
    # ModelProvider interface
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> ModelResponse:
        """Send a completion request to the Anthropic API."""
        client = await self._get_client()
        system_prompt, anthropic_msgs = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)

        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens or self._max_tokens,
            "messages": anthropic_msgs,
        }

        if system_prompt:
            body["system"] = system_prompt
        if anthropic_tools:
            body["tools"] = anthropic_tools
        temp = temperature if temperature is not None else self._temperature
        if temp > 0:
            body["temperature"] = temp

        try:
            response = await client.post("/v1/messages", json=body)
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise ModelConnectionError(
                f"Cannot connect to Anthropic API at "
                f"{self._base_url}: {e}",
                original=e,
            ) from e
        except httpx.TimeoutException as e:
            raise ModelConnectionError(
                f"Anthropic request timed out ({self._model}): {e}",
                original=e,
            ) from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body_text = e.response.text[:200]
            if status == 401:
                msg = "Invalid Anthropic API key"
            elif status == 429:
                retry_after = e.response.headers.get("retry-after", "")
                if retry_after:
                    msg = f"Anthropic rate limit exceeded (retry-after: {retry_after}s)"
                else:
                    msg = "Anthropic rate limit exceeded"
            else:
                msg = f"Anthropic returned HTTP {status}"
            raise ModelConnectionError(
                f"{msg}: {body_text}", original=e,
            ) from e
        return self._parse_response(response.json())

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a completion from the Anthropic API."""
        client = await self._get_client()
        system_prompt, anthropic_msgs = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)

        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens or self._max_tokens,
            "messages": anthropic_msgs,
            "stream": True,
        }

        if system_prompt:
            body["system"] = system_prompt
        if anthropic_tools:
            body["tools"] = anthropic_tools
        temp = temperature if temperature is not None else self._temperature
        if temp > 0:
            body["temperature"] = temp

        # Accumulate tool calls from streaming events
        current_tool_calls: list[ToolCall] = []
        current_tool_json = ""
        current_tool_id = ""
        current_tool_name = ""
        input_tokens = 0
        output_tokens = 0

        try:
            cm = client.stream("POST", "/v1/messages", json=body)
            response = await cm.__aenter__()
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise ModelConnectionError(
                f"Cannot connect to Anthropic API at "
                f"{self._base_url}: {e}",
                original=e,
            ) from e
        except httpx.TimeoutException as e:
            raise ModelConnectionError(
                f"Anthropic streaming timed out "
                f"({self._model}): {e}",
                original=e,
            ) from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body_text = e.response.text[:200]
            if status == 401:
                msg = "Invalid Anthropic API key"
            elif status == 429:
                msg = "Anthropic rate limit exceeded"
            else:
                msg = f"Anthropic returned HTTP {status}"
            raise ModelConnectionError(
                f"{msg}: {body_text}", original=e,
            ) from e

        try:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                if event_type == "message_start":
                    usage_info = event.get("message", {}).get("usage", {})
                    input_tokens = usage_info.get("input_tokens", 0)

                elif event_type == "content_block_start":
                    block = event.get("content_block", {})
                    if block.get("type") == "tool_use":
                        current_tool_id = block.get("id", "")
                        current_tool_name = block.get("name", "")
                        current_tool_json = ""

                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield StreamChunk(text=delta.get("text", ""))
                    elif delta.get("type") == "input_json_delta":
                        current_tool_json += delta.get("partial_json", "")

                elif event_type == "content_block_stop":
                    if current_tool_name:
                        try:
                            arguments = json.loads(current_tool_json) if current_tool_json else {}
                        except json.JSONDecodeError:
                            arguments = {}
                        current_tool_calls.append(ToolCall(
                            id=current_tool_id,
                            name=current_tool_name,
                            arguments=arguments,
                        ))
                        current_tool_name = ""
                        current_tool_id = ""
                        current_tool_json = ""

                elif event_type == "message_delta":
                    usage_info = event.get("usage", {})
                    output_tokens = usage_info.get("output_tokens", 0)

            # Yield final chunk with tool calls and usage
            yield StreamChunk(
                tool_calls=(
                    current_tool_calls if current_tool_calls else None
                ),
                usage=TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                ),
            )
        except httpx.ReadError as e:
            raise ModelConnectionError(
                f"Anthropic stream interrupted "
                f"({self._model}): {e}",
                original=e,
            ) from e
        finally:
            await cm.__aexit__(None, None, None)
