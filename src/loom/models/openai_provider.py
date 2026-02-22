"""OpenAI-compatible model provider.

Connects to any OpenAI-compatible API endpoint:
MLX server, LM Studio, vLLM, llama.cpp server, etc.
Supports multimodal content (images, documents) via content parts.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx

from loom.config import ModelConfig
from loom.content_utils import encode_file_base64, encode_image_base64
from loom.models.base import (
    ModelConnectionError,
    ModelProvider,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCall,
)
from loom.models.request_diagnostics import (
    collect_request_diagnostics,
    log_request_diagnostics,
)

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(ModelProvider):
    """Provider for OpenAI-compatible API endpoints."""

    ASSISTANT_CONTENT_FALLBACK = "Tool call context omitted."

    def __init__(self, config: ModelConfig, provider_name: str = "", tier_override: int = 0):
        self._config = config
        headers: dict[str, str] = {}
        api_key = config.api_key.strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(300.0),
            headers=headers,
        )
        self._model = config.model
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        self._reasoning_effort = str(getattr(config, "reasoning_effort", "") or "").strip()
        self._provider_name = provider_name or config.model
        self._roles = list(config.roles)
        self._tier = tier_override or self._infer_tier()
        self._capabilities = config.resolved_capabilities

    def _infer_tier(self) -> int:
        """Infer tier from model name if not explicitly set."""
        model_lower = self._model.lower()
        if any(k in model_lower for k in ["70b", "72b", "m2.1", "large"]):
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

    @staticmethod
    def _needs_reasoning_content_retry(status_code: int, body_text: str) -> bool:
        """Detect providers that require reasoning_content on tool-call turns."""
        body = body_text.lower()
        return (
            status_code == 400
            and "reasoning_content" in body
            and "tool call message" in body
        )

    @staticmethod
    def _inject_reasoning_content(messages: list[dict]) -> tuple[list[dict], bool]:
        """Add reasoning_content to assistant tool-call messages when missing."""
        patched: list[dict] = []
        changed = False

        for msg in messages:
            if (
                msg.get("role") == "assistant"
                and msg.get("tool_calls")
                and "reasoning_content" not in msg
            ):
                updated = dict(msg)
                content = updated.get("content")
                if isinstance(content, str) and content.strip():
                    updated["reasoning_content"] = content
                else:
                    # Some reasoning-enabled OpenAI-compatible providers reject
                    # empty reasoning_content on assistant tool-call turns.
                    updated["reasoning_content"] = (
                        OpenAICompatibleProvider.ASSISTANT_CONTENT_FALLBACK
                    )
                patched.append(updated)
                changed = True
                continue
            patched.append(msg)

        return patched, changed

    @classmethod
    def _extract_text_fragments(cls, value: object) -> list[str]:
        """Extract plain-text fragments from varied OpenAI-compatible payload shapes."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, (int, float, bool)):
            return [str(value)]
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                parts.extend(cls._extract_text_fragments(item))
            return parts
        if isinstance(value, dict):
            # Common content-part schema: {"type":"text","text":"..."}
            part_type = str(value.get("type", "")).strip().lower()
            if part_type in {"text", "output_text", "reasoning", "reasoning_text"}:
                candidate = (
                    value.get("text")
                    if "text" in value
                    else value.get("output_text")
                )
                if candidate is not None:
                    return cls._extract_text_fragments(candidate)

            parts: list[str] = []
            for key in (
                "content",
                "text",
                "output_text",
                "reasoning_content",
                "reasoning",
                "value",
                "summary",
            ):
                if key in value:
                    parts.extend(cls._extract_text_fragments(value.get(key)))
            return parts
        return []

    @classmethod
    def _extract_response_text(
        cls,
        *,
        data: dict,
        choice: dict,
        message: dict,
    ) -> str:
        """Extract assistant text across OpenAI and OpenAI-compatible response variants."""
        fragments = cls._extract_text_fragments(message.get("content"))
        if not fragments:
            fragments = cls._extract_text_fragments(message.get("reasoning_content"))
        if not fragments:
            fragments = cls._extract_text_fragments(choice.get("text"))
        if not fragments:
            fragments = cls._extract_text_fragments(data.get("output_text"))
        if not fragments:
            fragments = cls._extract_text_fragments(data.get("content"))
        if not fragments:
            output = data.get("output")
            if isinstance(output, list):
                for item in output:
                    if not isinstance(item, dict):
                        continue
                    fragments.extend(cls._extract_text_fragments(item.get("content")))
                    if not fragments:
                        fragments.extend(
                            cls._extract_text_fragments(item.get("output_text")),
                        )

        cleaned = [str(part).strip() for part in fragments if str(part).strip()]
        return "\n".join(cleaned).strip()

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> ModelResponse:
        converted = self._build_openai_messages(messages)
        payload: dict = {
            "model": self._model,
            "messages": converted,
            "max_tokens": max_tokens or self._max_tokens,
            "temperature": temperature if temperature is not None else self._temperature,
        }
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        if tools:
            payload["tools"] = self._format_tools(tools)
        if response_format:
            payload["response_format"] = response_format
        diagnostics = collect_request_diagnostics(
            messages=payload.get("messages", []),
            tools=payload.get("tools", []),
            payload=payload,
        )
        log_request_diagnostics(
            logger=logger,
            provider_name=self._provider_name,
            model_name=self._model,
            operation="complete",
            diagnostics=diagnostics,
        )

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
            body_text = await self._http_error_body(e.response)
            if self._needs_reasoning_content_retry(e.response.status_code, body_text):
                patched_messages, changed = self._inject_reasoning_content(converted)
                if changed:
                    retry_payload = dict(payload)
                    retry_payload["messages"] = patched_messages
                    try:
                        response = await self._client.post(
                            "/chat/completions", json=retry_payload,
                        )
                        response.raise_for_status()
                    except httpx.HTTPStatusError as retry_err:
                        retry_body = await self._http_error_body(retry_err.response)
                        raise ModelConnectionError(
                            f"Model server returned HTTP "
                            f"{retry_err.response.status_code}: "
                            f"{retry_body}",
                            original=retry_err,
                        ) from retry_err
                    except httpx.ConnectError as retry_err:
                        raise ModelConnectionError(
                            f"Cannot connect to model server at "
                            f"{self._client.base_url}: {retry_err}",
                            original=retry_err,
                        ) from retry_err
                    except httpx.TimeoutException as retry_err:
                        raise ModelConnectionError(
                            f"Model request timed out ({self._model}): {retry_err}",
                            original=retry_err,
                        ) from retry_err
                else:
                    raise ModelConnectionError(
                        f"Model server returned HTTP "
                        f"{e.response.status_code}: "
                        f"{body_text}",
                        original=e,
                    ) from e
            else:
                raise ModelConnectionError(
                    f"Model server returned HTTP "
                    f"{e.response.status_code}: "
                    f"{body_text}",
                    original=e,
                ) from e
        latency = int((time.monotonic() - start) * 1000)

        data = response.json()
        choices = data.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            raise ModelConnectionError(
                f"Malformed response from {self._model}: missing or empty 'choices'"
            )
        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = str(choice.get("finish_reason", "") or "").strip()

        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = []
            for index, tc in enumerate(message["tool_calls"]):
                func = tc.get("function", {})
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        logger.warning("Malformed tool args from OpenAI: %s", str(args)[:200])
                        args = {}
                tc_id = str(tc.get("id", "")).strip() or f"call_{index}"
                tool_calls.append(ToolCall(
                    id=tc_id,
                    name=func.get("name", ""),
                    arguments=args,
                ))

        usage_data = data.get("usage", {})
        usage = TokenUsage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        text = self._extract_response_text(
            data=data if isinstance(data, dict) else {},
            choice=choice if isinstance(choice, dict) else {},
            message=message if isinstance(message, dict) else {},
        )
        if not text and not tool_calls:
            logger.warning(
                "OpenAI-compatible response had empty assistant text: "
                "model=%s provider=%s message_keys=%s choice_keys=%s",
                self._model,
                self._provider_name,
                sorted(message.keys()) if isinstance(message, dict) else [],
                sorted(choice.keys()) if isinstance(choice, dict) else [],
            )

        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            raw=json.dumps(message),
            usage=usage,
            model=self._model,
            latency_ms=latency,
            finish_reason=finish_reason,
        )

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a completion from OpenAI-compatible endpoint via SSE."""
        converted = self._build_openai_messages(messages)
        payload: dict = {
            "model": self._model,
            "messages": converted,
            "max_tokens": max_tokens or self._max_tokens,
            "temperature": temperature if temperature is not None else self._temperature,
            "stream": True,
        }
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        if tools:
            payload["tools"] = self._format_tools(tools)
        diagnostics = collect_request_diagnostics(
            messages=payload.get("messages", []),
            tools=payload.get("tools", []),
            payload=payload,
        )
        log_request_diagnostics(
            logger=logger,
            provider_name=self._provider_name,
            model_name=self._model,
            operation="stream",
            diagnostics=diagnostics,
        )

        request_payload = payload
        tried_reasoning_retry = False

        try:
            while True:
                async with self._client.stream(
                    "POST", "/chat/completions", json=request_payload,
                ) as response:
                    if response.is_error:
                        body_text = await self._http_error_body(response)
                        if (
                            not tried_reasoning_retry
                            and self._needs_reasoning_content_retry(
                                response.status_code, body_text,
                            )
                        ):
                            patched_messages, changed = self._inject_reasoning_content(
                                request_payload.get("messages", []),
                            )
                            if changed:
                                request_payload = dict(request_payload)
                                request_payload["messages"] = patched_messages
                                tried_reasoning_retry = True
                                continue
                        raise ModelConnectionError(
                            f"Model server returned HTTP "
                            f"{response.status_code}: "
                            f"{body_text}",
                        )

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
                                    call_id = str(buf.get("id", "")).strip() or f"call_{idx}"
                                    parsed_tools.append(ToolCall(
                                        id=call_id,
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
        except httpx.ReadError as e:
            raise ModelConnectionError(
                f"Stream interrupted ({self._model}): {e}",
                original=e,
            ) from e

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

    def _build_openai_messages(self, messages: list[dict]) -> list[dict]:
        """Convert messages to OpenAI format with multimodal content parts.

        For tool results containing content blocks, the text output is kept
        as the tool result (must be a string), and multimodal parts are
        injected as a follow-up user message with image_url/file content.
        """
        if not self._capabilities or not self._capabilities.vision:
            return self._normalize_openai_messages(messages)

        result = []
        for msg in messages:
            if msg.get("role") == "tool":
                raw_content = msg.get("content", "")
                text_content = self._build_tool_result_content(raw_content)
                out = dict(msg)
                out["content"] = text_content
                result.append(out)

                # Inject multimodal parts as a follow-up user message
                parts = self._extract_multimodal_parts(raw_content)
                if parts:
                    result.append({
                        "role": "user",
                        "content": parts,
                    })
            else:
                result.append(msg)
        return self._normalize_openai_messages(result)

    @staticmethod
    def _normalize_openai_messages(messages: list[dict]) -> list[dict]:
        """Normalize message payloads for stricter OpenAI-compatible servers.

        Some providers reject:
        - assistant tool-call messages with null content
        - assistant messages with empty/whitespace-only content
        - missing tool-call IDs
        - tool messages with missing tool_call_id
        """
        normalized: list[dict] = []
        pending_tool_ids: list[str] = []

        for msg_index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue

            out = dict(message)
            role = str(out.get("role", "")).strip()
            if out.get("content") is None:
                out["content"] = ""

            if role == "assistant":
                content = out.get("content")
                if isinstance(content, str):
                    if not content.strip():
                        out["content"] = OpenAICompatibleProvider.ASSISTANT_CONTENT_FALLBACK
                elif content is None:
                    out["content"] = OpenAICompatibleProvider.ASSISTANT_CONTENT_FALLBACK
                else:
                    content_as_str = str(content)
                    out["content"] = (
                        content_as_str
                        if content_as_str.strip()
                        else OpenAICompatibleProvider.ASSISTANT_CONTENT_FALLBACK
                    )

                raw_tool_calls = out.get("tool_calls")
                tool_calls: list[dict] = []
                pending_tool_ids = []
                if isinstance(raw_tool_calls, list):
                    for tool_index, tool_call in enumerate(raw_tool_calls):
                        if not isinstance(tool_call, dict):
                            continue
                        call = dict(tool_call)
                        call_id = str(call.get("id", "")).strip()
                        if not call_id:
                            call_id = f"call_{msg_index}_{tool_index}"
                        call["id"] = call_id
                        call["type"] = "function"

                        function = call.get("function")
                        if not isinstance(function, dict):
                            function = {}
                        function_payload = dict(function)

                        name = function_payload.get("name")
                        function_payload["name"] = str(name or "")

                        arguments = function_payload.get("arguments")
                        if isinstance(arguments, dict):
                            function_payload["arguments"] = json.dumps(arguments)
                        elif arguments is None:
                            function_payload["arguments"] = "{}"
                        elif not isinstance(arguments, str):
                            function_payload["arguments"] = str(arguments)

                        call["function"] = function_payload
                        tool_calls.append(call)
                        pending_tool_ids.append(call_id)

                out["tool_calls"] = tool_calls

            elif role == "tool":
                current_id = str(out.get("tool_call_id", "")).strip()
                if not current_id and pending_tool_ids:
                    out["tool_call_id"] = pending_tool_ids.pop(0)
                elif current_id:
                    out["tool_call_id"] = current_id
                    if current_id in pending_tool_ids:
                        pending_tool_ids.remove(current_id)

            normalized.append(out)

        return normalized

    def _build_tool_result_content(self, result_json: str) -> str:
        """Convert tool result with content blocks to OpenAI format.

        OpenAI tool results must be strings, so we extract image data
        and include it as a user follow-up message if needed. For now,
        just return the text output — images are handled by converting
        tool result messages into multipart user messages when needed.
        """
        try:
            parsed = json.loads(result_json)
        except (json.JSONDecodeError, TypeError):
            return result_json

        # For OpenAI, tool results must be strings. Return text output.
        # Content blocks are handled at the message-building level.
        return parsed.get("output", result_json)

    def _extract_multimodal_parts(self, content: str) -> list[dict] | None:
        """Extract multimodal parts from a tool result JSON string.

        Returns a list of OpenAI content parts, or None if no multimodal
        content is present.
        """
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None

        blocks = parsed.get("content_blocks", [])
        if not blocks:
            return None

        caps = self._capabilities
        parts: list[dict] = []

        for block in blocks:
            btype = block.get("type", "")

            if btype == "text":
                parts.append({"type": "text", "text": block.get("text", "")})

            elif btype == "image" and caps.vision:
                source_path = block.get("source_path", "")
                data = encode_image_base64(Path(source_path)) if source_path else None
                if data:
                    media = block.get("media_type", "image/png")
                    parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media};base64,{data}",
                            "detail": "high",
                        },
                    })
                else:
                    parts.append({"type": "text", "text": block.get("text_fallback", "")})

            elif btype == "document" and caps.native_pdf:
                source_path = block.get("source_path", "")
                data = encode_file_base64(Path(source_path)) if source_path else None
                if data:
                    parts.append({
                        "type": "file",
                        "file": {
                            "filename": Path(source_path).name,
                            "file_data": f"data:application/pdf;base64,{data}",
                        },
                    })
                else:
                    parts.append({"type": "text", "text": block.get("text_fallback", "")})

            else:
                fb = block.get("text_fallback", "")
                if fb:
                    parts.append({"type": "text", "text": fb})

        return parts if parts else None

    async def close(self) -> None:
        await self._client.aclose()
