"""Subtask runner: encapsulates single-subtask execution.

Owns the tool-calling loop, response validation, verification gates,
and memory extraction for one subtask.  Returns compact structured
results so the orchestrator never touches raw prompts or messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from loom.config import Config
from loom.engine.verification import VerificationGates, VerificationResult
from loom.events.bus import EventBus
from loom.events.types import TOKEN_STREAMED, TOOL_CALL_COMPLETED, TOOL_CALL_STARTED
from loom.models.base import ModelResponse
from loom.models.router import ModelRouter, ResponseValidator
from loom.prompts.assembler import PromptAssembler
from loom.state.memory import MemoryEntry, MemoryManager
from loom.state.task_state import Subtask, Task, TaskStateManager
from loom.tools.registry import ToolRegistry, ToolResult
from loom.tools.workspace import ChangeLog

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation during subtask execution."""

    tool: str
    args: dict
    result: ToolResult
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class SubtaskResultStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class SubtaskResult:
    """Result of a subtask execution."""

    status: SubtaskResultStatus = SubtaskResultStatus.SUCCESS
    summary: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    duration_seconds: float = 0.0
    tokens_used: int = 0
    model_used: str = ""


class SubtaskRunner:
    """Runs a single subtask to completion.

    Encapsulates:
    - Prompt assembly (memory retrieval + executor prompt)
    - Inner tool-calling loop
    - Response validation
    - Verification gates
    - Memory extraction (fire-and-forget)

    The orchestrator calls ``run()`` and gets back compact
    ``(SubtaskResult, VerificationResult)`` — no raw messages leak out.
    """

    MAX_TOOL_ITERATIONS = 20
    MAX_SUBTASK_WALL_CLOCK = 600  # 10 minutes per subtask

    def __init__(
        self,
        model_router: ModelRouter,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager,
        prompt_assembler: PromptAssembler,
        state_manager: TaskStateManager,
        verification: VerificationGates,
        config: Config,
        event_bus: EventBus | None = None,
    ):
        self._router = model_router
        self._tools = tool_registry
        self._memory = memory_manager
        self._prompts = prompt_assembler
        self._state = state_manager
        self._verification = verification
        self._config = config
        self._validator = ResponseValidator()
        self._event_bus = event_bus

    async def run(
        self,
        task: Task,
        subtask: Subtask,
        *,
        model_tier: int | None = None,
        retry_context: str = "",
        changelog: ChangeLog | None = None,
    ) -> tuple[SubtaskResult, VerificationResult]:
        """Execute a subtask: prompt → tool loop → verify → extract memory.

        Returns (SubtaskResult, VerificationResult).
        Memory extraction is fire-and-forget — it does not block the return.
        """
        start_time = time.monotonic()
        workspace = Path(task.workspace) if task.workspace else None

        # 1. Assemble prompt
        memory_entries = await self._memory.query_relevant(task.id, subtask.id)
        prompt = self._prompts.build_executor_prompt(
            task=task,
            subtask=subtask,
            state_manager=self._state,
            memory_entries=memory_entries,
            available_tools=self._tools.all_schemas(),
        )
        if retry_context:
            prompt = prompt + "\n\n" + retry_context

        # 2. Select model
        effective_tier = model_tier if model_tier is not None else subtask.model_tier
        model = self._router.select(tier=effective_tier, role="executor")

        # 3. Tool-calling loop
        messages: list[dict] = [{"role": "user", "content": prompt}]
        tool_calls_record: list[ToolCallRecord] = []
        total_tokens = 0
        response = None
        streaming = self._config.execution.enable_streaming

        for iteration in range(self.MAX_TOOL_ITERATIONS):
            # Wall-clock timeout check
            if time.monotonic() - start_time > self.MAX_SUBTASK_WALL_CLOCK:
                break
            if streaming:
                response = await self._stream_completion(
                    model, messages, self._tools.all_schemas(),
                    task_id=task.id, subtask_id=subtask.id,
                )
            else:
                response = await model.complete(
                    messages, tools=self._tools.all_schemas()
                )
            total_tokens += response.usage.total_tokens

            if response.has_tool_calls():
                # Validate tool calls before execution
                validation = self._validator.validate_tool_calls(
                    response, self._tools.all_schemas()
                )
                if not validation.valid:
                    messages.append({
                        "role": "assistant",
                        "content": response.text or None,
                    })
                    messages.append({
                        "role": "system",
                        "content": (
                            f"TOOL CALL ERROR: {validation.error}\n"
                            f"{validation.suggestion}\n"
                            "Please retry with valid tool calls."
                        ),
                    })
                    continue

                # Process validated tool calls
                messages.append({
                    "role": "assistant",
                    "content": response.text or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                })

                for tc in response.tool_calls:
                    self._emit_tool_event(
                        TOOL_CALL_STARTED, task.id, subtask.id,
                        tc.name, tc.arguments,
                    )
                    tool_result = await self._tools.execute(
                        tc.name, tc.arguments,
                        workspace=workspace,
                        changelog=changelog,
                        subtask_id=subtask.id,
                    )
                    tool_calls_record.append(ToolCallRecord(
                        tool=tc.name, args=tc.arguments, result=tool_result,
                    ))
                    self._emit_tool_event(
                        TOOL_CALL_COMPLETED, task.id, subtask.id,
                        tc.name, tc.arguments,
                        result=tool_result,
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result.to_json(),
                    })

                # Anti-amnesia reminder
                messages.append({
                    "role": "system",
                    "content": self._build_todo_reminder(task, subtask),
                })
            else:
                # Text-only response — subtask complete
                break

        elapsed = time.monotonic() - start_time
        summary = response.text[:200] if response and response.text else "No output"

        result = SubtaskResult(
            status=SubtaskResultStatus.SUCCESS,
            summary=summary,
            tool_calls=tool_calls_record,
            duration_seconds=elapsed,
            tokens_used=total_tokens,
            model_used=model.name,
        )

        # 4. Verification
        verification = await self._verification.verify(
            subtask=subtask,
            result_summary=summary,
            tool_calls=tool_calls_record,
            workspace=workspace,
            tier=subtask.verification_tier,
        )

        if not verification.passed:
            result.status = SubtaskResultStatus.FAILED

        # 5. Memory extraction — fire-and-forget
        self._spawn_memory_extraction(task.id, subtask.id, result)

        return result, verification

    def _spawn_memory_extraction(
        self, task_id: str, subtask_id: str, result: SubtaskResult,
    ) -> None:
        """Schedule memory extraction as a background task.

        Does not block the caller.  Failures are silently ignored
        (memory extraction is best-effort).
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._extract_memory(task_id, subtask_id, result)
            )
        except RuntimeError:
            logger.debug("Memory extraction skipped: no running event loop")

    async def _extract_memory(
        self, task_id: str, subtask_id: str, result: SubtaskResult,
    ) -> None:
        """Extract structured memory entries from subtask execution."""
        try:
            model = self._router.select(tier=1, role="extractor")
        except Exception as e:
            logger.debug("Memory extraction skipped (no extractor model): %s", e)
            return

        tool_lines = []
        for tc in result.tool_calls:
            status = "OK" if tc.result.success else f"FAILED: {tc.result.error}"
            line = f"- {tc.tool}({json.dumps(tc.args)}) → {status}"
            # Note multimodal content in the tool result
            if tc.result.content_blocks:
                block_types = [getattr(b, "type", "?") for b in tc.result.content_blocks]
                line += f" [content: {', '.join(block_types)}]"
            tool_lines.append(line)
        tool_calls_formatted = "\n".join(tool_lines) if tool_lines else "No tool calls."

        prompt = self._prompts.build_extractor_prompt(
            subtask_id=subtask_id,
            tool_calls_formatted=tool_calls_formatted,
            model_output=result.summary,
        )

        try:
            response = await model.complete([{"role": "user", "content": prompt}])
            entries = self._parse_memory_entries(response, task_id, subtask_id)
            if entries:
                await self._memory.store_many(entries)
        except Exception as e:
            logger.debug("Memory extraction failed for subtask %s: %s", subtask_id, e)

    def _parse_memory_entries(
        self, response: ModelResponse, task_id: str, subtask_id: str,
    ) -> list[MemoryEntry]:
        """Parse extractor model response into MemoryEntry objects.

        Accepts both formats:
        - JSON array: [{...}, ...]          (what the template asks for)
        - JSON object: {"entries": [...]}   (what validate_json_response expects)
        """
        raw_entries: list[dict] = []

        # First try parsing as a raw JSON array (matches the template)
        text = (response.text or "").strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    raw_entries = parsed
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: try the {"entries": [...]} format via validator
        if not raw_entries:
            validation = self._validator.validate_json_response(
                response, expected_keys=["entries"]
            )
            if validation.valid and validation.parsed is not None:
                raw_entries = validation.parsed.get("entries", [])

        entries = []
        for e in raw_entries:
            if not isinstance(e, dict):
                continue
            entry_type = e.get("type", "discovery")
            if entry_type not in (
                "decision", "error", "tool_result", "discovery", "artifact", "context"
            ):
                entry_type = "discovery"
            entries.append(MemoryEntry(
                task_id=task_id,
                subtask_id=subtask_id,
                entry_type=entry_type,
                summary=str(e.get("summary", ""))[:150],
                detail=str(e.get("detail", "")),
                tags=str(e.get("tags", "")),
            ))
        return entries

    async def _stream_completion(
        self,
        model,
        messages: list[dict],
        tools: list[dict],
        *,
        task_id: str = "",
        subtask_id: str = "",
    ) -> ModelResponse:
        """Stream a model completion, emitting TOKEN_STREAMED events.

        Collects all chunks and returns a complete ModelResponse,
        matching the interface of model.complete().
        """
        from loom.events.bus import Event

        text_parts: list[str] = []
        final_tool_calls = None
        final_usage = None

        async for chunk in model.stream(messages, tools=tools):
            if chunk.text:
                text_parts.append(chunk.text)
                # Emit token event
                if self._event_bus:
                    self._event_bus.emit(Event(
                        event_type=TOKEN_STREAMED,
                        task_id=task_id,
                        data={
                            "subtask_id": subtask_id,
                            "token": chunk.text,
                            "model": model.name,
                        },
                    ))
            if chunk.tool_calls is not None:
                final_tool_calls = chunk.tool_calls
            if chunk.usage is not None:
                final_usage = chunk.usage

        from loom.models.base import TokenUsage

        return ModelResponse(
            text="".join(text_parts),
            tool_calls=final_tool_calls,
            raw="",
            usage=final_usage or TokenUsage(),
            model=model.name,
        )

    def _emit_tool_event(
        self,
        event_type: str,
        task_id: str,
        subtask_id: str,
        tool_name: str,
        tool_args: dict,
        *,
        result: ToolResult | None = None,
    ) -> None:
        """Emit a tool call event to the event bus."""
        if not self._event_bus:
            return
        from loom.events.bus import Event

        data: dict = {
            "subtask_id": subtask_id,
            "tool": tool_name,
            "args": tool_args,
        }
        if result is not None:
            data["success"] = result.success
            data["error"] = result.error or ""
            if result.content_blocks:
                from loom.content import serialize_block
                data["content_blocks"] = [
                    serialize_block(b) for b in result.content_blocks
                ]
        self._event_bus.emit(Event(
            event_type=event_type, task_id=task_id, data=data,
        ))

    @staticmethod
    def _build_todo_reminder(task: Task, subtask: Subtask) -> str:
        return (
            f"CURRENT TASK STATE:\n"
            f"Goal: {task.goal}\n"
            f"Current subtask: {subtask.id} — {subtask.description}\n\n"
            f"REMAINING WORK FOR THIS SUBTASK:\n"
            f"Continue working on: {subtask.description}\n"
            f"Do NOT move to the next subtask. Complete ONLY this one.\n"
            f"When finished, provide a summary of what you accomplished."
        )
