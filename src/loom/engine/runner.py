"""Subtask runner: encapsulates single-subtask execution.

Owns the tool-calling loop, response validation, verification gates,
and memory extraction for one subtask.  Returns compact structured
results so the orchestrator never touches raw prompts or messages.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from loom.config import Config
from loom.engine.verification import VerificationGates, VerificationResult
from loom.models.base import ModelResponse
from loom.models.router import ModelRouter, ResponseValidator
from loom.prompts.assembler import PromptAssembler
from loom.state.memory import MemoryEntry, MemoryManager
from loom.state.task_state import Subtask, Task, TaskStateManager
from loom.tools.registry import ToolRegistry, ToolResult
from loom.tools.workspace import ChangeLog


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


@dataclass
class SubtaskResult:
    """Result of a subtask execution."""

    status: str  # success, failed, blocked
    summary: str
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

    def __init__(
        self,
        model_router: ModelRouter,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager,
        prompt_assembler: PromptAssembler,
        state_manager: TaskStateManager,
        verification: VerificationGates,
        config: Config,
    ):
        self._router = model_router
        self._tools = tool_registry
        self._memory = memory_manager
        self._prompts = prompt_assembler
        self._state = state_manager
        self._verification = verification
        self._config = config
        self._validator = ResponseValidator()

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

        for _ in range(self.MAX_TOOL_ITERATIONS):
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
                    tool_result = await self._tools.execute(
                        tc.name, tc.arguments,
                        workspace=workspace,
                        changelog=changelog,
                        subtask_id=subtask.id,
                    )
                    tool_calls_record.append(ToolCallRecord(
                        tool=tc.name, args=tc.arguments, result=tool_result,
                    ))
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
            status="success",
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
            result.status = "failed"

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
            pass  # No running loop — skip

    async def _extract_memory(
        self, task_id: str, subtask_id: str, result: SubtaskResult,
    ) -> None:
        """Extract structured memory entries from subtask execution."""
        try:
            model = self._router.select(tier=1, role="extractor")
        except Exception:
            return

        tool_lines = []
        for tc in result.tool_calls:
            status = "OK" if tc.result.success else f"FAILED: {tc.result.error}"
            tool_lines.append(f"- {tc.tool}({json.dumps(tc.args)}) → {status}")
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
        except Exception:
            pass

    def _parse_memory_entries(
        self, response: ModelResponse, task_id: str, subtask_id: str,
    ) -> list[MemoryEntry]:
        """Parse extractor model response into MemoryEntry objects."""
        validation = self._validator.validate_json_response(
            response, expected_keys=["entries"]
        )
        if not validation.valid or validation.parsed is None:
            return []

        entries = []
        for e in validation.parsed.get("entries", []):
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
