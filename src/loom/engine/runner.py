"""Subtask runner: encapsulates single-subtask execution.

Owns the tool-calling loop, response validation, verification gates,
and memory extraction for one subtask.  Returns compact structured
results so the orchestrator never touches raw prompts or messages.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from loom.auth.runtime import AuthResolutionError, build_run_auth_context
from loom.config import Config
from loom.engine.semantic_compactor import SemanticCompactor
from loom.engine.verification import Check, VerificationGates, VerificationResult
from loom.events.bus import EventBus
from loom.events.types import (
    ARTIFACT_CONFINEMENT_VIOLATION,
    MODEL_INVOCATION,
    TOKEN_STREAMED,
    TOOL_CALL_COMPLETED,
    TOOL_CALL_STARTED,
)
from loom.models.base import ModelResponse
from loom.models.request_diagnostics import (
    collect_request_diagnostics,
    collect_response_diagnostics,
)
from loom.models.retry import ModelRetryPolicy, call_with_model_retry
from loom.models.router import ModelRouter, ResponseValidator
from loom.prompts.assembler import PromptAssembler
from loom.state.evidence import (
    extract_evidence_records,
    merge_evidence_records,
    summarize_evidence_records,
)
from loom.state.memory import MemoryEntry, MemoryManager
from loom.state.task_state import Subtask, Task, TaskStateManager
from loom.tools.registry import ToolRegistry, ToolResult
from loom.tools.workspace import ChangeLog
from loom.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)
_COMPACTOR_EVENT_CONTEXT: contextvars.ContextVar[tuple[str, str] | None] = (
    contextvars.ContextVar("runner_compactor_event_context", default=None)
)


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation during subtask execution."""

    tool: str
    args: dict
    result: ToolResult
    call_id: str = ""
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
    evidence_records: list[dict] = field(default_factory=list)


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
    MAX_SUBTASK_WALL_CLOCK = 1200  # 20 minutes per subtask
    MAX_MODEL_CONTEXT_TOKENS = 24_000
    MAX_STATE_SUMMARY_CHARS = 320
    MAX_VERIFICATION_SUMMARY_CHARS = 6000
    DEFAULT_TOOL_RESULT_OUTPUT_CHARS = 3_000
    HEAVY_TOOL_RESULT_OUTPUT_CHARS = 1_200
    COMPACT_TOOL_RESULT_OUTPUT_CHARS = 320
    COMPACT_TEXT_OUTPUT_CHARS = 600
    MINIMAL_TEXT_OUTPUT_CHARS = 180
    TOOL_CALL_ARGUMENT_CONTEXT_CHARS = 320
    COMPACT_TOOL_CALL_ARGUMENT_CHARS = 140
    _HEAVY_OUTPUT_TOOLS = frozenset({
        "web_fetch",
        "web_fetch_html",
        "web_search",
        "read_file",
        "search_files",
        "ripgrep_search",
        "list_directory",
        "glob_find",
        "conversation_recall",
    })
    TOOL_CALL_CONTEXT_PLACEHOLDER = "Tool call context omitted."
    LEGACY_TOOL_CALL_CONTEXT_PLACEHOLDER = "Tool call required to continue."
    _TOOL_CALL_CONTEXT_PLACEHOLDERS = frozenset({
        TOOL_CALL_CONTEXT_PLACEHOLDER.lower(),
        LEGACY_TOOL_CALL_CONTEXT_PLACEHOLDER.lower(),
    })
    _TODO_REMINDER_PREFIX = "CURRENT TASK STATE:\n"
    _WRITE_MUTATING_TOOLS = frozenset({
        "write_file",
        "edit_file",
        "document_write",
        "move_file",
        "delete_file",
        "spreadsheet",
    })
    _SPREADSHEET_WRITE_OPERATIONS = frozenset({
        "create",
        "add_rows",
        "add_column",
        "update_cell",
    })
    _VARIANT_SUFFIX_MARKERS = (
        "v",
        "rev",
        "copy",
        "draft",
        "final",
        "updated",
        "new",
    )

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
        runner_limits = getattr(getattr(config, "limits", None), "runner", None)
        self._max_tool_iterations = int(
            getattr(runner_limits, "max_tool_iterations", self.MAX_TOOL_ITERATIONS),
        )
        self._max_subtask_wall_clock_seconds = int(
            getattr(
                runner_limits,
                "max_subtask_wall_clock_seconds",
                self.MAX_SUBTASK_WALL_CLOCK,
            ),
        )
        self._max_model_context_tokens = int(
            getattr(
                runner_limits,
                "max_model_context_tokens",
                self.MAX_MODEL_CONTEXT_TOKENS,
            ),
        )
        self._max_state_summary_chars = int(
            getattr(
                runner_limits,
                "max_state_summary_chars",
                self.MAX_STATE_SUMMARY_CHARS,
            ),
        )
        self._max_verification_summary_chars = int(
            getattr(
                runner_limits,
                "max_verification_summary_chars",
                self.MAX_VERIFICATION_SUMMARY_CHARS,
            ),
        )
        self._default_tool_result_output_chars = int(
            getattr(
                runner_limits,
                "default_tool_result_output_chars",
                self.DEFAULT_TOOL_RESULT_OUTPUT_CHARS,
            ),
        )
        self._heavy_tool_result_output_chars = int(
            getattr(
                runner_limits,
                "heavy_tool_result_output_chars",
                self.HEAVY_TOOL_RESULT_OUTPUT_CHARS,
            ),
        )
        self._compact_tool_result_output_chars = int(
            getattr(
                runner_limits,
                "compact_tool_result_output_chars",
                self.COMPACT_TOOL_RESULT_OUTPUT_CHARS,
            ),
        )
        self._compact_text_output_chars = int(
            getattr(
                runner_limits,
                "compact_text_output_chars",
                self.COMPACT_TEXT_OUTPUT_CHARS,
            ),
        )
        self._minimal_text_output_chars = int(
            getattr(
                runner_limits,
                "minimal_text_output_chars",
                self.MINIMAL_TEXT_OUTPUT_CHARS,
            ),
        )
        self._tool_call_argument_context_chars = int(
            getattr(
                runner_limits,
                "tool_call_argument_context_chars",
                self.TOOL_CALL_ARGUMENT_CONTEXT_CHARS,
            ),
        )
        self._compact_tool_call_argument_chars = int(
            getattr(
                runner_limits,
                "compact_tool_call_argument_chars",
                self.COMPACT_TOOL_CALL_ARGUMENT_CHARS,
            ),
        )
        self._evidence_context_text_max_chars = int(
            getattr(
                getattr(config, "limits", None),
                "evidence_context_text_max_chars",
                4000,
            ),
        )

        compactor_limits = getattr(getattr(config, "limits", None), "compactor", None)
        self._compactor = SemanticCompactor(
            model_router,
            model_event_hook=self._emit_compactor_model_event,
            role="compactor",
            tier=1,
            allow_role_fallback=True,
            max_chunk_chars=int(
                getattr(
                    compactor_limits,
                    "max_chunk_chars",
                    SemanticCompactor._MAX_CHUNK_CHARS,
                ),
            ),
            max_chunks_per_round=int(
                getattr(
                    compactor_limits,
                    "max_chunks_per_round",
                    SemanticCompactor._MAX_CHUNKS_PER_ROUND,
                ),
            ),
            max_reduction_rounds=int(
                getattr(
                    compactor_limits,
                    "max_reduction_rounds",
                    SemanticCompactor._MAX_REDUCTION_ROUNDS,
                ),
            ),
            min_compact_target_chars=int(
                getattr(
                    compactor_limits,
                    "min_compact_target_chars",
                    SemanticCompactor._MIN_COMPACT_TARGET_CHARS,
                ),
            ),
            response_tokens_floor=int(
                getattr(
                    compactor_limits,
                    "response_tokens_floor",
                    SemanticCompactor._RESPONSE_TOKENS_FLOOR,
                ),
            ),
            response_tokens_ratio=float(
                getattr(
                    compactor_limits,
                    "response_tokens_ratio",
                    SemanticCompactor._RESPONSE_TOKENS_RATIO,
                ),
            ),
            response_tokens_buffer=int(
                getattr(
                    compactor_limits,
                    "response_tokens_buffer",
                    SemanticCompactor._RESPONSE_TOKENS_BUFFER,
                ),
            ),
            json_headroom_chars_floor=int(
                getattr(
                    compactor_limits,
                    "json_headroom_chars_floor",
                    SemanticCompactor._JSON_HEADROOM_CHARS_FLOOR,
                ),
            ),
            json_headroom_chars_ratio=float(
                getattr(
                    compactor_limits,
                    "json_headroom_chars_ratio",
                    SemanticCompactor._JSON_HEADROOM_CHARS_RATIO,
                ),
            ),
            json_headroom_chars_cap=int(
                getattr(
                    compactor_limits,
                    "json_headroom_chars_cap",
                    SemanticCompactor._JSON_HEADROOM_CHARS_CAP,
                ),
            ),
            chars_per_token_estimate=float(
                getattr(
                    compactor_limits,
                    "chars_per_token_estimate",
                    SemanticCompactor._CHARS_PER_TOKEN_ESTIMATE,
                ),
            ),
            token_headroom=int(
                getattr(
                    compactor_limits,
                    "token_headroom",
                    SemanticCompactor._TOKEN_HEADROOM,
                ),
            ),
            target_chars_ratio=float(
                getattr(
                    compactor_limits,
                    "target_chars_ratio",
                    SemanticCompactor._TARGET_CHARS_RATIO,
                ),
            ),
        )

    @staticmethod
    def _read_roots_for_task(task: Task, workspace: Path | None) -> list[Path]:
        """Resolve additional read-only roots for this task.

        Only accepts roots that are ancestors of the task workspace so reads
        can widen safely to parent trees without becoming arbitrary.
        """
        if workspace is None:
            return []
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        raw_roots = metadata.get("read_roots", [])
        if isinstance(raw_roots, str):
            raw_roots = [raw_roots]
        if not isinstance(raw_roots, list):
            return []

        resolved_workspace = workspace.resolve()
        roots: list[Path] = []
        seen: set[Path] = set()
        for raw in raw_roots:
            try:
                candidate = Path(str(raw)).expanduser().resolve()
            except Exception:
                continue
            # Disallow filesystem-root grants.
            if candidate == Path(candidate.anchor):
                continue
            try:
                resolved_workspace.relative_to(candidate)
            except ValueError:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            roots.append(candidate)
        return roots

    async def run(
        self,
        task: Task,
        subtask: Subtask,
        *,
        model_tier: int | None = None,
        retry_context: str = "",
        changelog: ChangeLog | None = None,
        prior_successful_tool_calls: list[ToolCallRecord] | None = None,
        prior_evidence_records: list[dict] | None = None,
        expected_deliverables: list[str] | None = None,
        enforce_deliverable_paths: bool = False,
        edit_existing_only: bool = False,
        retry_strategy: str = "",
    ) -> tuple[SubtaskResult, VerificationResult]:
        """Execute a subtask: prompt → tool loop → verify → extract memory.

        Returns (SubtaskResult, VerificationResult).
        Memory extraction is fire-and-forget — it does not block the return.
        """
        start_time = time.monotonic()
        compactor_context_token = _COMPACTOR_EVENT_CONTEXT.set((task.id, subtask.id))
        workspace = Path(task.workspace) if task.workspace else None
        read_roots = self._read_roots_for_task(task, workspace)
        auth_context = None
        try:
            metadata = task.metadata if isinstance(task.metadata, dict) else {}
            auth_context = build_run_auth_context(
                workspace=workspace,
                metadata=metadata,
            )
        except AuthResolutionError as e:
            failure_summary = f"Auth preflight failed: {e}"
            result = SubtaskResult(
                status=SubtaskResultStatus.FAILED,
                summary=failure_summary,
                duration_seconds=time.monotonic() - start_time,
                model_used="",
            )
            verification = VerificationResult(
                tier=1,
                passed=False,
                checks=[Check(name="auth_preflight", passed=False, detail=str(e))],
                feedback=failure_summary,
                outcome="fail",
                reason_code="auth_preflight_failed",
                metadata={"auth_error": str(e)},
            )
            _COMPACTOR_EVENT_CONTEXT.reset(compactor_context_token)
            return result, verification

        # 1. Assemble prompt
        memory_entries = await self._memory.query_relevant(task.id, subtask.id)
        evidence_summary = summarize_evidence_records(
            prior_evidence_records or [],
            max_entries=10,
        )
        prompt = self._prompts.build_executor_prompt(
            task=task,
            subtask=subtask,
            state_manager=self._state,
            memory_entries=memory_entries,
            available_tools=self._tools.all_schemas(),
            evidence_ledger_summary=evidence_summary,
        )
        if retry_context:
            prompt = prompt + "\n\n" + retry_context

        # 2. Select model
        effective_tier = model_tier if model_tier is not None else subtask.model_tier
        model = self._router.select(tier=effective_tier, role="executor")

        # 3. Tool-calling loop
        messages: list[dict] = [{"role": "user", "content": prompt}]
        tool_calls_record: list[ToolCallRecord] = []
        evidence_records_current: list[dict] = []
        known_evidence_ids: set[str] = {
            str(item.get("evidence_id", "")).strip()
            for item in (prior_evidence_records or [])
            if isinstance(item, dict)
        }
        known_evidence_ids.discard("")
        total_tokens = 0
        response = None
        streaming = self._config.execution.enable_streaming
        completed_normally = False
        interruption_reason: str | None = None
        budget_exhaustion_note: str | None = None
        canonical_deliverables = self._normalize_deliverable_paths(
            expected_deliverables or [],
            workspace=workspace,
        )
        iteration_budget = self._tool_iteration_budget(
            subtask=subtask,
            retry_strategy=retry_strategy,
            has_expected_deliverables=bool(canonical_deliverables),
            base_budget=self._max_tool_iterations,
        )

        for iteration in range(iteration_budget):
            # Wall-clock timeout check
            if time.monotonic() - start_time > self._max_subtask_wall_clock_seconds:
                interruption_reason = (
                    "Execution exceeded subtask time budget "
                    f"({self._max_subtask_wall_clock_seconds}s) before completion."
                )
                break
            messages = await self._compact_messages_for_model(messages)
            tool_schemas = self._tools.all_schemas()
            operation = "stream" if streaming else "complete"
            response = None
            policy = ModelRetryPolicy.from_execution_config(self._config.execution)
            invocation_attempt = 0
            request_diag = None

            async def _invoke_model():
                nonlocal invocation_attempt, request_diag
                invocation_attempt += 1
                request_diag = collect_request_diagnostics(
                    messages=messages,
                    tools=tool_schemas,
                    origin=f"runner.execute_subtask.{operation}",
                )
                self._emit_model_event(
                    task_id=task.id,
                    subtask_id=subtask.id,
                    model_name=model.name,
                    phase="start",
                    details={
                        **request_diag.to_event_payload(),
                        "iteration": iteration + 1,
                        "operation": operation,
                        "invocation_attempt": invocation_attempt,
                        "invocation_max_attempts": policy.max_attempts,
                    },
                )
                if streaming:
                    return await self._stream_completion(
                        model,
                        messages,
                        tool_schemas,
                        task_id=task.id,
                        subtask_id=subtask.id,
                    )
                return await model.complete(messages, tools=tool_schemas)

            def _on_invocation_failure(
                attempt: int,
                max_attempts: int,
                error: BaseException,
                remaining: int,
            ) -> None:
                self._emit_model_event(
                    task_id=task.id,
                    subtask_id=subtask.id,
                    model_name=model.name,
                    phase="done",
                    details={
                        "origin": request_diag.origin if request_diag else "",
                        "iteration": iteration + 1,
                        "operation": operation,
                        "invocation_attempt": attempt,
                        "invocation_max_attempts": max_attempts,
                        "retry_queue_remaining": remaining,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )

            try:
                response = await call_with_model_retry(
                    _invoke_model,
                    policy=policy,
                    on_failure=_on_invocation_failure,
                )
            except Exception as e:
                interruption_reason = (
                    "Model invocation failed after "
                    f"{invocation_attempt} attempt(s): {type(e).__name__}: {e}"
                )
                response = None
            else:
                response_diag = collect_response_diagnostics(response)
                self._emit_model_event(
                    task_id=task.id,
                    subtask_id=subtask.id,
                    model_name=model.name,
                    phase="done",
                    details={
                        "origin": request_diag.origin if request_diag else "",
                        "iteration": iteration + 1,
                        "operation": operation,
                        "invocation_attempt": invocation_attempt,
                        "invocation_max_attempts": policy.max_attempts,
                        **response_diag.to_event_payload(),
                    },
                )
                total_tokens += response.usage.total_tokens

            if interruption_reason:
                break
            if response is None:
                interruption_reason = (
                    "Execution ended before receiving a model response."
                )
                break

            if response.has_tool_calls():
                # Validate tool calls before execution
                validation = self._validator.validate_tool_calls(
                    response, self._tools.all_schemas()
                )
                if not validation.valid:
                    messages.append({
                        "role": "assistant",
                        "content": response.text or "",
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
                compact_tool_calls = await self._serialize_tool_calls_for_message(
                    response.tool_calls or []
                )
                messages.append({
                    "role": "assistant",
                    "content": response.text or "",
                    "tool_calls": compact_tool_calls,
                })

                for tc in response.tool_calls:
                    self._emit_tool_event(
                        TOOL_CALL_STARTED, task.id, subtask.id,
                        tc.name, tc.arguments,
                    )
                    policy_error = self._validate_deliverable_write_policy(
                        tool_name=tc.name,
                        tool_args=tc.arguments,
                        workspace=workspace,
                        expected_deliverables=canonical_deliverables,
                        enforce_deliverable_paths=enforce_deliverable_paths,
                        edit_existing_only=edit_existing_only,
                    )
                    if policy_error:
                        tool_result = ToolResult.fail(policy_error)
                    else:
                        tool_result = await self._tools.execute(
                            tc.name, tc.arguments,
                            workspace=workspace,
                            read_roots=read_roots,
                            changelog=changelog,
                            subtask_id=subtask.id,
                            auth_context=auth_context,
                        )
                    record = ToolCallRecord(
                        tool=tc.name,
                        args=tc.arguments,
                        result=tool_result,
                        call_id=str(getattr(tc, "id", "") or ""),
                    )
                    tool_calls_record.append(record)
                    new_evidence = extract_evidence_records(
                        task_id=task.id,
                        subtask_id=subtask.id,
                        tool_calls=[record],
                        existing_ids=known_evidence_ids,
                        context_text_max_chars=self._evidence_context_text_max_chars,
                    )
                    if new_evidence:
                        evidence_records_current.extend(new_evidence)
                        for item in new_evidence:
                            evidence_id = str(item.get("evidence_id", "")).strip()
                            if evidence_id:
                                known_evidence_ids.add(evidence_id)
                        data = tool_result.data if isinstance(tool_result.data, dict) else {}
                        data = dict(data)
                        data["evidence_ids"] = [
                            str(item.get("evidence_id", "")).strip()
                            for item in new_evidence
                            if str(item.get("evidence_id", "")).strip()
                        ]
                        if not tool_result.data:
                            tool_result.data = data
                        else:
                            tool_result.data = data
                    self._emit_tool_event(
                        TOOL_CALL_COMPLETED, task.id, subtask.id,
                        tc.name, tc.arguments,
                        result=tool_result,
                        workspace=workspace,
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": await self._serialize_tool_result_for_model(
                            tc.name, tool_result,
                        ),
                    })

                # Anti-amnesia reminder
                messages.append({
                    # Some OpenAI-compatible providers reject repeated in-thread
                    # system messages during tool-call loops.
                    "role": "user",
                    "content": self._build_todo_reminder(task, subtask),
                })
            else:
                # Text-only response — subtask complete
                completed_normally = True
                break
        else:
            if response is not None and response.has_tool_calls():
                budget_exhaustion_note = (
                    "Execution reached tool-iteration budget "
                    f"({iteration_budget} turns) while additional "
                    "tool calls were still required."
                )

        if interruption_reason is None and not completed_normally:
            if response is None:
                interruption_reason = (
                    "Execution ended before receiving a model response."
                )

        elapsed = time.monotonic() - start_time
        model_output = response.text if response and response.text else ""
        model_output_clean = self._strip_tool_call_placeholders(model_output)
        if interruption_reason:
            if model_output_clean:
                model_output = (
                    f"{interruption_reason} Last model response: {model_output_clean}"
                )
            else:
                model_output = interruption_reason
        else:
            model_output = model_output_clean
            if budget_exhaustion_note:
                if model_output_clean:
                    model_output = (
                        f"{budget_exhaustion_note} Last model response: "
                        f"{model_output_clean}"
                    )
                else:
                    model_output = budget_exhaustion_note
        summary = await self._summarize_model_output(
            model_output,
            max_chars=self._max_state_summary_chars,
            label="subtask state summary",
        )
        verification_summary = await self._summarize_model_output(
            model_output,
            max_chars=self._max_verification_summary_chars,
            label="subtask verification summary",
        )

        result = SubtaskResult(
            status=(
                SubtaskResultStatus.FAILED
                if interruption_reason
                else SubtaskResultStatus.SUCCESS
            ),
            summary=summary,
            tool_calls=tool_calls_record,
            duration_seconds=elapsed,
            tokens_used=total_tokens,
            model_used=model.name,
            evidence_records=evidence_records_current,
        )

        if interruption_reason:
            verification = VerificationResult(
                tier=1,
                passed=False,
                confidence=0.0,
                checks=[Check(
                    name="execution_completed",
                    passed=False,
                    detail=interruption_reason,
                )],
                feedback=interruption_reason,
            )
            self._spawn_memory_extraction(task.id, subtask.id, result)
            _COMPACTOR_EVENT_CONTEXT.reset(compactor_context_token)
            return result, verification

        # 4. Verification
        evidence_tool_calls = list(prior_successful_tool_calls or [])
        combined_evidence_records = merge_evidence_records(
            prior_evidence_records or [],
            evidence_records_current,
        )
        verification = await self._verification.verify(
            subtask=subtask,
            result_summary=verification_summary,
            tool_calls=tool_calls_record,
            evidence_tool_calls=evidence_tool_calls,
            evidence_records=combined_evidence_records,
            workspace=workspace,
            tier=subtask.verification_tier,
            task_id=task.id,
        )

        if not verification.passed:
            result.status = SubtaskResultStatus.FAILED

        # 5. Memory extraction — fire-and-forget
        self._spawn_memory_extraction(task.id, subtask.id, result)

        _COMPACTOR_EVENT_CONTEXT.reset(compactor_context_token)
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
        request_messages = [{"role": "user", "content": prompt}]

        try:
            request_diag = collect_request_diagnostics(
                messages=request_messages,
                origin="runner.extract_memory.complete",
            )
            self._emit_model_event(
                task_id=task_id,
                subtask_id=subtask_id,
                model_name=model.name,
                phase="start",
                details=request_diag.to_event_payload(),
            )
            policy = ModelRetryPolicy.from_execution_config(self._config.execution)
            response = await call_with_model_retry(
                lambda: model.complete(request_messages),
                policy=policy,
            )
            self._emit_model_event(
                task_id=task_id,
                subtask_id=subtask_id,
                model_name=model.name,
                phase="done",
                details={
                    "origin": request_diag.origin,
                    **collect_response_diagnostics(response).to_event_payload(),
                },
            )
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
                summary=str(e.get("summary", "")),
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
        workspace: Path | None = None,
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

        if (
            result is not None
            and not result.success
            and self._is_artifact_confinement_violation(result.error)
        ):
            violation_data: dict[str, object] = {
                "subtask_id": subtask_id,
                "tool": tool_name,
                "args": tool_args,
                "error": result.error or "",
            }
            if workspace is not None:
                violation_data["workspace"] = str(workspace)
            attempted_path = str(tool_args.get("path", "")).strip()
            if attempted_path:
                violation_data["attempted_path"] = attempted_path
            self._event_bus.emit(Event(
                event_type=ARTIFACT_CONFINEMENT_VIOLATION,
                task_id=task_id,
                data=violation_data,
            ))

    @staticmethod
    def _is_artifact_confinement_violation(error: str | None) -> bool:
        text = str(error or "").lower()
        return "safety violation" in text and "escapes workspace" in text

    def _emit_compactor_model_event(self, payload: dict) -> None:
        """Bridge semantic-compactor model events into task model_invocation events."""
        context = _COMPACTOR_EVENT_CONTEXT.get()
        if not context:
            return
        task_id, subtask_id = context
        model_name = str(payload.get("model", "")).strip() or "unknown"
        phase = str(payload.get("phase", "")).strip() or "done"
        details = {
            key: value
            for key, value in payload.items()
            if key not in {"model", "phase"}
        }
        self._emit_model_event(
            task_id=task_id,
            subtask_id=subtask_id,
            model_name=model_name,
            phase=phase,
            details=details,
        )

    def _emit_model_event(
        self,
        *,
        task_id: str,
        subtask_id: str,
        model_name: str,
        phase: str,
        details: dict | None = None,
    ) -> None:
        """Emit model invocation lifecycle to the event bus."""
        if not self._event_bus:
            return
        from loom.events.bus import Event

        data: dict = {
            "subtask_id": subtask_id,
            "model": model_name,
            "phase": phase,
        }
        if isinstance(details, dict) and details:
            data.update(details)
        self._event_bus.emit(Event(
            event_type=MODEL_INVOCATION,
            task_id=task_id,
            data=data,
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

    @classmethod
    def _tool_iteration_budget(
        cls,
        *,
        subtask: Subtask,
        retry_strategy: str,
        has_expected_deliverables: bool,
        base_budget: int | None = None,
    ) -> int:
        del subtask, retry_strategy, has_expected_deliverables  # configured globally
        budget = int(base_budget) if isinstance(base_budget, int) else cls.MAX_TOOL_ITERATIONS
        return max(1, min(200, budget))

    @staticmethod
    def _normalize_path_for_policy(path_text: str, workspace: Path | None) -> str:
        text = str(path_text or "").strip()
        if not text:
            return ""
        path = Path(text).expanduser()
        if path.is_absolute():
            if workspace is None:
                return path.as_posix().lstrip("./")
            try:
                return path.resolve().relative_to(workspace.resolve()).as_posix()
            except Exception:
                return path.as_posix().lstrip("./")
        parts = [part for part in path.parts if part not in {"", "."}]
        if workspace is not None and parts and parts[0] == workspace.name:
            parts = parts[1:]
        if not parts:
            return ""
        return Path(*parts).as_posix().lstrip("./")

    @classmethod
    def _normalize_deliverable_paths(
        cls,
        expected_deliverables: list[str],
        *,
        workspace: Path | None,
    ) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in expected_deliverables:
            value = cls._normalize_path_for_policy(str(item), workspace)
            if not value or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    @classmethod
    def _is_mutating_file_tool(cls, tool_name: str, tool_args: dict) -> bool:
        name = str(tool_name or "").strip().lower()
        if name not in cls._WRITE_MUTATING_TOOLS:
            return False
        if name != "spreadsheet":
            return True
        operation = str(tool_args.get("operation", "")).strip().lower()
        return operation in cls._SPREADSHEET_WRITE_OPERATIONS

    @classmethod
    def _target_paths_for_policy(
        cls,
        *,
        tool_name: str,
        tool_args: dict,
        workspace: Path | None,
    ) -> list[str]:
        if not cls._is_mutating_file_tool(tool_name, tool_args):
            return []
        keys = ("path", "destination", "source", "file", "file_path", "filepath")
        result: list[str] = []
        seen: set[str] = set()
        for key in keys:
            raw = tool_args.get(key)
            if raw is None:
                continue
            value = cls._normalize_path_for_policy(str(raw), workspace)
            if not value or value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    @classmethod
    def _looks_like_deliverable_variant(
        cls,
        *,
        candidate: str,
        canonical: str,
    ) -> bool:
        cand = Path(candidate)
        base = Path(canonical)
        if cand == base:
            return False
        if cand.parent != base.parent or cand.suffix.lower() != base.suffix.lower():
            return False
        cand_stem = cand.stem.lower()
        base_stem = base.stem.lower()
        if not cand_stem.startswith(base_stem):
            return False
        remainder = cand_stem[len(base_stem):]
        if not remainder or remainder[0] not in {"-", "_"}:
            return False
        tail = remainder[1:]
        for marker in cls._VARIANT_SUFFIX_MARKERS:
            if tail == marker:
                return True
            if tail.startswith(marker):
                suffix = tail[len(marker):]
                if not suffix:
                    return True
                if suffix.isdigit():
                    return True
                if suffix.startswith("-") and suffix[1:].isdigit():
                    return True
                if suffix.startswith("_") and suffix[1:].isdigit():
                    return True
        return bool(re.fullmatch(r"[a-z]{1,4}\d+", tail))

    @classmethod
    def _validate_deliverable_write_policy(
        cls,
        *,
        tool_name: str,
        tool_args: dict,
        workspace: Path | None,
        expected_deliverables: list[str],
        enforce_deliverable_paths: bool,
        edit_existing_only: bool,
    ) -> str | None:
        canonical = list(expected_deliverables)
        if not canonical:
            return None
        paths = cls._target_paths_for_policy(
            tool_name=tool_name,
            tool_args=tool_args,
            workspace=workspace,
        )
        if not paths:
            return None
        canonical_set = set(canonical)
        for path in paths:
            if path in canonical_set:
                continue
            if any(
                cls._looks_like_deliverable_variant(candidate=path, canonical=item)
                for item in canonical
            ):
                allowed = ", ".join(canonical)
                return (
                    "Canonical deliverable policy violation: "
                    f"'{path}' looks like a versioned copy of a required file. "
                    f"Update canonical file(s) instead: {allowed}."
                )
        if enforce_deliverable_paths:
            extras = [path for path in paths if path not in canonical_set]
            if extras:
                allowed = ", ".join(canonical)
                return (
                    "Canonical deliverable policy violation: "
                    "retry/remediation writes must stay in canonical deliverables. "
                    f"Unexpected target(s): {', '.join(extras)}. "
                    f"Allowed: {allowed}."
                )
        if (
            edit_existing_only
            and str(tool_name or "").strip().lower() in {"move_file", "delete_file"}
        ):
            return (
                "Remediation requires in-place edits to canonical deliverables. "
                "Do not rename or delete files during retry."
            )
        return None

    def _tool_output_limit(self, tool_name: str) -> int:
        heavy_limit = int(
            getattr(
                self,
                "_heavy_tool_result_output_chars",
                self.HEAVY_TOOL_RESULT_OUTPUT_CHARS,
            ),
        )
        default_limit = int(
            getattr(
                self,
                "_default_tool_result_output_chars",
                self.DEFAULT_TOOL_RESULT_OUTPUT_CHARS,
            ),
        )
        if tool_name in self._HEAVY_OUTPUT_TOOLS:
            return heavy_limit
        return default_limit

    @staticmethod
    def _hard_cap_text(text: str, max_chars: int) -> str:
        """Deterministically bound text length when semantic compaction misses."""
        value = str(text or "")
        if max_chars <= 0:
            return ""
        if len(value) <= max_chars:
            return value
        if max_chars <= 40:
            return value[:max_chars]

        marker = "...[truncated]..."
        remaining = max_chars - len(marker)
        if remaining <= 0:
            return value[:max_chars]

        head = max(16, int(remaining * 0.65))
        tail = max(8, remaining - head)
        if head + tail > remaining:
            tail = max(0, remaining - head)
        compacted = f"{value[:head]}{marker}{value[-tail:] if tail else ''}"
        return compacted[:max_chars]

    async def _compact_text(self, text: str, *, max_chars: int, label: str) -> str:
        compacted = await self._compactor.compact(
            str(text or ""),
            max_chars=max_chars,
            label=label,
        )
        if len(compacted) > max_chars:
            logger.warning(
                "Compaction exceeded budget for %s: got %d chars (limit %d)",
                label,
                len(compacted),
                max_chars,
            )
        return compacted

    async def _summarize_model_output(
        self,
        output: str,
        *,
        max_chars: int,
        label: str,
    ) -> str:
        """Produce a bounded summary via semantic compaction."""
        text = self._strip_tool_call_placeholders(output)
        if not text:
            return "No output"
        if len(text) <= max_chars:
            return text
        return await self._compact_text(text, max_chars=max_chars, label=label)

    @classmethod
    def _strip_tool_call_placeholders(cls, output: str) -> str:
        """Remove provider/compaction placeholder lines from model text."""
        text = str(output or "")
        if not text.strip():
            return ""
        kept: list[str] = []
        for line in text.splitlines():
            if line.strip().lower() in cls._TOOL_CALL_CONTEXT_PLACEHOLDERS:
                continue
            kept.append(line)
        return "\n".join(kept).strip()

    async def _summarize_tool_data(self, data: dict | None) -> dict | None:
        if not isinstance(data, dict) or not data:
            return None
        compact_tool_arg_chars = int(
            getattr(
                self,
                "_compact_tool_call_argument_chars",
                self.COMPACT_TOOL_CALL_ARGUMENT_CHARS,
            ),
        )
        # Keep scalar fields slightly tighter than nested objects/lists.
        compact_tool_data_text_chars = max(
            80,
            int(round(compact_tool_arg_chars * 0.75)),
        )
        if len(data) > 12:
            packed = json.dumps(data, ensure_ascii=False, default=str)
            summary_text = await self._compact_text(
                packed,
                max_chars=900,
                label="tool data payload",
            )
            return {
                "summary": summary_text,
                "key_count": len(data),
            }
        summary: dict = {}
        for key, value in data.items():
            if isinstance(value, str):
                summary[key] = await self._compact_text(
                    value,
                    max_chars=compact_tool_data_text_chars,
                    label=f"tool data {key}",
                )
            elif isinstance(value, (int, float, bool)) or value is None:
                summary[key] = value
            elif isinstance(value, dict):
                packed = json.dumps(value, ensure_ascii=False, default=str)
                summary[key] = await self._compact_text(
                    packed,
                    max_chars=compact_tool_arg_chars,
                    label=f"tool data object {key}",
                )
            elif isinstance(value, list):
                packed = json.dumps(value, ensure_ascii=False, default=str)
                summary[key] = await self._compact_text(
                    packed,
                    max_chars=compact_tool_arg_chars,
                    label=f"tool data list {key}",
                )
            else:
                summary[key] = str(type(value).__name__)
        return summary or None

    async def _summarize_tool_call_arguments(
        self,
        args: object,
        *,
        max_chars: int,
        label: str,
    ) -> dict:
        if isinstance(args, dict):
            summary = await self._summarize_tool_data(args) or {}
            packed = json.dumps(summary, ensure_ascii=False, default=str)
            if len(packed) <= max_chars:
                return summary
            compacted = await self._compact_text(
                packed,
                max_chars=max_chars,
                label=label,
            )
            return {
                "summary": compacted,
                "key_count": len(args),
            }

        packed = json.dumps(args, ensure_ascii=False, default=str)
        compacted = await self._compact_text(
            packed,
            max_chars=max_chars,
            label=label,
        )
        return {"summary": compacted}

    async def _serialize_tool_calls_for_message(self, tool_calls: list) -> list[dict]:
        arg_limit = int(
            getattr(
                self,
                "_tool_call_argument_context_chars",
                self.TOOL_CALL_ARGUMENT_CONTEXT_CHARS,
            ),
        )
        serialized: list[dict] = []
        for tc in tool_calls:
            name = str(getattr(tc, "name", "") or "tool")
            args = getattr(tc, "arguments", {})
            compact_args = await self._summarize_tool_call_arguments(
                args,
                max_chars=arg_limit,
                label=f"{name} tool call args",
            )
            serialized.append({
                "id": str(getattr(tc, "id", "") or ""),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(compact_args, ensure_ascii=False, default=str),
                },
            })
        return serialized

    async def _compact_assistant_tool_calls(
        self,
        tool_calls: object,
        *,
        max_chars: int,
    ) -> list[dict] | None:
        if not isinstance(tool_calls, list):
            return None

        compacted_calls: list[dict] = []
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            compact_item = dict(item)
            function = compact_item.get("function")
            if not isinstance(function, dict):
                compacted_calls.append(compact_item)
                continue

            function_copy = dict(function)
            name = str(function_copy.get("name", "") or "tool")
            raw_args = function_copy.get("arguments", "{}")
            parsed_args: object
            if isinstance(raw_args, dict):
                parsed_args = raw_args
            elif isinstance(raw_args, str):
                try:
                    decoded = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    parsed_args = {"raw": raw_args}
                else:
                    parsed_args = decoded
            else:
                parsed_args = {"raw": str(raw_args)}

            compact_args = await self._summarize_tool_call_arguments(
                parsed_args,
                max_chars=max_chars,
                label=f"{name} assistant tool-call args",
            )
            function_copy["arguments"] = json.dumps(
                compact_args,
                ensure_ascii=False,
                default=str,
            )
            compact_item["function"] = function_copy
            compacted_calls.append(compact_item)

        return compacted_calls or None

    async def _serialize_content_blocks_for_model(
        self,
        blocks: list | None,
        *,
        max_chars: int,
    ) -> list[dict] | None:
        if not blocks:
            return None
        from loom.content import serialize_block

        serialized_blocks: list[dict] = []
        for block in blocks:
            try:
                payload = serialize_block(block)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            compact = dict(payload)
            for key in ("text", "text_fallback", "extracted_text", "thinking"):
                value = compact.get(key)
                if isinstance(value, str):
                    compact[key] = await self._compact_text(
                        value,
                        max_chars=max_chars,
                        label=f"content block {key}",
                    )
            serialized_blocks.append(compact)
        return serialized_blocks or None

    async def _serialize_tool_result_for_model(
        self,
        tool_name: str,
        result: ToolResult,
        *,
        max_output_chars: int | None = None,
    ) -> str:
        limit = max_output_chars or self._tool_output_limit(tool_name)
        output_text = await self._compact_text(
            result.output,
            max_chars=limit,
            label=f"{tool_name} tool output",
        )
        payload: dict = {
            "success": result.success,
            "output": output_text,
            "error": result.error,
            "files_changed": list(result.files_changed),
        }
        if len(payload["files_changed"]) > 20:
            files_text = "\n".join(payload["files_changed"])
            payload["files_changed_summary"] = await self._compact_text(
                files_text,
                max_chars=380,
                label=f"{tool_name} files changed",
            )
            payload["files_changed_count"] = len(payload["files_changed"])
            payload.pop("files_changed", None)

        data_summary = await self._summarize_tool_data(result.data)
        if data_summary:
            payload["data"] = data_summary
        blocks = await self._serialize_content_blocks_for_model(
            result.content_blocks,
            max_chars=min(limit, 400),
        )
        if blocks:
            payload["content_blocks"] = blocks
        return json.dumps(payload)

    async def _compact_tool_message_content(
        self,
        content: str,
        *,
        max_output_chars: int,
    ) -> str:
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return await self._compact_text(
                str(content),
                max_chars=max_output_chars,
                label="tool message content",
            )

        if not isinstance(parsed, dict):
            return await self._compact_text(
                str(content),
                max_chars=max_output_chars,
                label="tool message payload",
            )

        output = await self._compact_text(
            str(parsed.get("output", "")),
            max_chars=max_output_chars,
            label="tool message output",
        )
        payload: dict = {
            "success": bool(parsed.get("success", False)),
            "output": output,
            "error": parsed.get("error"),
            "files_changed": list(parsed.get("files_changed", [])),
        }
        if len(payload["files_changed"]) > 20:
            files_text = "\n".join(str(item) for item in payload["files_changed"])
            payload["files_changed_summary"] = await self._compact_text(
                files_text,
                max_chars=320,
                label="tool message files changed",
            )
            payload["files_changed_count"] = len(payload["files_changed"])
            payload.pop("files_changed", None)

        data_summary = await self._summarize_tool_data(parsed.get("data"))
        if data_summary:
            payload["data"] = data_summary

        raw_blocks = parsed.get("content_blocks")
        if isinstance(raw_blocks, list) and raw_blocks:
            compact_blocks: list[dict] = []
            for block in raw_blocks:
                if not isinstance(block, dict):
                    continue
                compact = dict(block)
                for key in ("text", "text_fallback", "extracted_text", "thinking"):
                    value = compact.get(key)
                    if isinstance(value, str):
                        compact[key] = await self._compact_text(
                            value,
                            max_chars=min(max_output_chars, 400),
                            label=f"tool block {key}",
                        )
                compact_blocks.append(compact)
            if compact_blocks:
                payload["content_blocks"] = compact_blocks
        return json.dumps(payload)

    @staticmethod
    def _estimate_message_tokens(messages: list[dict]) -> int:
        total = 0
        for message in messages:
            try:
                total += estimate_tokens(json.dumps(message))
            except (TypeError, ValueError):
                total += estimate_tokens(str(message))
        return total

    async def _compact_messages_for_model(self, messages: list[dict]) -> list[dict]:
        """Compact older messages to keep tool loops within context budget."""
        if len(messages) < 3:
            return messages

        context_budget = int(
            getattr(self, "_max_model_context_tokens", self.MAX_MODEL_CONTEXT_TOKENS),
        )
        compact_tool_call_args = int(
            getattr(
                self,
                "_compact_tool_call_argument_chars",
                self.COMPACT_TOOL_CALL_ARGUMENT_CHARS,
            ),
        )
        compact_tool_output = int(
            getattr(
                self,
                "_compact_tool_result_output_chars",
                self.COMPACT_TOOL_RESULT_OUTPUT_CHARS,
            ),
        )
        compact_text_chars = int(
            getattr(self, "_compact_text_output_chars", self.COMPACT_TEXT_OUTPUT_CHARS),
        )
        minimal_text_chars = int(
            getattr(self, "_minimal_text_output_chars", self.MINIMAL_TEXT_OUTPUT_CHARS),
        )

        if self._estimate_message_tokens(messages) <= context_budget:
            return messages

        compacted: list[dict] = [
            dict(message) if isinstance(message, dict) else message
            for message in messages
        ]

        # Always compact assistant tool-call argument payloads first.
        # These can be extremely large (e.g., document_write content) and can
        # exceed backend request byte caps even in short recent histories.
        for msg in compacted:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip().lower()
            if role != "assistant" or not msg.get("tool_calls"):
                continue
            compact_calls = await self._compact_assistant_tool_calls(
                msg.get("tool_calls"),
                max_chars=compact_tool_call_args,
            )
            if compact_calls is not None:
                msg["tool_calls"] = compact_calls

        async def _apply_pass(
            *,
            preserve_recent: int,
            tool_chars: int,
            text_chars: int,
        ) -> None:
            preserve_from = max(1, len(compacted) - preserve_recent)
            for idx, msg in enumerate(compacted):
                if idx == 0 or idx >= preserve_from or not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "")).strip().lower()
                if role == "tool":
                    msg["content"] = await self._compact_tool_message_content(
                        msg.get("content", ""),
                        max_output_chars=tool_chars,
                    )
                elif role == "assistant":
                    if msg.get("tool_calls"):
                        msg["content"] = self.TOOL_CALL_CONTEXT_PLACEHOLDER
                        compact_calls = await self._compact_assistant_tool_calls(
                            msg.get("tool_calls"),
                            max_chars=max(80, min(text_chars, compact_tool_call_args)),
                        )
                        if compact_calls is not None:
                            msg["tool_calls"] = compact_calls
                    else:
                        content = msg.get("content")
                        if isinstance(content, str) and len(content) > text_chars:
                            msg["content"] = await self._compact_text(
                                content,
                                max_chars=text_chars,
                                label="assistant context",
                            )
                elif role == "user":
                    content = msg.get("content")
                    if isinstance(content, str):
                        if content.startswith(self._TODO_REMINDER_PREFIX):
                            msg["content"] = (
                                "Continue current subtask only. "
                                "Do NOT move to the next subtask."
                            )
                        elif len(content) > text_chars:
                            msg["content"] = await self._compact_text(
                                content,
                                max_chars=text_chars,
                                label="user context",
                            )
                elif role == "system":
                    content = msg.get("content")
                    if isinstance(content, str) and len(content) > text_chars:
                        msg["content"] = await self._compact_text(
                            content,
                            max_chars=text_chars,
                            label="system context",
                        )

        await _apply_pass(
            preserve_recent=8,
            tool_chars=compact_tool_output,
            text_chars=compact_text_chars,
        )
        if self._estimate_message_tokens(compacted) <= context_budget:
            return compacted

        await _apply_pass(
            preserve_recent=4,
            tool_chars=120,
            text_chars=minimal_text_chars,
        )
        if self._estimate_message_tokens(compacted) <= context_budget:
            return compacted

        # Final semantic merge pass: replace old context with one compact brief.
        preserve_from = max(1, len(compacted) - 3)
        old_context = compacted[1:preserve_from]
        recent = compacted[preserve_from:]
        while recent and isinstance(recent[0], dict) and recent[0].get("role") == "tool":
            recent = recent[1:]

        if old_context:
            merged_lines: list[str] = []
            for msg in old_context:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "")).strip().lower() or "unknown"
                content = msg.get("content", "")
                if role == "assistant" and msg.get("tool_calls"):
                    tool_names = [
                        str(tc.get("function", {}).get("name", "tool"))
                        for tc in list(msg.get("tool_calls", []))
                        if isinstance(tc, dict)
                    ]
                    merged_lines.append(
                        f"[assistant/tool_call] {', '.join(tool_names) or 'tool call'}"
                    )
                    continue
                if not isinstance(content, str):
                    content = str(content)
                merged_lines.append(f"[{role}] {content}")

            merged_text = "\n".join(merged_lines).strip()
            if merged_text:
                merged_summary = await self._compact_text(
                    merged_text,
                    max_chars=700,
                    label="prior conversation context",
                )
                compacted = [
                    compacted[0],
                    {"role": "user", "content": f"Prior compacted context:\n{merged_summary}"},
                    *recent,
                ]
                if (
                    self._estimate_message_tokens(compacted)
                    <= context_budget
                ):
                    return compacted

        return compacted
