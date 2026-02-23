"""Subtask runner: encapsulates single-subtask execution.

Owns the tool-calling loop, response validation, verification gates,
and memory extraction for one subtask.  Returns compact structured
results so the orchestrator never touches raw prompts or messages.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

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


class CompactionClass(StrEnum):
    CRITICAL = "critical"
    TOOL_TRACE = "tool_trace"
    HISTORICAL_CONTEXT = "historical_context"
    BACKGROUND_EXTRACTION = "background_extraction"


class CompactionPressureTier(StrEnum):
    NORMAL = "normal"
    PRESSURE = "pressure"
    CRITICAL = "critical"


@dataclass(frozen=True)
class _CompactionPlan:
    critical_indices: tuple[int, ...]
    stage1_tool_args: tuple[int, ...]
    stage2_tool_output: tuple[int, ...]
    stage3_historical: tuple[int, ...]
    stage4_merge: tuple[int, ...]


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
    RUNNER_COMPACTION_POLICY_MODE = "tiered"
    PRESERVE_RECENT_CRITICAL_MESSAGES = 6
    COMPACTION_PRESSURE_RATIO_SOFT = 0.86
    COMPACTION_PRESSURE_RATIO_HARD = 1.02
    COMPACTION_NO_GAIN_MIN_DELTA_CHARS = 24
    COMPACTION_NO_GAIN_ATTEMPT_LIMIT = 2
    COMPACTION_TIMEOUT_GUARD_SECONDS = 30
    EXTRACTOR_TIMEOUT_GUARD_SECONDS = 20
    EXTRACTOR_TOOL_ARGS_MAX_CHARS = 260
    EXTRACTOR_TOOL_TRACE_MAX_CHARS = 3600
    EXTRACTOR_PROMPT_MAX_CHARS = 9000
    COMPACTION_CHURN_WARNING_CALLS = 10
    ENABLE_FILETYPE_INGEST_ROUTER = True
    ENABLE_MODEL_OVERFLOW_FALLBACK = True
    OVERFLOW_FALLBACK_TOOL_MESSAGE_MIN_CHARS = 4_000
    OVERFLOW_FALLBACK_TOOL_OUTPUT_EXCERPT_CHARS = 1_200
    _OVERFLOW_BINARY_CONTENT_KINDS = frozenset({
        "pdf",
        "office_doc",
        "image",
        "archive",
        "unknown_binary",
    })
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
        policy_mode = str(
            getattr(
                runner_limits,
                "runner_compaction_policy_mode",
                self.RUNNER_COMPACTION_POLICY_MODE,
            ),
        ).strip().lower()
        self._runner_compaction_policy_mode = (
            policy_mode
            if policy_mode in {"legacy", "tiered", "off"}
            else self.RUNNER_COMPACTION_POLICY_MODE
        )
        self._preserve_recent_critical_messages = max(
            2,
            int(
                getattr(
                    runner_limits,
                    "preserve_recent_critical_messages",
                    self.PRESERVE_RECENT_CRITICAL_MESSAGES,
                ),
            ),
        )
        self._compaction_pressure_ratio_soft = max(
            0.4,
            float(
                getattr(
                    runner_limits,
                    "compaction_pressure_ratio_soft",
                    self.COMPACTION_PRESSURE_RATIO_SOFT,
                ),
            ),
        )
        hard_ratio = float(
            getattr(
                runner_limits,
                "compaction_pressure_ratio_hard",
                self.COMPACTION_PRESSURE_RATIO_HARD,
            ),
        )
        self._compaction_pressure_ratio_hard = max(
            self._compaction_pressure_ratio_soft + 0.01,
            hard_ratio,
        )
        self._compaction_no_gain_min_delta_chars = max(
            1,
            int(
                getattr(
                    runner_limits,
                    "compaction_no_gain_min_delta_chars",
                    self.COMPACTION_NO_GAIN_MIN_DELTA_CHARS,
                ),
            ),
        )
        self._compaction_no_gain_attempt_limit = max(
            1,
            int(
                getattr(
                    runner_limits,
                    "compaction_no_gain_attempt_limit",
                    self.COMPACTION_NO_GAIN_ATTEMPT_LIMIT,
                ),
            ),
        )
        self._compaction_timeout_guard_seconds = max(
            0.0,
            float(
                getattr(
                    runner_limits,
                    "compaction_timeout_guard_seconds",
                    self.COMPACTION_TIMEOUT_GUARD_SECONDS,
                ),
            ),
        )
        self._extractor_timeout_guard_seconds = max(
            0.0,
            float(
                getattr(
                    runner_limits,
                    "extractor_timeout_guard_seconds",
                    self.EXTRACTOR_TIMEOUT_GUARD_SECONDS,
                ),
            ),
        )
        self._extractor_tool_args_max_chars = max(
            80,
            int(
                getattr(
                    runner_limits,
                    "extractor_tool_args_max_chars",
                    self.EXTRACTOR_TOOL_ARGS_MAX_CHARS,
                ),
            ),
        )
        self._extractor_tool_trace_max_chars = max(
            300,
            int(
                getattr(
                    runner_limits,
                    "extractor_tool_trace_max_chars",
                    self.EXTRACTOR_TOOL_TRACE_MAX_CHARS,
                ),
            ),
        )
        self._extractor_prompt_max_chars = max(
            600,
            int(
                getattr(
                    runner_limits,
                    "extractor_prompt_max_chars",
                    self.EXTRACTOR_PROMPT_MAX_CHARS,
                ),
            ),
        )
        self._compaction_churn_warning_calls = max(
            1,
            int(
                getattr(
                    runner_limits,
                    "compaction_churn_warning_calls",
                    self.COMPACTION_CHURN_WARNING_CALLS,
                ),
            ),
        )
        self._enable_filetype_ingest_router = bool(
            getattr(
                runner_limits,
                "enable_filetype_ingest_router",
                self.ENABLE_FILETYPE_INGEST_ROUTER,
            ),
        )
        self._enable_model_overflow_fallback = bool(
            getattr(
                runner_limits,
                "enable_model_overflow_fallback",
                self.ENABLE_MODEL_OVERFLOW_FALLBACK,
            ),
        )
        self._ingest_artifact_retention_max_age_days = max(
            0,
            int(
                getattr(
                    runner_limits,
                    "ingest_artifact_retention_max_age_days",
                    14,
                ),
            ),
        )
        self._ingest_artifact_retention_max_files_per_scope = max(
            1,
            int(
                getattr(
                    runner_limits,
                    "ingest_artifact_retention_max_files_per_scope",
                    96,
                ),
            ),
        )
        self._ingest_artifact_retention_max_bytes_per_scope = max(
            1024,
            int(
                getattr(
                    runner_limits,
                    "ingest_artifact_retention_max_bytes_per_scope",
                    268_435_456,
                ),
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
        self._subtask_deadline_monotonic: float | None = None
        self._last_compaction_diagnostics: dict[str, Any] = {}
        self._runner_compaction_cache: dict[tuple[str, int, str], str] = {}
        self._runner_compaction_no_gain: dict[tuple[str, int, str], int] = {}
        self._runner_compaction_overshoot: set[tuple[str, int, str]] = set()
        self._compaction_runtime_stats: dict[str, Any] = {
            "compactor_calls": 0,
            "skip_reasons": {},
        }

    def _reset_compaction_runtime_stats(self) -> None:
        self._compaction_runtime_stats = {
            "compactor_calls": 0,
            "skip_reasons": {},
        }

    def _record_compaction_skip(self, reason: str) -> None:
        if not reason:
            return
        stats = getattr(self, "_compaction_runtime_stats", None)
        if not isinstance(stats, dict):
            return
        skip_reasons = stats.setdefault("skip_reasons", {})
        if isinstance(skip_reasons, dict):
            skip_reasons[reason] = int(skip_reasons.get(reason, 0)) + 1

    def _record_compactor_call(self) -> None:
        stats = getattr(self, "_compaction_runtime_stats", None)
        if not isinstance(stats, dict):
            return
        stats["compactor_calls"] = int(stats.get("compactor_calls", 0)) + 1

    def _ensure_runner_compaction_state(self) -> None:
        if not isinstance(getattr(self, "_runner_compaction_cache", None), dict):
            self._runner_compaction_cache = {}
        if not isinstance(getattr(self, "_runner_compaction_no_gain", None), dict):
            self._runner_compaction_no_gain = {}
        if not isinstance(getattr(self, "_runner_compaction_overshoot", None), set):
            self._runner_compaction_overshoot = set()
        if not isinstance(getattr(self, "_compaction_runtime_stats", None), dict):
            self._reset_compaction_runtime_stats()

    @staticmethod
    def _compaction_cache_key(text: str, *, max_chars: int, label: str) -> tuple[str, int, str]:
        digest = hashlib.sha1(
            str(text or "").encode("utf-8", errors="ignore"),
        ).hexdigest()
        return (digest, int(max_chars), str(label or "context"))

    @staticmethod
    def _trim_compaction_cache(cache: dict, *, max_entries: int = 512) -> None:
        if len(cache) <= max_entries:
            return
        overflow = len(cache) - max_entries
        for key in list(cache.keys())[:overflow]:
            cache.pop(key, None)

    def _remaining_subtask_seconds(self) -> float:
        deadline = getattr(self, "_subtask_deadline_monotonic", None)
        if not isinstance(deadline, (float, int)) or deadline <= 0:
            return float(
                getattr(
                    self,
                    "_max_subtask_wall_clock_seconds",
                    self.MAX_SUBTASK_WALL_CLOCK,
                ),
            )
        return max(0.0, float(deadline) - time.monotonic())

    def _is_timeout_guard_active(self, remaining_seconds: float | None = None) -> bool:
        remaining = (
            float(remaining_seconds)
            if isinstance(remaining_seconds, (float, int))
            else self._remaining_subtask_seconds()
        )
        guard = float(
            getattr(
                self,
                "_compaction_timeout_guard_seconds",
                self.COMPACTION_TIMEOUT_GUARD_SECONDS,
            ),
        )
        return remaining <= max(0.0, guard)

    def _runner_compaction_mode(self) -> str:
        mode = str(
            getattr(
                self,
                "_runner_compaction_policy_mode",
                self.RUNNER_COMPACTION_POLICY_MODE,
            ),
        ).strip().lower()
        return (
            mode
            if mode in {"legacy", "tiered", "off"}
            else self.RUNNER_COMPACTION_POLICY_MODE
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
        self._subtask_deadline_monotonic = (
            start_time + self._max_subtask_wall_clock_seconds
        )
        self._reset_compaction_runtime_stats()
        self._last_compaction_diagnostics = {
            "compaction_policy_mode": str(
                getattr(
                    self,
                    "_runner_compaction_policy_mode",
                    self.RUNNER_COMPACTION_POLICY_MODE,
                ),
            ),
            "compaction_stage": "none",
            "compaction_candidate_count": 0,
            "compaction_skipped_reason": "not_started",
        }
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
            self._subtask_deadline_monotonic = None
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
            remaining_seconds = self._remaining_subtask_seconds()
            if remaining_seconds <= 0:
                interruption_reason = (
                    "Execution exceeded subtask time budget "
                    f"({self._max_subtask_wall_clock_seconds}s) before completion."
                )
                break
            messages = await self._compact_messages_for_model(
                messages,
                remaining_seconds=remaining_seconds,
            )
            tool_schemas = self._tools.all_schemas()
            operation = "stream" if streaming else "complete"
            response = None
            policy = ModelRetryPolicy.from_execution_config(self._config.execution)
            invocation_attempt = 0
            request_diag = None
            overflow_fallback_pending = False
            overflow_fallback_attempted = False
            overflow_fallback_report: dict[str, Any] | None = None

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
                        "remaining_subtask_seconds": round(
                            self._remaining_subtask_seconds(),
                            3,
                        ),
                        **dict(getattr(self, "_last_compaction_diagnostics", {})),
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

            def _should_retry_invocation(error: BaseException) -> bool:
                nonlocal overflow_fallback_pending
                if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                    return False
                if self._is_model_request_overflow_error(error):
                    if self._enable_model_overflow_fallback and not overflow_fallback_attempted:
                        overflow_fallback_pending = True
                        return True
                    return False
                return True

            def _on_invocation_failure(
                attempt: int,
                max_attempts: int,
                error: BaseException,
                remaining: int,
            ) -> None:
                nonlocal messages, overflow_fallback_pending
                nonlocal overflow_fallback_attempted, overflow_fallback_report
                if overflow_fallback_pending:
                    overflow_fallback_pending = False
                    overflow_fallback_attempted = True
                    messages, overflow_fallback_report = self._apply_model_overflow_fallback(
                        messages,
                    )
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
                        "overflow_error_detected": self._is_model_request_overflow_error(error),
                        "overflow_fallback_attempted": overflow_fallback_attempted,
                        **(overflow_fallback_report or {}),
                    },
                )

            try:
                response = await call_with_model_retry(
                    _invoke_model,
                    policy=policy,
                    should_retry=_should_retry_invocation,
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
                        execute_args = dict(tc.arguments)
                        if tc.name in {"web_fetch", "web_fetch_html"}:
                            execute_args["_enable_filetype_ingest_router"] = bool(
                                self._enable_filetype_ingest_router,
                            )
                            execute_args["_artifact_retention_max_age_days"] = int(
                                self._ingest_artifact_retention_max_age_days,
                            )
                            execute_args["_artifact_retention_max_files_per_scope"] = int(
                                self._ingest_artifact_retention_max_files_per_scope,
                            )
                            execute_args["_artifact_retention_max_bytes_per_scope"] = int(
                                self._ingest_artifact_retention_max_bytes_per_scope,
                            )
                        tool_result = await self._tools.execute(
                            tc.name, execute_args,
                            workspace=workspace,
                            read_roots=read_roots,
                            scratch_dir=self._config.scratch_path,
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
            self._subtask_deadline_monotonic = None
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

        self._subtask_deadline_monotonic = None
        _COMPACTOR_EVENT_CONTEXT.reset(compactor_context_token)
        return result, verification

    def _spawn_memory_extraction(
        self, task_id: str, subtask_id: str, result: SubtaskResult,
    ) -> None:
        """Schedule memory extraction as a background task.

        Does not block the caller.  Failures are silently ignored
        (memory extraction is best-effort).
        """
        remaining_seconds = self._remaining_subtask_seconds()
        extractor_guard = float(
            getattr(
                self,
                "_extractor_timeout_guard_seconds",
                self.EXTRACTOR_TIMEOUT_GUARD_SECONDS,
            ),
        )
        if remaining_seconds <= extractor_guard:
            logger.debug(
                "Memory extraction skipped for %s: timeout guard active "
                "(remaining=%.2fs, guard=%.2fs)",
                subtask_id,
                remaining_seconds,
                extractor_guard,
            )
            return
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

        compacted_fields: set[str] = set()
        extractor_tool_args_max = int(
            getattr(
                self,
                "_extractor_tool_args_max_chars",
                self.EXTRACTOR_TOOL_ARGS_MAX_CHARS,
            ),
        )
        extractor_tool_trace_max = int(
            getattr(
                self,
                "_extractor_tool_trace_max_chars",
                self.EXTRACTOR_TOOL_TRACE_MAX_CHARS,
            ),
        )
        extractor_prompt_max = int(
            getattr(
                self,
                "_extractor_prompt_max_chars",
                self.EXTRACTOR_PROMPT_MAX_CHARS,
            ),
        )

        tool_lines = []
        for tc in result.tool_calls:
            status = "OK" if tc.result.success else f"FAILED: {tc.result.error}"
            raw_args_text = json.dumps(tc.args, ensure_ascii=False, default=str)
            compact_args = await self._summarize_tool_call_arguments(
                tc.args,
                max_chars=extractor_tool_args_max,
                label=f"{tc.tool} extractor args",
            )
            compact_args_text = json.dumps(
                compact_args,
                ensure_ascii=False,
                default=str,
            )
            if compact_args_text != raw_args_text:
                compacted_fields.add("tool_args")
            line = f"- {tc.tool}({compact_args_text}) → {status}"
            # Note multimodal content in the tool result
            if tc.result.content_blocks:
                block_types = [getattr(b, "type", "?") for b in tc.result.content_blocks]
                line += f" [content: {', '.join(block_types)}]"
            tool_lines.append(line)
        tool_calls_formatted = "\n".join(tool_lines) if tool_lines else "No tool calls."
        if len(tool_calls_formatted) > extractor_tool_trace_max:
            tool_calls_formatted = await self._compact_text(
                tool_calls_formatted,
                max_chars=extractor_tool_trace_max,
                label="memory extractor tool trace",
            )
            compacted_fields.add("tool_trace")

        model_output = str(result.summary or "")
        prompt = self._prompts.build_extractor_prompt(
            subtask_id=subtask_id,
            tool_calls_formatted=tool_calls_formatted,
            model_output=model_output,
        )
        if len(prompt) > extractor_prompt_max:
            output_budget = max(
                220,
                min(int(extractor_prompt_max * 0.35), len(model_output)),
            )
            if len(model_output) > output_budget:
                model_output = await self._compact_text(
                    model_output,
                    max_chars=output_budget,
                    label="memory extractor model output",
                )
                compacted_fields.add("model_output")
            prompt = self._prompts.build_extractor_prompt(
                subtask_id=subtask_id,
                tool_calls_formatted=tool_calls_formatted,
                model_output=model_output,
            )
        if len(prompt) > extractor_prompt_max:
            tightened_trace_budget = max(
                220,
                min(
                    int(extractor_prompt_max * 0.45),
                    max(220, extractor_tool_trace_max // 2),
                ),
            )
            tool_calls_formatted = await self._compact_text(
                tool_calls_formatted,
                max_chars=tightened_trace_budget,
                label="memory extractor tool trace strict",
            )
            compacted_fields.add("tool_trace")
            prompt = self._prompts.build_extractor_prompt(
                subtask_id=subtask_id,
                tool_calls_formatted=tool_calls_formatted,
                model_output=model_output,
            )

        request_messages = [{"role": "user", "content": prompt}]
        extractor_prompt_est_tokens = self._estimate_message_tokens(request_messages)

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
                details={
                    **request_diag.to_event_payload(),
                    "extractor_prompt_chars": len(prompt),
                    "extractor_prompt_est_tokens": extractor_prompt_est_tokens,
                    "extractor_compacted_fields": sorted(compacted_fields),
                },
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
                    "extractor_prompt_chars": len(prompt),
                    "extractor_prompt_est_tokens": extractor_prompt_est_tokens,
                    "extractor_compacted_fields": sorted(compacted_fields),
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
        value = str(text or "")
        if self._runner_compaction_mode() == "off":
            self._record_compaction_skip("policy_disabled")
            return value
        if max_chars <= 0:
            return ""
        if len(value) <= max_chars:
            return value

        self._ensure_runner_compaction_state()
        key = self._compaction_cache_key(value, max_chars=max_chars, label=label)
        cached = self._runner_compaction_cache.get(key)
        if cached is not None:
            self._record_compaction_skip("cache_hit")
            return cached

        if key in self._runner_compaction_overshoot:
            self._record_compaction_skip("no_gain")
            return value

        no_gain_attempt_limit = int(
            getattr(
                self,
                "_compaction_no_gain_attempt_limit",
                self.COMPACTION_NO_GAIN_ATTEMPT_LIMIT,
            ),
        )
        no_gain_attempts = int(self._runner_compaction_no_gain.get(key, 0))
        if no_gain_attempts >= no_gain_attempt_limit:
            self._record_compaction_skip("no_gain")
            return value

        compacted = await self._compactor.compact(
            value,
            max_chars=max_chars,
            label=label,
        )
        self._record_compactor_call()
        self._runner_compaction_cache[key] = compacted
        self._trim_compaction_cache(self._runner_compaction_cache)

        min_delta = int(
            getattr(
                self,
                "_compaction_no_gain_min_delta_chars",
                self.COMPACTION_NO_GAIN_MIN_DELTA_CHARS,
            ),
        )
        reduction_delta = max(0, len(value) - len(compacted))
        if reduction_delta < max(1, min_delta):
            self._runner_compaction_no_gain[key] = no_gain_attempts + 1
            self._trim_compaction_cache(self._runner_compaction_no_gain)
        else:
            self._runner_compaction_no_gain.pop(key, None)

        if len(compacted) > max_chars:
            logger.warning(
                "Compaction exceeded budget for %s: got %d chars (limit %d)",
                label,
                len(compacted),
                max_chars,
            )
            self._runner_compaction_overshoot.add(key)
            if len(self._runner_compaction_overshoot) > 512:
                overflow = len(self._runner_compaction_overshoot) - 512
                for stale in list(self._runner_compaction_overshoot)[:overflow]:
                    self._runner_compaction_overshoot.discard(stale)
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
    def _is_model_request_overflow_error(error: BaseException | str) -> bool:
        text = str(error or "").strip().lower()
        if not text:
            return False
        return any(marker in text for marker in (
            "total message size",
            "exceeds limit",
            "exceeded model token limit",
            "maximum context length",
            "context length exceeded",
            "context_length_exceeded",
            "too many tokens",
        ))

    @staticmethod
    def _tool_call_name_index(messages: list[dict]) -> dict[str, str]:
        index: dict[str, str] = {}
        for message in messages:
            if not isinstance(message, dict):
                continue
            calls = message.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for item in calls:
                if not isinstance(item, dict):
                    continue
                call_id = str(item.get("id", "")).strip()
                fn = item.get("function")
                name = ""
                if isinstance(fn, dict):
                    name = str(fn.get("name", "")).strip()
                if call_id and name:
                    index[call_id] = name
        return index

    @classmethod
    def _overflow_excerpt(cls, value: str, *, max_chars: int) -> str:
        text = str(value or "").strip()
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        if max_chars <= 32:
            return text[:max_chars]
        head = max_chars - 20
        return f"{text[:head].rstrip()} ...[excerpt]"

    def _rewrite_tool_payload_for_overflow(
        self,
        *,
        content: str,
        tool_name: str,
    ) -> tuple[str | None, int]:
        content_text = str(content or "")
        if not content_text:
            return None, 0

        min_chars = int(
            getattr(
                self,
                "_overflow_fallback_tool_message_min_chars",
                self.OVERFLOW_FALLBACK_TOOL_MESSAGE_MIN_CHARS,
            ),
        )
        excerpt_chars = int(
            getattr(
                self,
                "_overflow_fallback_tool_output_excerpt_chars",
                self.OVERFLOW_FALLBACK_TOOL_OUTPUT_EXCERPT_CHARS,
            ),
        )
        if len(content_text) < min_chars:
            return None, 0

        try:
            parsed = json.loads(content_text)
        except (json.JSONDecodeError, TypeError):
            excerpt = self._overflow_excerpt(content_text, max_chars=excerpt_chars)
            payload = {
                "success": True,
                "output": (
                    f"{excerpt}\n\n"
                    "[overflow fallback applied: condensed oversized non-JSON tool payload]"
                ),
                "error": None,
                "files_changed": [],
                "data": {
                    "overflow_fallback": True,
                    "tool_name": tool_name,
                    "original_chars": len(content_text),
                },
            }
            compacted = json.dumps(payload, ensure_ascii=False)
            return compacted, len(content_text) - len(compacted)

        if not isinstance(parsed, dict):
            return None, 0

        raw_output = str(parsed.get("output", ""))
        data = parsed.get("data")
        data_dict = data if isinstance(data, dict) else {}
        kind = str(data_dict.get("content_kind", "")).strip().lower()
        artifact_ref = str(data_dict.get("artifact_ref", "")).strip()
        should_rewrite = (
            kind in self._OVERFLOW_BINARY_CONTENT_KINDS
            or bool(artifact_ref)
            or tool_name in self._HEAVY_OUTPUT_TOOLS
            or len(raw_output) >= excerpt_chars * 2
            or len(content_text) >= min_chars * 2
        )
        if not should_rewrite:
            return None, 0

        output_excerpt = self._overflow_excerpt(raw_output, max_chars=excerpt_chars)
        if output_excerpt:
            summary_output = (
                f"{output_excerpt}\n\n"
                "[overflow fallback applied: condensed oversized tool payload]"
            )
        else:
            summary_output = (
                "[overflow fallback applied: tool payload condensed to reduce "
                "request size]"
            )

        compact_payload: dict[str, Any] = {
            "success": bool(parsed.get("success", False)),
            "output": summary_output,
            "error": parsed.get("error"),
            "files_changed": list(parsed.get("files_changed", []))[:5],
        }
        files_changed = parsed.get("files_changed")
        if isinstance(files_changed, list) and len(files_changed) > 5:
            compact_payload["files_changed_count"] = len(files_changed)

        if data_dict:
            keep_keys = (
                "artifact_ref",
                "artifact_path",
                "artifact_workspace_relpath",
                "content_kind",
                "media_type",
                "size_bytes",
                "declared_size_bytes",
                "status_code",
                "source_url",
                "url",
                "truncated",
                "extract_text",
                "handler",
                "extracted_chars",
                "extraction_truncated",
            )
            compact_data = {
                key: data_dict[key]
                for key in keep_keys
                if key in data_dict
            }
            compact_data["overflow_fallback"] = True
            compact_data["tool_name"] = tool_name
            compact_data["original_chars"] = len(content_text)
            compact_payload["data"] = compact_data
        else:
            compact_payload["data"] = {
                "overflow_fallback": True,
                "tool_name": tool_name,
                "original_chars": len(content_text),
            }

        compacted = json.dumps(compact_payload, ensure_ascii=False)
        delta = len(content_text) - len(compacted)
        if delta <= 0:
            return None, 0
        return compacted, delta

    def _apply_model_overflow_fallback(
        self,
        messages: list[dict],
    ) -> tuple[list[dict], dict[str, Any]]:
        if not messages:
            return messages, {
                "overflow_fallback_applied": False,
                "overflow_fallback_rewritten_messages": 0,
                "overflow_fallback_chars_reduced": 0,
                "overflow_fallback_skipped_reason": "empty_history",
            }

        latest_tool_idx = max(
            (
                idx
                for idx, msg in enumerate(messages)
                if isinstance(msg, dict)
                and str(msg.get("role", "")).strip().lower() == "tool"
            ),
            default=-1,
        )
        call_name_index = self._tool_call_name_index(messages)

        rewritten_messages = list(messages)
        rewritten_count = 0
        chars_reduced = 0
        candidate_count = 0

        for idx, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            if role != "tool":
                continue
            if idx == latest_tool_idx:
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            candidate_count += 1
            call_id = str(message.get("tool_call_id", "")).strip()
            tool_name = call_name_index.get(call_id, "")
            rewritten_content, delta = self._rewrite_tool_payload_for_overflow(
                content=content,
                tool_name=tool_name,
            )
            if not rewritten_content or delta <= 0:
                continue
            new_message = dict(message)
            new_message["content"] = rewritten_content
            rewritten_messages[idx] = new_message
            rewritten_count += 1
            chars_reduced += delta

        if rewritten_count <= 0:
            return messages, {
                "overflow_fallback_applied": False,
                "overflow_fallback_rewritten_messages": 0,
                "overflow_fallback_chars_reduced": 0,
                "overflow_fallback_candidate_messages": candidate_count,
                "overflow_fallback_skipped_reason": "no_eligible_tool_payloads",
            }

        return rewritten_messages, {
            "overflow_fallback_applied": True,
            "overflow_fallback_rewritten_messages": rewritten_count,
            "overflow_fallback_chars_reduced": chars_reduced,
            "overflow_fallback_candidate_messages": candidate_count,
            "overflow_fallback_skipped_reason": "",
        }

    @staticmethod
    def _estimate_message_tokens(messages: list[dict]) -> int:
        total = 0
        for message in messages:
            try:
                total += estimate_tokens(json.dumps(message))
            except (TypeError, ValueError):
                total += estimate_tokens(str(message))
        return total

    def _compute_compaction_pressure_tier(
        self,
        usage_ratio: float,
    ) -> CompactionPressureTier:
        soft = float(
            getattr(
                self,
                "_compaction_pressure_ratio_soft",
                self.COMPACTION_PRESSURE_RATIO_SOFT,
            ),
        )
        hard = float(
            getattr(
                self,
                "_compaction_pressure_ratio_hard",
                self.COMPACTION_PRESSURE_RATIO_HARD,
            ),
        )
        hard = max(soft + 0.01, hard)
        if usage_ratio <= soft:
            return CompactionPressureTier.NORMAL
        if usage_ratio <= hard:
            return CompactionPressureTier.PRESSURE
        return CompactionPressureTier.CRITICAL

    def _critical_message_indices(
        self,
        messages: list[dict],
    ) -> tuple[int, ...]:
        preserve_recent = int(
            getattr(
                self,
                "_preserve_recent_critical_messages",
                self.PRESERVE_RECENT_CRITICAL_MESSAGES,
            ),
        )
        preserve_recent = max(2, preserve_recent)
        critical: set[int] = {0}
        narrative_indices: list[int] = []
        for idx, msg in enumerate(messages):
            if idx == 0 or not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip().lower()
            content = msg.get("content")
            if (
                role == "user"
                and isinstance(content, str)
                and content.startswith(self._TODO_REMINDER_PREFIX)
            ):
                critical.add(idx)
                continue
            if role == "assistant" and msg.get("tool_calls"):
                continue
            if role in {"assistant", "user", "system"}:
                narrative_indices.append(idx)
        for idx in narrative_indices[-preserve_recent:]:
            critical.add(idx)
        return tuple(sorted(critical))

    def _classify_message_for_compaction(
        self,
        message: dict,
        *,
        index: int,
        total: int,
        critical_indices: set[int],
    ) -> CompactionClass:
        del total
        if not isinstance(message, dict):
            return CompactionClass.HISTORICAL_CONTEXT
        role = str(message.get("role", "")).strip().lower()
        if role == "tool":
            return CompactionClass.TOOL_TRACE
        if role == "assistant" and message.get("tool_calls"):
            return CompactionClass.TOOL_TRACE
        if index in critical_indices:
            return CompactionClass.CRITICAL
        return CompactionClass.HISTORICAL_CONTEXT

    def _build_compaction_plan(
        self,
        messages: list[dict],
        *,
        tier: CompactionPressureTier,
    ) -> _CompactionPlan:
        critical_indices = set(self._critical_message_indices(messages))
        total = len(messages)
        newest_assistant_tool = -1
        newest_tool_result = -1
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip().lower()
            if role == "assistant" and msg.get("tool_calls"):
                newest_assistant_tool = idx
            elif role == "tool":
                newest_tool_result = idx

        preserve_tool_exchange = {
            idx for idx in {newest_assistant_tool, newest_tool_result} if idx >= 0
        }
        stage1_tool_args: list[int] = []
        stage2_tool_output: list[int] = []
        stage3_historical: list[int] = []
        for idx, msg in enumerate(messages):
            if idx == 0 or not isinstance(msg, dict):
                continue
            message_class = self._classify_message_for_compaction(
                msg,
                index=idx,
                total=total,
                critical_indices=critical_indices,
            )
            role = str(msg.get("role", "")).strip().lower()
            if message_class == CompactionClass.TOOL_TRACE:
                if idx in preserve_tool_exchange:
                    continue
                if role == "assistant" and msg.get("tool_calls"):
                    stage1_tool_args.append(idx)
                elif role == "tool":
                    stage2_tool_output.append(idx)
                continue
            if message_class == CompactionClass.HISTORICAL_CONTEXT:
                stage3_historical.append(idx)

        stage4_merge: list[int] = []
        if tier == CompactionPressureTier.CRITICAL and total > 4:
            preserve_recent = int(
                getattr(
                    self,
                    "_preserve_recent_critical_messages",
                    self.PRESERVE_RECENT_CRITICAL_MESSAGES,
                ),
            )
            merge_limit = max(1, total - max(3, preserve_recent + 1))
            for idx in range(1, merge_limit):
                if idx in preserve_tool_exchange:
                    continue
                stage4_merge.append(idx)

        return _CompactionPlan(
            critical_indices=tuple(sorted(critical_indices)),
            stage1_tool_args=tuple(stage1_tool_args),
            stage2_tool_output=tuple(stage2_tool_output),
            stage3_historical=tuple(stage3_historical),
            stage4_merge=tuple(stage4_merge),
        )

    def _set_compaction_diagnostics(self, payload: dict[str, Any]) -> None:
        self._last_compaction_diagnostics = payload

    async def _compact_messages_for_model(
        self,
        messages: list[dict],
        *,
        remaining_seconds: float | None = None,
    ) -> list[dict]:
        mode = self._runner_compaction_mode()
        if mode == "off":
            estimate = self._estimate_message_tokens(messages)
            self._set_compaction_diagnostics({
                "compaction_policy_mode": mode,
                "compaction_stage": "none",
                "compaction_candidate_count": 0,
                "compaction_skipped_reason": "policy_disabled",
                "compaction_est_tokens_before": estimate,
                "compaction_est_tokens_after": estimate,
                "compaction_compactor_calls": 0,
            })
            return messages
        if mode == "tiered":
            return await self._compact_messages_for_model_tiered(
                messages,
                remaining_seconds=remaining_seconds,
            )
        return await self._compact_messages_for_model_legacy(messages)

    async def _compact_messages_for_model_tiered(
        self,
        messages: list[dict],
        *,
        remaining_seconds: float | None = None,
    ) -> list[dict]:
        self._reset_compaction_runtime_stats()
        mode = self._runner_compaction_mode()
        if len(messages) < 3:
            self._set_compaction_diagnostics({
                "compaction_policy_mode": mode,
                "compaction_pressure_tier": CompactionPressureTier.NORMAL.value,
                "compaction_pressure_ratio": 0.0,
                "compaction_stage": "none",
                "compaction_candidate_count": 0,
                "compaction_skipped_reason": "short_history",
                "compaction_compactor_calls": 0,
            })
            return messages

        context_budget = int(
            getattr(self, "_max_model_context_tokens", self.MAX_MODEL_CONTEXT_TOKENS),
        )
        soft_ratio = float(
            getattr(
                self,
                "_compaction_pressure_ratio_soft",
                self.COMPACTION_PRESSURE_RATIO_SOFT,
            ),
        )
        hard_ratio = float(
            getattr(
                self,
                "_compaction_pressure_ratio_hard",
                self.COMPACTION_PRESSURE_RATIO_HARD,
            ),
        )
        hard_ratio = max(soft_ratio + 0.01, hard_ratio)
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

        estimate_before = self._estimate_message_tokens(messages)
        pressure_ratio = (
            estimate_before / max(1, context_budget)
            if context_budget > 0
            else 0.0
        )
        tier = self._compute_compaction_pressure_tier(pressure_ratio)
        if tier == CompactionPressureTier.NORMAL:
            self._set_compaction_diagnostics({
                "compaction_policy_mode": mode,
                "compaction_pressure_tier": tier.value,
                "compaction_pressure_ratio": round(pressure_ratio, 4),
                "compaction_stage": "none",
                "compaction_candidate_count": 0,
                "compaction_skipped_reason": "no_pressure",
                "compaction_est_tokens_before": estimate_before,
                "compaction_est_tokens_after": estimate_before,
                "compaction_compactor_calls": 0,
            })
            return messages

        compacted: list[dict] = [
            dict(message) if isinstance(message, dict) else message
            for message in messages
        ]
        plan = self._build_compaction_plan(compacted, tier=tier)
        timeout_guard_active = self._is_timeout_guard_active(remaining_seconds)
        total_candidates = (
            len(plan.stage1_tool_args)
            + len(plan.stage2_tool_output)
            + len(plan.stage3_historical)
            + len(plan.stage4_merge)
        )
        if total_candidates == 0:
            self._set_compaction_diagnostics({
                "compaction_policy_mode": mode,
                "compaction_pressure_tier": tier.value,
                "compaction_pressure_ratio": round(pressure_ratio, 4),
                "compaction_stage": "none",
                "compaction_candidate_count": 0,
                "compaction_skipped_reason": "policy_preserve",
                "compaction_est_tokens_before": estimate_before,
                "compaction_est_tokens_after": estimate_before,
                "compaction_compactor_calls": 0,
            })
            return compacted

        estimate_after = estimate_before
        pressure_after = pressure_ratio
        applied_stages: list[str] = []

        async def _compact_stage_1() -> bool:
            changed = False
            for idx in plan.stage1_tool_args:
                msg = compacted[idx]
                if not isinstance(msg, dict):
                    continue
                compact_calls = await self._compact_assistant_tool_calls(
                    msg.get("tool_calls"),
                    max_chars=compact_tool_call_args,
                )
                if compact_calls is not None and compact_calls != msg.get("tool_calls"):
                    msg["tool_calls"] = compact_calls
                    changed = True
            return changed

        async def _compact_stage_2() -> bool:
            changed = False
            for idx in plan.stage2_tool_output:
                msg = compacted[idx]
                if not isinstance(msg, dict):
                    continue
                prior = msg.get("content", "")
                compacted_content = await self._compact_tool_message_content(
                    prior,
                    max_output_chars=compact_tool_output,
                )
                if compacted_content != prior:
                    msg["content"] = compacted_content
                    changed = True
            return changed

        async def _compact_stage_3() -> bool:
            changed = False
            text_budget = (
                compact_text_chars
                if tier == CompactionPressureTier.PRESSURE
                else minimal_text_chars
            )
            for idx in plan.stage3_historical:
                msg = compacted[idx]
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "")).strip().lower()
                content = msg.get("content")
                if not isinstance(content, str):
                    continue
                if role == "user" and content.startswith(self._TODO_REMINDER_PREFIX):
                    continue
                if len(content) <= text_budget:
                    continue
                label = f"{role or 'message'} context"
                compacted_content = await self._compact_text(
                    content,
                    max_chars=text_budget,
                    label=label,
                )
                if compacted_content != content:
                    msg["content"] = compacted_content
                    changed = True
            return changed

        async def _compact_stage_4() -> bool:
            if tier != CompactionPressureTier.CRITICAL:
                return False
            if pressure_after <= hard_ratio:
                return False
            if not plan.stage4_merge:
                return False
            merge_lines: list[str] = []
            for idx in plan.stage4_merge:
                msg = compacted[idx]
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "")).strip().lower() or "unknown"
                if role == "assistant" and msg.get("tool_calls"):
                    tool_names = [
                        str(tc.get("function", {}).get("name", "tool"))
                        for tc in list(msg.get("tool_calls", []))
                        if isinstance(tc, dict)
                    ]
                    merge_lines.append(
                        f"[assistant/tool_call] {', '.join(tool_names) or 'tool call'}",
                    )
                    continue
                content = msg.get("content", "")
                if not isinstance(content, str):
                    content = str(content)
                merge_lines.append(f"[{role}] {content}")
            merged_text = "\n".join(merge_lines).strip()
            if not merged_text:
                return False
            merged_summary = await self._compact_text(
                merged_text,
                max_chars=max(480, int(compact_text_chars * 1.5)),
                label="prior conversation context",
            )
            merge_set = set(plan.stage4_merge)
            rebuilt = [compacted[0]]
            rebuilt.append({
                "role": "user",
                "content": f"Prior compacted context:\n{merged_summary}",
            })
            for idx, msg in enumerate(compacted[1:], start=1):
                if idx in merge_set:
                    continue
                rebuilt.append(msg)
            compacted[:] = rebuilt
            return True

        async def _run_stage(stage_name: str, apply_fn) -> bool:
            nonlocal estimate_after, pressure_after
            changed = await apply_fn()
            if changed:
                estimate_after = self._estimate_message_tokens(compacted)
                pressure_after = estimate_after / max(1, context_budget)
                applied_stages.append(stage_name)
            return changed

        await _run_stage("stage_1_tool_args", _compact_stage_1)
        if pressure_after <= soft_ratio:
            pass
        else:
            if timeout_guard_active:
                self._record_compaction_skip("timeout_guard")
            else:
                await _run_stage("stage_2_tool_outputs", _compact_stage_2)
                if pressure_after > soft_ratio:
                    await _run_stage("stage_3_historical", _compact_stage_3)
                if pressure_after > soft_ratio and tier == CompactionPressureTier.CRITICAL:
                    await _run_stage("stage_4_merge", _compact_stage_4)

        stats = dict(getattr(self, "_compaction_runtime_stats", {}))
        compactor_calls = int(stats.get("compactor_calls", 0))
        if compactor_calls > int(
            getattr(
                self,
                "_compaction_churn_warning_calls",
                self.COMPACTION_CHURN_WARNING_CALLS,
            ),
        ):
            logger.warning(
                "High compactor churn in runner tiered policy: calls=%d tier=%s "
                "tokens_before=%d tokens_after=%d",
                compactor_calls,
                tier.value,
                estimate_before,
                estimate_after,
            )

        skipped_reason = ""
        if timeout_guard_active and pressure_after > soft_ratio:
            skipped_reason = "timeout_guard"
        elif not applied_stages:
            skip_reasons = stats.get("skip_reasons", {})
            if isinstance(skip_reasons, dict) and skip_reasons:
                skipped_reason = str(
                    max(skip_reasons.items(), key=lambda item: item[1])[0],
                )
            else:
                skipped_reason = "no_gain"

        self._set_compaction_diagnostics({
            "compaction_policy_mode": mode,
            "compaction_pressure_tier": tier.value,
            "compaction_pressure_ratio": round(pressure_ratio, 4),
            "compaction_pressure_ratio_after": round(pressure_after, 4),
            "compaction_stage": applied_stages[-1] if applied_stages else "none",
            "compaction_applied_stages": applied_stages,
            "compaction_candidate_count": total_candidates,
            "compaction_skipped_reason": skipped_reason,
            "compaction_est_tokens_before": estimate_before,
            "compaction_est_tokens_after": estimate_after,
            "compaction_compactor_calls": compactor_calls,
            "compaction_skip_reasons": stats.get("skip_reasons", {}),
        })
        return compacted

    async def _compact_messages_for_model_legacy(
        self,
        messages: list[dict],
    ) -> list[dict]:
        """Legacy eager compaction path kept for rollout safety."""
        self._reset_compaction_runtime_stats()
        mode = self._runner_compaction_mode()
        if len(messages) < 3:
            self._set_compaction_diagnostics({
                "compaction_policy_mode": mode,
                "compaction_stage": "none",
                "compaction_candidate_count": 0,
                "compaction_skipped_reason": "short_history",
                "compaction_compactor_calls": 0,
            })
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

        estimate_before = self._estimate_message_tokens(messages)
        if estimate_before <= context_budget:
            self._set_compaction_diagnostics({
                "compaction_policy_mode": mode,
                "compaction_stage": "none",
                "compaction_candidate_count": 0,
                "compaction_skipped_reason": "no_pressure",
                "compaction_est_tokens_before": estimate_before,
                "compaction_est_tokens_after": estimate_before,
                "compaction_compactor_calls": 0,
            })
            return messages

        compacted: list[dict] = [
            dict(message) if isinstance(message, dict) else message
            for message in messages
        ]
        stage_name = "stage_1_tool_args"
        candidate_count = 0

        for msg in compacted:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip().lower()
            if role != "assistant" or not msg.get("tool_calls"):
                continue
            candidate_count += 1
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
            stage: str,
        ) -> None:
            nonlocal stage_name, candidate_count
            preserve_from = max(1, len(compacted) - preserve_recent)
            stage_name = stage
            for idx, msg in enumerate(compacted):
                if idx == 0 or idx >= preserve_from or not isinstance(msg, dict):
                    continue
                candidate_count += 1
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
            stage="stage_2_general",
        )
        if self._estimate_message_tokens(compacted) > context_budget:
            await _apply_pass(
                preserve_recent=4,
                tool_chars=120,
                text_chars=minimal_text_chars,
                stage="stage_3_minimal",
            )

        if self._estimate_message_tokens(compacted) > context_budget:
            preserve_from = max(1, len(compacted) - 3)
            old_context = compacted[1:preserve_from]
            recent = compacted[preserve_from:]
            while (
                recent
                and isinstance(recent[0], dict)
                and recent[0].get("role") == "tool"
            ):
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
                            f"[assistant/tool_call] {', '.join(tool_names) or 'tool call'}",
                        )
                        continue
                    if not isinstance(content, str):
                        content = str(content)
                    merged_lines.append(f"[{role}] {content}")

                merged_text = "\n".join(merged_lines).strip()
                if merged_text:
                    stage_name = "stage_4_merge"
                    merged_summary = await self._compact_text(
                        merged_text,
                        max_chars=700,
                        label="prior conversation context",
                    )
                    compacted = [
                        compacted[0],
                        {
                            "role": "user",
                            "content": f"Prior compacted context:\n{merged_summary}",
                        },
                        *recent,
                    ]

        estimate_after = self._estimate_message_tokens(compacted)
        stats = dict(getattr(self, "_compaction_runtime_stats", {}))
        self._set_compaction_diagnostics({
            "compaction_policy_mode": mode,
            "compaction_stage": stage_name,
            "compaction_candidate_count": candidate_count,
            "compaction_skipped_reason": "",
            "compaction_est_tokens_before": estimate_before,
            "compaction_est_tokens_after": estimate_after,
            "compaction_compactor_calls": int(stats.get("compactor_calls", 0)),
            "compaction_skip_reasons": stats.get("skip_reasons", {}),
        })
        return compacted
