"""Subtask runner: encapsulates single-subtask execution.

Owns the tool-calling loop, response validation, verification gates,
and memory extraction for one subtask.  Returns compact structured
results so the orchestrator never touches raw prompts or messages.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from loom.auth.runtime import build_run_auth_context as build_run_auth_context
from loom.config import Config
from loom.engine.semantic_compactor import SemanticCompactor
from loom.engine.verification import VerificationGates, VerificationResult
from loom.events.bus import EventBus
from loom.events.types import (
    ARTIFACT_CONFINEMENT_VIOLATION,
    MODEL_INVOCATION,
    TOKEN_STREAMED,
)
from loom.models.base import ModelResponse
from loom.models.router import ModelRouter, ResponseValidator
from loom.prompts.assembler import PromptAssembler
from loom.recovery.questions import QuestionAnswer, QuestionManager, QuestionRequest
from loom.state.memory import MemoryEntry, MemoryManager
from loom.state.task_state import Subtask, Task, TaskStateManager, TaskStatus
from loom.tools.registry import (
    ToolRegistry,
    ToolResult,
    normalize_tool_execution_surface,
)
from loom.tools.workspace import ChangeLog

from . import compaction as runner_compaction
from . import execution as runner_execution
from . import memory as runner_memory
from . import policy as runner_policy
from . import telemetry as runner_telemetry
from .execution import _COMPACTOR_EVENT_CONTEXT
from .settings import RunnerSettings
from .types import (
    CompactionClass,
    CompactionPressureTier,
    SubtaskResult,
    ToolCallRecord,
    _CompactionPlan,
)
from .types import SubtaskResultStatus as SubtaskResultStatus

logger = logging.getLogger(__name__)


def _resolve_build_run_auth_context():
    """Preserve package-level monkeypatch seam for auth context construction."""
    try:
        import loom.engine.runner as runner_module
    except Exception:
        return build_run_auth_context
    candidate = getattr(runner_module, "build_run_auth_context", None)
    if callable(candidate):
        return candidate
    return build_run_auth_context


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
    ENABLE_ARTIFACT_TELEMETRY_EVENTS = True
    ARTIFACT_TELEMETRY_MAX_METADATA_CHARS = 1200
    ENABLE_MODEL_OVERFLOW_FALLBACK = True
    EXECUTOR_COMPLETION_CONTRACT_MODE = "off"
    ENABLE_MUTATION_IDEMPOTENCY = False
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
    _VERIFIED_SEAL_OUTCOMES = frozenset({
        "pass",
        "pass_with_warnings",
        "partial_verified",
    })
    _SEAL_CONFIRMATION_EVIDENCE_TOOLS = frozenset({
        "read_file",
        "spreadsheet",
        "web_search",
        "web_fetch",
        "web_fetch_html",
        "fact_checker",
    })

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
        question_manager: QuestionManager | None = None,
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
        self._question_manager = question_manager
        settings = RunnerSettings.from_config(config, runner_defaults=self)
        self._max_tool_iterations = settings.max_tool_iterations
        self._max_subtask_wall_clock_seconds = settings.max_subtask_wall_clock_seconds
        self._max_model_context_tokens = settings.max_model_context_tokens
        self._max_state_summary_chars = settings.max_state_summary_chars
        self._max_verification_summary_chars = settings.max_verification_summary_chars
        self._default_tool_result_output_chars = settings.default_tool_result_output_chars
        self._heavy_tool_result_output_chars = settings.heavy_tool_result_output_chars
        self._compact_tool_result_output_chars = settings.compact_tool_result_output_chars
        self._compact_text_output_chars = settings.compact_text_output_chars
        self._minimal_text_output_chars = settings.minimal_text_output_chars
        self._tool_call_argument_context_chars = settings.tool_call_argument_context_chars
        self._compact_tool_call_argument_chars = settings.compact_tool_call_argument_chars
        self._runner_compaction_policy_mode = settings.runner_compaction_policy_mode
        self._preserve_recent_critical_messages = settings.preserve_recent_critical_messages
        self._compaction_pressure_ratio_soft = settings.compaction_pressure_ratio_soft
        self._compaction_pressure_ratio_hard = settings.compaction_pressure_ratio_hard
        self._compaction_no_gain_min_delta_chars = settings.compaction_no_gain_min_delta_chars
        self._compaction_no_gain_attempt_limit = settings.compaction_no_gain_attempt_limit
        self._compaction_timeout_guard_seconds = settings.compaction_timeout_guard_seconds
        self._extractor_timeout_guard_seconds = settings.extractor_timeout_guard_seconds
        self._extractor_tool_args_max_chars = settings.extractor_tool_args_max_chars
        self._extractor_tool_trace_max_chars = settings.extractor_tool_trace_max_chars
        self._extractor_prompt_max_chars = settings.extractor_prompt_max_chars
        self._compaction_churn_warning_calls = settings.compaction_churn_warning_calls
        self._enable_filetype_ingest_router = settings.enable_filetype_ingest_router
        self._enable_artifact_telemetry_events = settings.enable_artifact_telemetry_events
        self._artifact_telemetry_max_metadata_chars = (
            settings.artifact_telemetry_max_metadata_chars
        )
        self._enable_model_overflow_fallback = settings.enable_model_overflow_fallback
        self._ingest_artifact_retention_max_age_days = (
            settings.ingest_artifact_retention_max_age_days
        )
        self._ingest_artifact_retention_max_files_per_scope = (
            settings.ingest_artifact_retention_max_files_per_scope
        )
        self._ingest_artifact_retention_max_bytes_per_scope = (
            settings.ingest_artifact_retention_max_bytes_per_scope
        )
        self._executor_completion_contract_mode = settings.executor_completion_contract_mode
        self._enable_mutation_idempotency = settings.enable_mutation_idempotency
        self._ask_user_v2_enabled = settings.ask_user_v2_enabled
        self._ask_user_runtime_blocking_enabled = settings.ask_user_runtime_blocking_enabled
        self._ask_user_policy = settings.ask_user_policy
        self._ask_user_timeout_seconds = settings.ask_user_timeout_seconds
        self._ask_user_timeout_default_response = (
            settings.ask_user_timeout_default_response
        )
        self._ask_user_max_pending_per_task = settings.ask_user_max_pending_per_task
        self._ask_user_max_questions_per_subtask = (
            settings.ask_user_max_questions_per_subtask
        )
        self._ask_user_min_seconds_between_questions = (
            settings.ask_user_min_seconds_between_questions
        )
        self._evidence_context_text_max_chars = settings.evidence_context_text_max_chars

        self._compactor = SemanticCompactor(
            model_router,
            model_event_hook=self._emit_compactor_model_event,
            role="compactor",
            tier=1,
            allow_role_fallback=True,
            **settings.compactor_kwargs,
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
        self._active_subtask_telemetry_counters: dict[str, int] | None = None

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

    @staticmethod
    def _task_status_text(task: Task) -> str:
        """Normalize current task status to lowercase text."""
        raw = getattr(task, "status", "")
        if hasattr(raw, "value"):
            raw = getattr(raw, "value")
        return str(raw or "").strip().lower()

    async def _wait_for_task_control_window(self, task: Task) -> bool:
        """Block while paused; return False when cancellation is observed."""
        while True:
            status = self._task_status_text(task)
            if status == TaskStatus.PAUSED.value:
                await asyncio.sleep(0.1)
                continue
            if status == TaskStatus.CANCELLED.value:
                return False
            return True

    def _ask_user_runtime_enabled(self) -> bool:
        return bool(
            self._ask_user_v2_enabled
            and self._ask_user_runtime_blocking_enabled
            and self._question_manager is not None
        )

    @staticmethod
    def _execution_surface_for_task(task: Task) -> str:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        raw_surface = (
            metadata.get("execution_surface", "")
            or metadata.get("run_surface", "")
            or metadata.get("client_surface", "")
        )
        if not str(raw_surface or "").strip():
            task_id = str(getattr(task, "id", "") or "").strip().lower()
            if task_id.startswith("cowork-"):
                return "tui"
        return normalize_tool_execution_surface(raw_surface, default="api")

    def _set_waiting_for_user_input(
        self,
        *,
        task: Task,
        subtask: Subtask,
        request: QuestionRequest,
    ) -> None:
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        task.metadata["awaiting_user_input"] = {
            "question_id": str(request.question_id or "").strip(),
            "subtask_id": str(subtask.id or "").strip(),
            "question": str(request.question or "").strip(),
            "question_type": str(request.question_type or "").strip(),
            "requested_at": datetime.now().isoformat(),
        }
        try:
            self._state.save(task)
        except Exception:
            logger.debug(
                "Failed to persist awaiting_user_input marker for %s/%s",
                task.id,
                subtask.id,
                exc_info=True,
            )

    def _clear_waiting_for_user_input(
        self,
        *,
        task: Task,
        question_id: str = "",
    ) -> None:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if not isinstance(metadata, dict):
            return
        marker = metadata.get("awaiting_user_input")
        if question_id:
            marker_question_id = ""
            if isinstance(marker, dict):
                marker_question_id = str(marker.get("question_id", "") or "").strip()
            if marker_question_id and marker_question_id != question_id:
                return
        if "awaiting_user_input" not in metadata:
            return
        metadata.pop("awaiting_user_input", None)
        task.metadata = metadata
        try:
            self._state.save(task)
        except Exception:
            logger.debug(
                "Failed clearing awaiting_user_input marker for %s",
                task.id,
                exc_info=True,
            )

    @staticmethod
    def _ask_user_limit_error(reason: str) -> ToolResult:
        guidance = (
            "Proceed using the best explicit assumption available and include "
            "a risk note describing the unresolved uncertainty."
        )
        return ToolResult.fail(f"{reason} {guidance}")

    async def _persist_ask_user_answer_memory(
        self,
        *,
        task: Task,
        subtask: Subtask,
        request: QuestionRequest,
        answer: QuestionAnswer,
    ) -> None:
        detail_lines = [f"Question: {request.question}"]
        answer_text = answer.text_response.strip()
        if answer_text:
            detail_lines.append(f"Answer: {answer_text}")
        elif answer.response_type:
            detail_lines.append(f"Answer type: {answer.response_type}")

        user_summary = answer_text or answer.response_type or "Clarification received"
        if len(user_summary) > 140:
            user_summary = f"{user_summary[:139].rstrip()}…"

        try:
            await self._memory.store(MemoryEntry(
                task_id=task.id,
                subtask_id=subtask.id,
                entry_type="user_instruction",
                summary=user_summary,
                detail="\n".join(detail_lines),
                tags="ask_user,clarification",
            ))
        except Exception:
            logger.debug(
                "Failed storing ask_user instruction memory for %s/%s",
                task.id,
                subtask.id,
                exc_info=True,
            )

        if not answer.selected_option_ids:
            return
        selected_labels = ", ".join(answer.selected_labels) or ", ".join(answer.selected_option_ids)
        decision_summary = f"Clarification decision: {selected_labels}"
        if len(decision_summary) > 140:
            decision_summary = f"{decision_summary[:139].rstrip()}…"
        try:
            await self._memory.store(MemoryEntry(
                task_id=task.id,
                subtask_id=subtask.id,
                entry_type="decision",
                summary=decision_summary,
                detail=json.dumps(
                    {
                        "question_id": answer.question_id,
                        "selected_option_ids": list(answer.selected_option_ids),
                        "selected_labels": list(answer.selected_labels),
                        "response_type": answer.response_type,
                        "source": answer.source,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                tags="ask_user,decision",
            ))
        except Exception:
            logger.debug(
                "Failed storing ask_user decision memory for %s/%s",
                task.id,
                subtask.id,
                exc_info=True,
            )

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
    def _new_subtask_telemetry_counters() -> dict[str, int]:
        return runner_telemetry.new_subtask_telemetry_counters()

    @staticmethod
    def _safe_int(value: Any) -> int:
        return runner_telemetry.safe_int(value)

    @staticmethod
    def _safe_float(value: Any) -> float:
        return runner_telemetry.safe_float(value)

    @staticmethod
    def _normalize_reason_code(reason: str) -> str:
        return runner_telemetry.normalize_reason_code(reason)

    def _telemetry_events_enabled(self) -> bool:
        return runner_telemetry.telemetry_events_enabled(self)

    def _increment_subtask_counter(self, key: str, amount: int = 1) -> None:
        runner_telemetry.increment_subtask_counter(self, key, amount)

    @classmethod
    def _sanitize_url_for_telemetry(cls, raw_url: Any) -> str:
        return runner_telemetry.sanitize_url_for_telemetry(raw_url)

    @staticmethod
    def _stable_json_length(value: Any) -> int:
        try:
            serialized = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except (TypeError, ValueError):
            serialized = json.dumps(str(value), ensure_ascii=False)
        return len(serialized)

    @staticmethod
    def _stable_json_digest(value: Any) -> str:
        try:
            serialized = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except (TypeError, ValueError):
            serialized = json.dumps(str(value), ensure_ascii=False)
        return hashlib.sha1(
            serialized.encode("utf-8", errors="ignore"),
        ).hexdigest()[:16]

    @classmethod
    def _normalize_handler_metadata_value(cls, raw: Any) -> Any:
        return runner_telemetry.normalize_handler_metadata_value(raw)

    def _summarize_oversize_handler_metadata(
        self,
        *,
        normalized: Any,
        original_chars: int,
        max_chars: int,
    ) -> dict[str, Any]:
        return runner_telemetry.summarize_oversize_handler_metadata(
            self,
            normalized=normalized,
            original_chars=original_chars,
            max_chars=max_chars,
        )

    def _sanitize_handler_metadata(self, raw: Any) -> Any:
        return runner_telemetry.sanitize_handler_metadata(self, raw)

    def _emit_telemetry_event(
        self,
        *,
        event_type: str,
        task_id: str,
        data: dict[str, Any],
        counter_key: str = "",
        counter_amount: int = 1,
    ) -> None:
        runner_telemetry.emit_telemetry_event(
            self,
            event_type=event_type,
            task_id=task_id,
            data=data,
            counter_key=counter_key,
            counter_amount=counter_amount,
        )

    @staticmethod
    def _normalize_run_id(task: Task) -> str:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if not isinstance(metadata, dict):
            return ""
        return str(metadata.get("run_id", "") or "").strip()

    @staticmethod
    def _mutation_target_from_args(arguments: dict[str, Any]) -> str:
        return runner_policy.mutation_target_from_args(arguments)

    def _mutation_idempotency_key(
        self,
        *,
        task: Task,
        subtask: Subtask,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[str, str]:
        return runner_policy.mutation_idempotency_key(
            task=task,
            subtask_id=subtask.id,
            tool_name=tool_name,
            arguments=arguments,
            normalize_run_id=self._normalize_run_id,
            stable_json_digest=self._stable_json_digest,
        )

    @staticmethod
    def _extract_completion_json(text: str) -> dict[str, Any] | None:
        payload = str(text or "").strip()
        if not payload:
            return None
        payload = ResponseValidator._strip_markdown_fences(payload)
        decoder = json.JSONDecoder()
        candidates: list[dict[str, Any]] = []

        def _consider(candidate: Any) -> None:
            if not isinstance(candidate, dict):
                return
            required = {"status", "deliverables_touched", "verification_notes"}
            if required.issubset(set(candidate.keys())):
                candidates.append(candidate)

        try:
            parsed = decoder.decode(payload)
            _consider(parsed)
        except json.JSONDecodeError:
            pass

        for i, ch in enumerate(payload):
            if ch != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(payload[i:])
            except json.JSONDecodeError:
                continue
            _consider(candidate)

        return candidates[0] if candidates else None

    def _validate_completion_contract(self, response_text: str) -> tuple[bool, str]:
        contract = self._extract_completion_json(response_text)
        if contract is None:
            return False, (
                "Missing completion JSON contract. Include keys: "
                "status, deliverables_touched, verification_notes."
            )
        status = str(contract.get("status", "")).strip().lower()
        if status not in {"success", "partial", "failed"}:
            return False, "Completion contract 'status' must be one of success|partial|failed."
        touched = contract.get("deliverables_touched")
        if not isinstance(touched, list):
            return False, "Completion contract 'deliverables_touched' must be an array."
        notes = contract.get("verification_notes")
        if not isinstance(notes, str):
            return False, "Completion contract 'verification_notes' must be a string."
        return True, ""

    def _completion_contract_mutation_mismatch(
        self,
        *,
        response_text: str,
        tool_calls: list[ToolCallRecord],
        workspace: Path | None,
    ) -> str:
        contract = self._extract_completion_json(response_text)
        if not isinstance(contract, dict):
            return ""
        touched = contract.get("deliverables_touched")
        if not isinstance(touched, list):
            return ""
        declared = [
            str(item).strip()
            for item in touched
            if str(item).strip()
        ]
        declared_normalized = self._normalize_deliverable_paths(
            declared,
            workspace=workspace,
        )
        actual: list[str] = []
        seen: set[str] = set()
        for call in list(tool_calls or []):
            result = getattr(call, "result", None)
            if not getattr(result, "success", False):
                continue
            changed = list(getattr(result, "files_changed", []) or [])
            for raw in changed:
                normalized = self._normalize_path_for_policy(
                    str(raw),
                    workspace,
                )
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                actual.append(normalized)
        if set(declared_normalized) == set(actual):
            return ""
        declared_text = ", ".join(declared_normalized) if declared_normalized else "(none)"
        actual_text = ", ".join(actual) if actual else "(none)"
        return (
            "Completion contract mismatch: "
            "deliverables_touched does not match actual file mutations. "
            f"Declared: {declared_text}. Actual: {actual_text}."
        )

    def _artifact_event_common_payload(
        self,
        *,
        subtask_id: str,
        tool_name: str,
        tool_args: dict,
        result: ToolResult,
    ) -> dict[str, Any]:
        return runner_telemetry.artifact_event_common_payload(
            self,
            subtask_id=subtask_id,
            tool_name=tool_name,
            tool_args=tool_args,
            result=result,
        )

    def _emit_artifact_ingest_telemetry(
        self,
        *,
        task_id: str,
        subtask_id: str,
        tool_name: str,
        tool_args: dict,
        result: ToolResult,
    ) -> None:
        runner_telemetry.emit_artifact_ingest_telemetry(
            self,
            task_id=task_id,
            subtask_id=subtask_id,
            tool_name=tool_name,
            tool_args=tool_args,
            result=result,
        )

    def _emit_artifact_read_telemetry(
        self,
        *,
        task_id: str,
        subtask_id: str,
        tool_name: str,
        tool_args: dict,
        result: ToolResult,
    ) -> None:
        runner_telemetry.emit_artifact_read_telemetry(
            self,
            task_id=task_id,
            subtask_id=subtask_id,
            tool_name=tool_name,
            tool_args=tool_args,
            result=result,
        )

    def _compaction_decision_from_diagnostics(
        self,
        diagnostics: dict[str, Any],
    ) -> tuple[str, str]:
        return runner_telemetry.compaction_decision_from_diagnostics(self, diagnostics)

    def _emit_compaction_policy_decision_from_diagnostics(
        self,
        *,
        task_id: str,
        subtask_id: str,
    ) -> None:
        runner_telemetry.emit_compaction_policy_decision_from_diagnostics(
            self,
            task_id=task_id,
            subtask_id=subtask_id,
        )

    def _emit_overflow_fallback_telemetry(
        self,
        *,
        task_id: str,
        subtask_id: str,
        report: dict[str, Any],
    ) -> None:
        runner_telemetry.emit_overflow_fallback_telemetry(
            self,
            task_id=task_id,
            subtask_id=subtask_id,
            report=report,
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
        forbidden_deliverables: list[str] | None = None,
        allowed_output_prefixes: list[str] | None = None,
        enforce_deliverable_paths: bool = False,
        edit_existing_only: bool = False,
        retry_strategy: str = "",
    ) -> tuple[SubtaskResult, VerificationResult]:
        """Execute a subtask: prompt -> tool loop -> verify -> extract memory."""
        return await runner_execution.run_subtask(
            self,
            task,
            subtask,
            model_tier=model_tier,
            retry_context=retry_context,
            changelog=changelog,
            prior_successful_tool_calls=prior_successful_tool_calls,
            prior_evidence_records=prior_evidence_records,
            expected_deliverables=expected_deliverables,
            forbidden_deliverables=forbidden_deliverables,
            allowed_output_prefixes=allowed_output_prefixes,
            enforce_deliverable_paths=enforce_deliverable_paths,
            edit_existing_only=edit_existing_only,
            retry_strategy=retry_strategy,
            build_run_auth_context_fn=_resolve_build_run_auth_context(),
        )

    def _spawn_memory_extraction(
        self, task_id: str, subtask_id: str, result: SubtaskResult,
    ) -> None:
        runner_memory.spawn_memory_extraction(
            self,
            task_id,
            subtask_id,
            result,
            logger=logger,
        )

    async def _extract_memory(
        self, task_id: str, subtask_id: str, result: SubtaskResult,
    ) -> None:
        await runner_memory.extract_memory(
            self,
            task_id,
            subtask_id,
            result,
            logger=logger,
        )

    def _parse_memory_entries(
        self, response: ModelResponse, task_id: str, subtask_id: str,
    ) -> list[MemoryEntry]:
        return runner_memory.parse_memory_entries(
            self,
            response,
            task_id,
            subtask_id,
        )

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
            files_changed = [
                str(item or "").strip()
                for item in list(getattr(result, "files_changed", []) or [])
                if str(item or "").strip()
            ]
            if len(files_changed) > 20:
                data["files_changed"] = files_changed[:20]
                data["files_changed_paths"] = files_changed[:20]
                data["files_changed_count"] = len(files_changed)
            else:
                data["files_changed"] = files_changed
                data["files_changed_paths"] = files_changed
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

    @staticmethod
    def _is_forbidden_output_path_error(error: str | None) -> bool:
        text = str(error or "").strip().lower()
        return "reason_code=forbidden_output_path" in text

    def _emit_compactor_model_event(self, payload: dict) -> None:
        """Bridge semantic-compactor model events into task model_invocation events."""
        context = _COMPACTOR_EVENT_CONTEXT.get()
        if not context:
            return
        if bool(payload.get("compactor_warning")):
            self._increment_subtask_counter("compactor_warning_count")
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
        return runner_policy.normalize_path_for_policy(path_text, workspace)

    @classmethod
    def _normalize_deliverable_paths(
        cls,
        expected_deliverables: list[str],
        *,
        workspace: Path | None,
    ) -> list[str]:
        return runner_policy.normalize_deliverable_paths(
            expected_deliverables,
            workspace=workspace,
        )

    @classmethod
    def _is_mutating_file_tool(cls, tool_name: str, tool_args: dict) -> bool:
        return runner_policy.is_mutating_file_tool(
            tool_name=tool_name,
            tool_args=tool_args,
            write_mutating_tools=cls._WRITE_MUTATING_TOOLS,
            spreadsheet_write_operations=cls._SPREADSHEET_WRITE_OPERATIONS,
        )

    @classmethod
    def _target_paths_for_policy(
        cls,
        *,
        tool_name: str,
        tool_args: dict,
        workspace: Path | None,
    ) -> list[str]:
        return runner_policy.target_paths_for_policy(
            tool_name=tool_name,
            tool_args=tool_args,
            workspace=workspace,
            is_mutating_file_tool_fn=cls._is_mutating_file_tool,
        )

    @classmethod
    def _looks_like_deliverable_variant(
        cls,
        *,
        candidate: str,
        canonical: str,
    ) -> bool:
        return runner_policy.looks_like_deliverable_variant(
            candidate=candidate,
            canonical=canonical,
            variant_suffix_markers=cls._VARIANT_SUFFIX_MARKERS,
        )

    @classmethod
    def _validate_deliverable_write_policy(
        cls,
        *,
        tool_name: str,
        tool_args: dict,
        workspace: Path | None,
        expected_deliverables: list[str],
        forbidden_deliverables: list[str],
        allowed_output_prefixes: list[str],
        enforce_deliverable_paths: bool,
        edit_existing_only: bool,
    ) -> str | None:
        return runner_policy.validate_deliverable_write_policy(
            tool_name=tool_name,
            tool_args=tool_args,
            workspace=workspace,
            expected_deliverables=expected_deliverables,
            forbidden_deliverables=forbidden_deliverables,
            allowed_output_prefixes=allowed_output_prefixes,
            enforce_deliverable_paths=enforce_deliverable_paths,
            edit_existing_only=edit_existing_only,
            normalize_deliverable_paths=cls._normalize_deliverable_paths,
            target_paths_for_policy=cls._target_paths_for_policy,
            looks_like_deliverable_variant=cls._looks_like_deliverable_variant,
        )

    @staticmethod
    def _artifact_seal_registry(task: Task) -> dict[str, dict[str, object]]:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        registry = metadata.get("artifact_seals")
        if not isinstance(registry, dict):
            registry = {}
            metadata["artifact_seals"] = registry
        task.metadata = metadata
        return registry

    @staticmethod
    def _seal_timestamp_is_after(candidate: str, baseline: str) -> bool:
        candidate_text = str(candidate or "").strip()
        baseline_text = str(baseline or "").strip()
        if not candidate_text:
            return False
        if not baseline_text:
            return True
        try:
            return datetime.fromisoformat(candidate_text) > datetime.fromisoformat(baseline_text)
        except Exception:
            return candidate_text > baseline_text

    @staticmethod
    def _verification_outcome_for_subtask(task: Task, subtask_id: str) -> str:
        if not subtask_id:
            return ""
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if not isinstance(metadata, dict):
            return ""
        scorecard = metadata.get("validity_scorecard")
        if not isinstance(scorecard, dict):
            return ""
        per_subtask = scorecard.get("subtask_metrics")
        if not isinstance(per_subtask, dict):
            return ""
        entry = per_subtask.get(subtask_id)
        if not isinstance(entry, dict):
            return ""
        return str(entry.get("verification_outcome", "") or "").strip().lower()

    @classmethod
    def _seal_origin_is_verified(
        cls,
        *,
        task: Task,
        seal: dict[str, object] | None,
    ) -> bool:
        if not isinstance(seal, dict):
            return False
        if bool(seal.get("verified_origin", False)):
            return True
        explicit_outcome = str(seal.get("verification_outcome", "") or "").strip().lower()
        if explicit_outcome in cls._VERIFIED_SEAL_OUTCOMES:
            return True
        subtask_id = str(seal.get("subtask_id", "") or "").strip()
        if not subtask_id:
            return False
        outcome = cls._verification_outcome_for_subtask(task, subtask_id)
        return outcome in cls._VERIFIED_SEAL_OUTCOMES

    @classmethod
    def _is_confirmation_evidence_call(cls, call: ToolCallRecord) -> bool:
        if not isinstance(call, ToolCallRecord):
            return False
        if not bool(getattr(call.result, "success", False)):
            return False
        tool_name = str(getattr(call, "tool", "") or "").strip().lower()
        if tool_name not in cls._SEAL_CONFIRMATION_EVIDENCE_TOOLS:
            return False
        if tool_name == "spreadsheet":
            operation = str(call.args.get("operation", "")).strip().lower()
            return operation in {"read", "summary"}
        return True

    @classmethod
    def _has_post_seal_confirmation_evidence(
        cls,
        *,
        baseline_timestamp: str,
        prior_successful_tool_calls: list[ToolCallRecord],
        current_tool_calls: list[ToolCallRecord],
    ) -> bool:
        for call in [*prior_successful_tool_calls, *current_tool_calls]:
            if not cls._is_confirmation_evidence_call(call):
                continue
            timestamp = str(getattr(call, "timestamp", "") or "").strip()
            if (
                baseline_timestamp
                and not cls._seal_timestamp_is_after(timestamp, baseline_timestamp)
            ):
                continue
            return True
        return False

    @classmethod
    def _latest_seal_timestamp(
        cls,
        protected_paths: list[tuple[str, dict[str, object]]],
    ) -> str:
        latest = ""
        for _path, seal in protected_paths:
            if not isinstance(seal, dict):
                continue
            sealed_at = str(seal.get("sealed_at", "") or "").strip()
            if not sealed_at:
                continue
            if cls._seal_timestamp_is_after(sealed_at, latest):
                latest = sealed_at
        return latest

    @classmethod
    def _validate_sealed_artifact_mutation_policy(
        cls,
        *,
        task: Task,
        tool_name: str,
        tool_args: dict,
        workspace: Path | None,
        prior_successful_tool_calls: list[ToolCallRecord],
        current_tool_calls: list[ToolCallRecord],
    ) -> str | None:
        return runner_policy.validate_sealed_artifact_mutation_policy(
            task=task,
            tool_name=tool_name,
            tool_args=tool_args,
            workspace=workspace,
            prior_successful_tool_calls=prior_successful_tool_calls,
            current_tool_calls=current_tool_calls,
            target_paths_for_policy=cls._target_paths_for_policy,
            artifact_seal_registry=cls._artifact_seal_registry,
            seal_origin_is_verified=cls._seal_origin_is_verified,
            latest_seal_timestamp=cls._latest_seal_timestamp,
            has_post_seal_confirmation_evidence=cls._has_post_seal_confirmation_evidence,
        )

    @classmethod
    def _mutation_paths_for_reseal(
        cls,
        *,
        tool_name: str,
        tool_args: dict,
        workspace: Path | None,
        tool_result: ToolResult,
    ) -> list[str]:
        paths = cls._target_paths_for_policy(
            tool_name=tool_name,
            tool_args=tool_args,
            workspace=workspace,
        )
        seen = set(paths)
        for raw in tool_result.files_changed:
            normalized = cls._normalize_path_for_policy(str(raw), workspace)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            paths.append(normalized)
        return paths

    @classmethod
    def _reseal_tracked_artifacts_after_mutation(
        cls,
        *,
        task: Task,
        workspace: Path | None,
        tool_name: str,
        tool_args: dict,
        tool_result: ToolResult,
        subtask_id: str,
        tool_call_id: str,
    ) -> int:
        return runner_policy.reseal_tracked_artifacts_after_mutation(
            task=task,
            workspace=workspace,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            subtask_id=subtask_id,
            tool_call_id=tool_call_id,
            artifact_seal_registry=cls._artifact_seal_registry,
            mutation_paths_for_reseal=cls._mutation_paths_for_reseal,
            normalize_path_for_policy=cls._normalize_path_for_policy,
            seal_origin_is_verified=cls._seal_origin_is_verified,
        )

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
        return await runner_compaction.compact_text(
            self,
            text,
            max_chars=max_chars,
            label=label,
            logger=logger,
        )

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
        return await runner_compaction.summarize_tool_data(self, data)

    async def _summarize_tool_call_arguments(
        self,
        args: object,
        *,
        max_chars: int,
        label: str,
    ) -> dict:
        return await runner_compaction.summarize_tool_call_arguments(
            self,
            args,
            max_chars=max_chars,
            label=label,
        )

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
        return await runner_compaction.compact_assistant_tool_calls(
            self,
            tool_calls,
            max_chars=max_chars,
        )

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
        return await runner_compaction.compact_tool_message_content(
            self,
            content,
            max_output_chars=max_output_chars,
        )

    @staticmethod
    def _is_model_request_overflow_error(error: BaseException | str) -> bool:
        return runner_compaction.is_model_request_overflow_error(error)

    @staticmethod
    def _tool_call_name_index(messages: list[dict]) -> dict[str, str]:
        return runner_compaction.tool_call_name_index(messages)

    @classmethod
    def _overflow_excerpt(cls, value: str, *, max_chars: int) -> str:
        return runner_compaction.overflow_excerpt(value, max_chars=max_chars)

    def _rewrite_tool_payload_for_overflow(
        self,
        *,
        content: str,
        tool_name: str,
    ) -> tuple[str | None, int]:
        return runner_compaction.rewrite_tool_payload_for_overflow(
            self,
            content=content,
            tool_name=tool_name,
        )

    def _apply_model_overflow_fallback(
        self,
        messages: list[dict],
    ) -> tuple[list[dict], dict[str, Any]]:
        return runner_compaction.apply_model_overflow_fallback(
            self,
            messages,
        )

    @staticmethod
    def _estimate_message_tokens(messages: list[dict]) -> int:
        return runner_compaction.estimate_message_tokens(messages)

    def _compute_compaction_pressure_tier(
        self,
        usage_ratio: float,
    ) -> CompactionPressureTier:
        return runner_compaction.compute_compaction_pressure_tier(
            self,
            usage_ratio,
        )

    def _critical_message_indices(
        self,
        messages: list[dict],
    ) -> tuple[int, ...]:
        return runner_compaction.critical_message_indices(self, messages)

    def _classify_message_for_compaction(
        self,
        message: dict,
        *,
        index: int,
        total: int,
        critical_indices: set[int],
    ) -> CompactionClass:
        return runner_compaction.classify_message_for_compaction(
            message,
            index=index,
            total=total,
            critical_indices=critical_indices,
        )

    def _build_compaction_plan(
        self,
        messages: list[dict],
        *,
        tier: CompactionPressureTier,
    ) -> _CompactionPlan:
        return runner_compaction.build_compaction_plan(
            self,
            messages,
            tier=tier,
        )

    def _set_compaction_diagnostics(self, payload: dict[str, Any]) -> None:
        runner_compaction.set_compaction_diagnostics(self, payload)

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
        return await runner_compaction.compact_messages_for_model_tiered(
            self,
            messages,
            remaining_seconds=remaining_seconds,
            logger=logger,
        )

    async def _compact_messages_for_model_legacy(
        self,
        messages: list[dict],
    ) -> list[dict]:
        return await runner_compaction.compact_messages_for_model_legacy(
            self,
            messages,
        )
