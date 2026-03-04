"""Cowork session: conversation-first interactive execution.

The core interaction loop:
1. User sends a message
2. Model responds with text + optional tool calls
3. Tool calls are executed, results fed back
4. Repeat until model produces text-only response
5. Display response to user, wait for next input

All conversation turns are persisted to SQLite (write-through).
Session state is maintained as always-in-context structured metadata.
The model can recall past conversation via the conversation_recall tool.
Complex work can be delegated to the task orchestrator via delegate_task.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loom.cowork.approval import ApprovalDecision, ToolApprover
from loom.cowork.memory_indexer import CoworkMemoryIndexer
from loom.cowork.session_state import SessionState, extract_state_from_tool_events
from loom.engine.semantic_compactor import SemanticCompactor
from loom.models.base import ModelProvider, TokenUsage, ToolCall
from loom.models.retry import (
    ModelRetryPolicy,
    call_with_model_retry,
    stream_with_model_retry,
)
from loom.tools.registry import (
    ToolRegistry,
    ToolResult,
    normalize_tool_auth_mode,
    normalize_tool_execution_surfaces,
    tool_auth_required,
)
from loom.tools.shell import high_risk_command_metadata
from loom.tools.tooling_common.wp_policy import (
    assess_wp_cli_risk,
    format_wp_risk_info,
)
from loom.utils.tokens import estimate_tokens as _estimate_tokens

if TYPE_CHECKING:
    from loom.learning.reflection import GapAnalysisEngine
    from loom.state.conversation_store import ConversationStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ToolCallEvent:
    """Emitted when a tool call starts or completes."""

    name: str
    args: dict
    tool_call_id: str = ""
    result: ToolResult | None = None
    elapsed_ms: int = 0


@dataclass
class CoworkTurn:
    """One full turn in the conversation (model response + tool calls)."""

    text: str = ""
    tool_calls: list[ToolCallEvent] = field(default_factory=list)
    tokens_used: int = 0
    model: str = ""
    latency_ms: int = 0
    total_time_ms: int = 0
    tokens_per_second: float = 0.0
    context_tokens: int = 0
    context_messages: int = 0
    omitted_messages: int = 0
    recall_index_used: bool = False


@dataclass(frozen=True)
class CoworkStopRequestedError(Exception):
    """Raised when an in-flight cowork turn is asked to stop cooperatively."""

    reason: str = "user_requested"
    stage: str = ""
    path: str = "cooperative"

    def __str__(self) -> str:
        base = "Cowork turn stop requested"
        reason = str(self.reason or "").strip()
        stage = str(self.stage or "").strip()
        if reason:
            base += f" ({reason})"
        if stage:
            base += f" at {stage}"
        return base


@dataclass(frozen=True)
class _ContextWindowStats:
    """Per-request context metadata for lightweight turn telemetry."""

    context_tokens: int = 0
    context_messages: int = 0
    omitted_messages: int = 0
    recall_index_used: bool = False


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

# Tools that signal "ask the user" — the tool loop should pause and
# return to the user instead of continuing.
_USER_INTERACTION_TOOLS = frozenset({"ask_user"})
_RUNTIME_INTERNAL_PREFIX = "_loom_"
_RUN_TOOL_PARENT_KEYS = frozenset({
    "_loom_parent_tool_call_id",
    "_loom_parent_tool_name",
})

MAX_TOOL_ITERATIONS = 40
MAX_IDENTICAL_TOOL_BATCH_STREAK = 3
MAX_IDENTICAL_TOOL_BATCH_RECOVERY_HINTS = 1
REPEATED_TOOL_BATCH_SYSTEM_HINT = (
    "[System: You are repeating the same tool-call batch with identical arguments. "
    "Do not call the same tools again unchanged. Synthesize a direct answer from "
    "existing tool outputs. Only call another tool if arguments change and you "
    "briefly justify why.]"
)
DEFAULT_TOOL_RESULT_OUTPUT_CHARS = 3_000
HEAVY_TOOL_RESULT_OUTPUT_CHARS = 1_200
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

# Phrases that suggest the user is referencing earlier context.
_DANGLING_REF_INDICATORS = [
    "like we discussed", "as before", "remember when",
    "go back to", "that file", "that error", "that function",
    "what we did", "earlier", "the previous", "as I said",
    "like I mentioned", "that thing", "we already",
]

_CORE_TOOL_NAMES = (
    "ask_user",
    "task_tracker",
    "conversation_recall",
    "delegate_task",
)

_GENERAL_TOOL_NAMES = (
    "glob_find",
    "ripgrep_search",
    "read_file",
    "list_directory",
    "shell_execute",
    "git_command",
    "web_search",
    "web_fetch",
)

_CODING_TOOL_NAMES = (
    "read_file",
    "write_file",
    "edit_file",
    "move_file",
    "delete_file",
    "glob_find",
    "ripgrep_search",
    "analyze_code",
    "read_artifact",
    "list_directory",
    "shell_execute",
    "git_command",
)

_WEB_TOOL_NAMES = (
    "web_search",
    "web_fetch",
    "web_fetch_html",
    "archive_access",
    "fact_checker",
    "citation_manager",
    "primary_source_ocr",
    "document_write",
)

_WRITING_TOOL_NAMES = (
    "document_write",
    "humanize_writing",
    "peer_review_simulator",
    "citation_manager",
    "fact_checker",
)

_FINANCE_TOOL_NAMES = (
    "market_data_api",
    "symbol_universe_api",
    "sec_fundamentals_api",
    "sentiment_feeds_api",
    "economic_data_api",
    "insider_trading_tracker",
    "short_interest_analyzer",
    "options_flow_analyzer",
    "earnings_surprise_predictor",
    "factor_exposure_engine",
    "valuation_engine",
    "portfolio_optimizer",
    "portfolio_evaluator",
    "portfolio_recommender",
    "historical_currency_normalizer",
    "inflation_calculator",
    "macro_regime_engine",
    "opportunity_ranker",
    "timeline_visualizer",
)

_GREETING_TOKENS = frozenset({
    "hi",
    "hello",
    "hey",
    "yo",
    "sup",
    "howdy",
})

_CODING_KEYWORDS = (
    "code",
    "coding",
    "bug",
    "debug",
    "test",
    "lint",
    "build",
    "compile",
    "refactor",
    "function",
    "class",
    "module",
    "repo",
    "repository",
    "file",
    "directory",
    "path",
    "readme",
    "git",
    "commit",
    "branch",
    "pr",
    "shell",
    "terminal",
    "script",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".rs",
    ".go",
    ".java",
)

_WEB_KEYWORDS = (
    "web",
    "url",
    "http",
    "https",
    "site",
    "website",
    "docs",
    "documentation",
    "article",
    "source",
    "citation",
    "research",
    "search",
)

_WRITING_KEYWORDS = (
    "write",
    "draft",
    "rewrite",
    "rephrase",
    "tone",
    "style",
    "report",
    "memo",
    "summary",
    "summarize",
    "outline",
    "proposal",
)

_FINANCE_KEYWORDS = (
    "stock",
    "ticker",
    "equity",
    "market",
    "portfolio",
    "valuation",
    "earnings",
    "sec",
    "insider",
    "macro",
    "inflation",
    "options",
    "factor",
    "currency",
    "sentiment",
    "investment",
)

_SPREADSHEET_KEYWORDS = (
    "spreadsheet",
    "csv",
    "xlsx",
    "rows",
    "columns",
    "table",
)

_MATH_KEYWORDS = (
    "calculate",
    "calculation",
    "math",
    "sum",
    "average",
    "mean",
    "median",
    "percent",
    "percentage",
    "ratio",
)

_FALLBACK_TOOL_NAMES = (
    "list_tools",
    "run_tool",
)
_HYBRID_FALLBACK_TOOLS = frozenset(_FALLBACK_TOOL_NAMES)
_HYBRID_TOOL_STALL_MARKERS = (
    "don't have",
    "do not have",
    "cannot access",
    "can't access",
    "no direct tool",
    "not available",
    "unavailable",
    "available tools are",
)
_HYBRID_RECOVERY_SYSTEM_HINT = (
    "[System: In hybrid mode, if the needed tool is not directly callable, "
    "first call list_tools with {\"detail\":\"compact\"} to discover names, "
    "then call list_tools with {\"detail\":\"schema\",\"query\":\"<tool name>\"} "
    "to inspect arguments, then call run_tool with the exact tool name and JSON "
    "arguments. Do not call list_tools schema broadly; keep schema lookup scoped "
    "to the specific tool(s) you plan to run. Do not stop at describing tool "
    "availability.]"
)

_TOOL_EXPOSURE_MODES = frozenset({"full", "adaptive", "hybrid"})
_TYPED_TOOL_SCHEMA_CAP = 16
_TYPED_TOOL_SCHEMA_BYTE_BUDGET = 12 * 1024
_COMPACT_CONTEXT_TOKEN_CAP = 24_000
_CONTEXT_OUTPUT_RESERVE_TOKENS = 4_000
_CONTEXT_RECENT_MESSAGE_CAP = 48
_RECALL_INDEX_MAX_USER_TOPICS = 4
_RECALL_INDEX_MAX_TOOL_NAMES = 6
_RECALL_INDEX_SNIPPET_CHARS = 96
_RECALL_INDEX_MAX_CHARS = 1200
_RECALL_INDEX_SECTION_LINE_CAP = 4
_RECALL_INDEX_LINE_CHARS = 220
_RESUMED_TOOL_OUTPUT_CHARS = 1400
_RESUMED_TOOL_ERROR_CHARS = 600
_RESUMED_TOOL_DATA_CHARS = 800
_RESUMED_TOOL_RAW_CHARS = 1800
_RESUMED_TOOL_FILES_PREVIEW = 8


def _normalize_tool_exposure_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode in _TOOL_EXPOSURE_MODES:
        return mode
    return "adaptive"


def _estimate_message_tokens(msg: dict) -> int:
    """Estimate tokens for a full message dict, including multimodal content."""
    content = msg.get("content") or ""
    total = _estimate_tokens(content)

    if msg.get("tool_calls"):
        total += _estimate_tokens(json.dumps(msg["tool_calls"]))

    # Account for multimodal content blocks in tool results
    if msg.get("role") == "tool" and content:
        try:
            parsed = json.loads(content)
            for block in parsed.get("content_blocks", []):
                btype = block.get("type", "")
                if btype == "image":
                    w = block.get("width", 0)
                    h = block.get("height", 0)
                    if w > 0 and h > 0:
                        # Cap at reasonable bounds to avoid overflow
                        total += min((w * h) // 750, 200_000)
                elif btype == "document":
                    pages = block.get("page_count", 1)
                    pr = block.get("page_range")
                    if pr and isinstance(pr, list) and len(pr) == 2:
                        pages = max(0, pr[1] - pr[0])
                    total += min(pages * 1500, 200_000)
        except (json.JSONDecodeError, TypeError) as e:
            logger.debug("Token estimation parse failed: %s", e)

    return max(1, total)


def _fallback_response_tokens(
    response_text: str,
    tool_calls: list[ToolCall] | None,
) -> int:
    """Estimate response tokens when provider usage is unavailable."""
    total = 0

    if response_text:
        total += _estimate_tokens(response_text)

    if tool_calls:
        serialized = [
            {"name": tc.name, "arguments": tc.arguments}
            for tc in tool_calls
        ]
        total += _estimate_tokens(json.dumps(serialized, separators=(",", ":")))

    return max(1, total)


def _tokens_per_second(tokens_used: int, total_time_ms: int) -> float:
    """Compute turn throughput from tokens and elapsed time."""
    if tokens_used <= 0 or total_time_ms <= 0:
        return 0.0
    return float(tokens_used) / (float(total_time_ms) / 1000.0)


def _compact_preview(text: Any, limit: int) -> tuple[str, bool]:
    """Normalize and truncate text for compact context hints."""
    normalized = " ".join(str(text or "").split())
    if limit <= 0:
        return "", bool(normalized)
    if len(normalized) <= limit:
        return normalized, False
    marker = "...[truncated]"
    if limit <= len(marker):
        return normalized[:limit], True
    keep = max(0, limit - len(marker))
    return f"{normalized[:keep]}{marker}", True


class CoworkSession:
    """Interactive cowork session.

    Maintains the full conversation history as a list[dict] in the
    OpenAI message format.  Each user message, assistant response, and
    tool result is preserved across the entire session.

    All messages are persisted to SQLite via write-through.  The in-memory
    list is a hot cache; the ConversationStore holds everything.

    The session does NOT own the I/O — it yields events that the CLI
    or TUI renders.  This keeps the session testable and transport-agnostic.
    """

    def __init__(
        self,
        model: ModelProvider,
        tools: ToolRegistry,
        compactor_model: ModelProvider | None = None,
        workspace: Path | None = None,
        scratch_dir: Path | None = None,
        system_prompt: str = "",
        max_context_messages: int = 200,
        approver: ToolApprover | None = None,
        store: ConversationStore | None = None,
        session_id: str = "",
        session_state: SessionState | None = None,
        max_context_tokens: int = _COMPACT_CONTEXT_TOKEN_CAP,
        reflection: GapAnalysisEngine | None = None,
        model_retry_policy: ModelRetryPolicy | None = None,
        tool_exposure_mode: str = "adaptive",
        auth_context: Any | None = None,
        enable_filetype_ingest_router: bool = True,
        ingest_artifact_retention_max_age_days: int = 14,
        ingest_artifact_retention_max_files_per_scope: int = 96,
        ingest_artifact_retention_max_bytes_per_scope: int = 268_435_456,
        delegate_progress_callback: (
            Callable[[dict[str, Any]], Awaitable[None] | None] | None
        ) = None,
        memory_index_enabled: bool = False,
        memory_index_llm_extraction_enabled: bool = True,
        memory_index_model: ModelProvider | None = None,
        memory_index_model_role: str = "",
        memory_index_role_strict: bool = False,
        memory_index_queue_max_batches: int = 32,
        memory_index_section_limit: int = _RECALL_INDEX_SECTION_LINE_CAP,
        recall_index_max_chars: int = _RECALL_INDEX_MAX_CHARS,
    ):
        self._model = model
        self._tools = tools
        self._workspace = workspace
        self._scratch_dir = scratch_dir
        self._max_context = max_context_messages
        self._max_context_tokens = max(
            4096,
            int(max_context_tokens),
        )
        self._approver = approver
        self._auth_context = auth_context
        self._total_tokens = 0
        self._turn_counter = 0
        self._message_counter = 0  # monotonic per-message counter for DB persistence

        # Persistence
        self._store = store
        self._session_id = session_id

        # Automatic gap analysis (ALM behavioral learning)
        self._reflection = reflection

        # Session state (Layer 1 for cowork)
        self._session_state = session_state or SessionState(
            workspace=str(workspace) if workspace else "",
            model_name=model.name,
            session_id=session_id,
        )
        self._static_system_prompt = system_prompt
        self._compactor = SemanticCompactor(model=compactor_model or model)
        self._model_retry_policy = model_retry_policy or ModelRetryPolicy()
        self._tool_exposure_mode = _normalize_tool_exposure_mode(tool_exposure_mode)
        self._enable_filetype_ingest_router = bool(enable_filetype_ingest_router)
        self._ingest_artifact_retention_max_age_days = max(
            0,
            int(ingest_artifact_retention_max_age_days),
        )
        self._ingest_artifact_retention_max_files_per_scope = max(
            1,
            int(ingest_artifact_retention_max_files_per_scope),
        )
        self._ingest_artifact_retention_max_bytes_per_scope = max(
            1024,
            int(ingest_artifact_retention_max_bytes_per_scope),
        )
        self._delegate_progress_callback = delegate_progress_callback
        self._memory_index_enabled = bool(memory_index_enabled)
        self._memory_index_llm_extraction_enabled = bool(
            memory_index_llm_extraction_enabled,
        )
        self._memory_index_model = memory_index_model
        self._memory_index_model_role = str(memory_index_model_role or "").strip().lower()
        self._memory_index_role_strict = bool(memory_index_role_strict)
        self._memory_index_queue_max_batches = max(1, int(memory_index_queue_max_batches))
        self._memory_index_section_limit = max(1, int(memory_index_section_limit))
        self._recall_index_max_chars = max(500, int(recall_index_max_chars))
        self._memory_indexer: CoworkMemoryIndexer | None = None

        # Learned behaviors section (injected into system prompt)
        self._behaviors_section = ""

        self._messages_lock = asyncio.Lock()
        self._counter_lock = asyncio.Lock()
        self._stop_requested = asyncio.Event()
        self._stop_reason = ""
        self._pause_requested = asyncio.Event()
        self._pending_inject_instructions: list[str] = []

        self._messages: list[dict] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": self._build_system_content()})

        self._bind_hybrid_tools()
        self._maybe_start_memory_indexer()

    @property
    def messages(self) -> list[dict]:
        """Read-only view of conversation history."""
        return list(self._messages)

    @property
    def workspace(self) -> Path | None:
        return self._workspace

    @property
    def total_tokens(self) -> int:
        """Cumulative token count across all turns in this session."""
        return self._total_tokens

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def session_state(self) -> SessionState:
        return self._session_state

    @property
    def stop_requested(self) -> bool:
        """Return True when a cooperative stop has been requested."""
        return self._stop_requested.is_set()

    def request_stop(self, reason: str = "user_requested") -> None:
        """Request cooperative interruption of the active cowork turn."""
        self._stop_reason = str(reason or "user_requested").strip() or "user_requested"
        self._stop_requested.set()

    def clear_stop_request(self) -> None:
        """Clear any prior cooperative stop request."""
        self._stop_requested.clear()
        self._stop_reason = ""

    @property
    def pause_requested(self) -> bool:
        """Return True when cooperative pause has been requested."""
        return self._pause_requested.is_set()

    def request_pause(self) -> None:
        """Request cooperative pause at the next safe execution boundary."""
        self._pause_requested.set()

    def request_resume(self) -> None:
        """Resume cowork execution after a cooperative pause."""
        self._pause_requested.clear()

    @property
    def has_pending_inject_instruction(self) -> bool:
        """Return True when an inject instruction is queued for next boundary."""
        return bool(self._pending_inject_instructions)

    @property
    def pending_inject_instruction_count(self) -> int:
        """Return count of queued inject instructions."""
        return len(self._pending_inject_instructions)

    def queue_inject_instruction(self, text: str) -> None:
        """Append an inject instruction for the next safe boundary."""
        clean = str(text or "").strip()
        if not clean:
            return
        self._pending_inject_instructions.append(clean)

    def dequeue_pending_inject_instruction(self) -> str:
        """Pop the next queued inject instruction (FIFO)."""
        if not self._pending_inject_instructions:
            return ""
        return str(self._pending_inject_instructions.pop(0) or "").strip()

    def clear_pending_inject_instruction(self) -> None:
        """Drop any pending inject instruction."""
        self._pending_inject_instructions = []

    def _raise_if_stop_requested(self, *, stage: str = "") -> None:
        """Raise CoworkStopRequestedError when a cooperative stop is pending."""
        if not self._stop_requested.is_set():
            return
        raise CoworkStopRequestedError(
            reason=self._stop_reason or "user_requested",
            stage=str(stage or "").strip(),
            path="cooperative",
        )

    async def _await_if_paused(self, *, stage: str = "") -> None:
        """Cooperatively pause execution until resumed or stopped."""
        while self._pause_requested.is_set():
            self._raise_if_stop_requested(stage=stage or "paused")
            await asyncio.sleep(0.05)

    async def _apply_pending_inject_instruction_if_any(self) -> bool:
        """Inject one queued steering instruction into conversation context."""
        instruction = self.dequeue_pending_inject_instruction()
        if not instruction:
            return False
        content = (
            "Steering instruction from user: "
            f"{instruction}\n"
            "Apply this instruction while preserving existing evidence."
        )
        self._messages.append({"role": "system", "content": content})
        await self._persist_turn("system", content=content)
        return True

    @property
    def compactor(self) -> SemanticCompactor:
        return self._compactor

    @property
    def memory_indexer(self) -> CoworkMemoryIndexer | None:
        return self._memory_indexer

    def _maybe_start_memory_indexer(self) -> None:
        if not self._memory_index_enabled:
            return
        if self._store is None or not self._session_id:
            return
        if self._memory_indexer is not None:
            return
        self._memory_indexer = CoworkMemoryIndexer(
            store=self._store,
            session_id=self._session_id,
            session_state=self._session_state,
            model=self._memory_index_model,
            model_role=self._memory_index_model_role,
            llm_extraction_enabled=self._memory_index_llm_extraction_enabled,
            role_strict=self._memory_index_role_strict,
            queue_max_batches=self._memory_index_queue_max_batches,
            section_limit=self._memory_index_section_limit,
        )

    async def _hydrate_memory_snapshot(self) -> None:
        if self._store is None or not self._session_id:
            return
        try:
            snapshot = await self._store.get_cowork_memory_active_snapshot(
                self._session_id,
                max_decisions=self._memory_index_section_limit,
                max_proposals=self._memory_index_section_limit,
                max_research=self._memory_index_section_limit,
                max_questions=self._memory_index_section_limit,
            )
            self._session_state.update_memory_snapshot(snapshot)
            state = await self._store.get_cowork_memory_index_state(self._session_id)
            self._session_state.update_memory_index_meta(
                last_indexed_turn=int(state.get("last_indexed_turn", 0) or 0),
                degraded=bool(state.get("index_degraded", False)),
                failure_count=int(state.get("failure_count", 0) or 0),
                last_error=str(state.get("last_error", "") or ""),
            )
        except Exception as e:
            logger.debug(
                "Failed hydrating cowork memory snapshot session=%s: %s",
                self._session_id,
                e,
            )

    async def wait_for_memory_index_idle(self, *, timeout_seconds: float = 5.0) -> bool:
        indexer = self._memory_indexer
        if indexer is None:
            return True
        return await indexer.wait_idle(timeout_seconds=timeout_seconds)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send(
        self, user_message: str,
    ) -> AsyncGenerator[CoworkTurn | ToolCallEvent | str, None]:
        """Send a user message and yield events as the model responds.

        Yields:
        - str: streamed text tokens (partial)
        - ToolCallEvent: tool call start/completion
        - CoworkTurn: final turn summary when the model finishes

        The caller should display streamed text inline, show tool calls
        as they happen, and use the final CoworkTurn for bookkeeping.
        """
        self.clear_stop_request()
        async with self._counter_lock:
            self._turn_counter += 1
        self._session_state.set_focus(user_message.split("\n")[0])

        # Possibly inject a recall hint
        hint = self._maybe_recall_hint(user_message)

        self._messages.append({"role": "user", "content": user_message})
        await self._persist_turn("user", content=user_message)

        if hint:
            self._messages.append({"role": "system", "content": hint})
            await self._persist_turn("system", content=hint)

        turn_started_at = time.monotonic()
        first_model_latency_ms: int | None = None
        context_stats: _ContextWindowStats | None = None
        total_tokens = 0
        all_tool_events: list[ToolCallEvent] = []
        text_parts: list[str] = []
        turn_tool_schemas = self._tool_schemas_for_turn(user_message)
        hybrid_recovery_used = False
        last_tool_batch_signature = ""
        identical_tool_batch_streak = 0
        repeated_tool_batch_recovery_hints_used = 0

        for _ in range(MAX_TOOL_ITERATIONS):
            await self._await_if_paused(stage="model_request")
            await self._apply_pending_inject_instruction_if_any()
            self._raise_if_stop_requested(stage="model_request")
            # Call the model
            async def _invoke_complete():
                nonlocal context_stats
                context_window, stats = self._context_window_with_stats()
                if context_stats is None:
                    context_stats = stats
                return await self._model.complete(
                    context_window,
                    tools=turn_tool_schemas or None,
                )

            response = await call_with_model_retry(
                _invoke_complete,
                policy=self._model_retry_policy,
            )
            self._raise_if_stop_requested(stage="model_response")
            if first_model_latency_ms is None:
                first_model_latency_ms = max(
                    1,
                    int((time.monotonic() - turn_started_at) * 1000),
                )
            response_tokens = int(getattr(response.usage, "total_tokens", 0) or 0)
            if response_tokens <= 0:
                response_tokens = _fallback_response_tokens(
                    response.text or "",
                    response.tool_calls,
                )
            total_tokens += response_tokens

            # Accumulate text across iterations
            if response.text:
                text_parts.append(response.text)

            if response.has_tool_calls():
                batch_signature = self._tool_batch_signature(response.tool_calls)
                response_has_text = bool(str(response.text or "").strip())
                if (
                    batch_signature
                    and not response_has_text
                    and batch_signature == last_tool_batch_signature
                ):
                    identical_tool_batch_streak += 1
                elif batch_signature:
                    identical_tool_batch_streak = 1
                else:
                    identical_tool_batch_streak = 0
                last_tool_batch_signature = batch_signature

                if (
                    identical_tool_batch_streak >= MAX_IDENTICAL_TOOL_BATCH_STREAK
                    and repeated_tool_batch_recovery_hints_used
                    < MAX_IDENTICAL_TOOL_BATCH_RECOVERY_HINTS
                ):
                    repeated_tool_batch_recovery_hints_used += 1
                    self._messages.append({
                        "role": "system",
                        "content": REPEATED_TOOL_BATCH_SYSTEM_HINT,
                    })
                    await self._persist_turn(
                        "system",
                        content=REPEATED_TOOL_BATCH_SYSTEM_HINT,
                    )
                    continue

                if (
                    identical_tool_batch_streak >= MAX_IDENTICAL_TOOL_BATCH_STREAK
                    and repeated_tool_batch_recovery_hints_used
                    >= MAX_IDENTICAL_TOOL_BATCH_RECOVERY_HINTS
                ):
                    fallback = self._build_repeated_tool_batch_fallback(all_tool_events)
                    self._messages.append({"role": "assistant", "content": fallback})
                    await self._persist_turn("assistant", content=fallback)
                    text_parts.append(fallback)
                    break

                tc_dicts = self._tool_calls_to_dicts(response.tool_calls)
                self._messages.append({
                    "role": "assistant",
                    "content": response.text or "",
                    "tool_calls": tc_dicts,
                })
                await self._persist_turn(
                    "assistant",
                    content=response.text,
                    tool_calls=tc_dicts,
                )

                ask_user_pending = False
                for tc in response.tool_calls:
                    await self._await_if_paused(stage=f"tool_start:{tc.name}")
                    self._raise_if_stop_requested(stage=f"tool_start:{tc.name}")
                    event = ToolCallEvent(
                        name=tc.name,
                        args=tc.arguments,
                        tool_call_id=str(getattr(tc, "id", "") or ""),
                    )
                    yield event  # signal: tool call starting

                    result, elapsed_ms = await self._execute_tool_call(
                        tc.name,
                        tc.arguments,
                        tool_call_id=event.tool_call_id,
                        caller_tool_name=tc.name,
                    )
                    event.result = result
                    event.elapsed_ms = elapsed_ms
                    all_tool_events.append(event)
                    yield event
                    self._raise_if_stop_requested(stage=f"tool_complete:{tc.name}")

                    await self._append_tool_result(tc.id, tc.name, result)

                    if tc.name in _USER_INTERACTION_TOOLS:
                        ask_user_pending = True

                if ask_user_pending:
                    break
            else:
                self._messages.append({"role": "assistant", "content": response.text or ""})
                await self._persist_turn("assistant", content=response.text or "")
                if self._should_retry_with_hybrid_fallback(
                    user_message=user_message,
                    response_text=response.text or "",
                    turn_tool_schemas=turn_tool_schemas,
                    recovery_used=hybrid_recovery_used,
                ):
                    hybrid_recovery_used = True
                    self._messages.append({
                        "role": "system",
                        "content": _HYBRID_RECOVERY_SYSTEM_HINT,
                    })
                    await self._persist_turn(
                        "system",
                        content=_HYBRID_RECOVERY_SYSTEM_HINT,
                    )
                    continue
                break

        interaction_elapsed_ms = max(
            1,
            int((time.monotonic() - turn_started_at) * 1000),
        )
        latency_ms = (
            first_model_latency_ms
            if isinstance(first_model_latency_ms, int) and first_model_latency_ms > 0
            else interaction_elapsed_ms
        )
        tokens_per_second = _tokens_per_second(total_tokens, interaction_elapsed_ms)

        # Post-turn bookkeeping
        self._maybe_trim()
        self._total_tokens += total_tokens
        self._session_state.turn_count = self._turn_counter
        self._session_state.total_tokens = self._total_tokens

        # Extract session state from tool events
        extract_state_from_tool_events(
            self._session_state, self._turn_counter, all_tool_events,
        )

        # Automatic reflection: analyze this exchange for behavioral patterns
        response_text = "\n\n".join(text_parts)
        await self._reflect(user_message, response_text)

        self._update_system_message()

        # Persist session metadata
        await self._persist_session_metadata()

        final_context = context_stats or _ContextWindowStats()
        yield CoworkTurn(
            text=response_text,
            tool_calls=all_tool_events,
            tokens_used=total_tokens,
            model=self._model.name,
            latency_ms=latency_ms,
            total_time_ms=interaction_elapsed_ms,
            tokens_per_second=tokens_per_second,
            context_tokens=final_context.context_tokens,
            context_messages=final_context.context_messages,
            omitted_messages=final_context.omitted_messages,
            recall_index_used=final_context.recall_index_used,
        )

    async def send_streaming(
        self, user_message: str,
    ) -> AsyncGenerator[CoworkTurn | ToolCallEvent | str, None]:
        """Like send() but streams text tokens as they arrive.

        Yields str chunks for incremental display, ToolCallEvents for
        tool calls, and a final CoworkTurn.
        """
        self.clear_stop_request()
        async with self._counter_lock:
            self._turn_counter += 1
        self._session_state.set_focus(user_message.split("\n")[0])

        hint = self._maybe_recall_hint(user_message)

        self._messages.append({"role": "user", "content": user_message})
        await self._persist_turn("user", content=user_message)

        if hint:
            self._messages.append({"role": "system", "content": hint})
            await self._persist_turn("system", content=hint)

        turn_started_at = time.monotonic()
        first_model_latency_ms: int | None = None
        context_stats: _ContextWindowStats | None = None
        total_tokens = 0
        all_tool_events: list[ToolCallEvent] = []
        all_text_parts: list[str] = []
        turn_tool_schemas = self._tool_schemas_for_turn(user_message)
        hybrid_recovery_used = False
        last_tool_batch_signature = ""
        identical_tool_batch_streak = 0
        repeated_tool_batch_recovery_hints_used = 0

        for _ in range(MAX_TOOL_ITERATIONS):
            await self._await_if_paused(stage="stream_model_request")
            await self._apply_pending_inject_instruction_if_any()
            self._raise_if_stop_requested(stage="stream_model_request")
            iter_text_parts: list[str] = []
            final_tool_calls: list[ToolCall] | None = None
            final_usage = None

            def _invoke_stream():
                nonlocal context_stats
                context_window, stats = self._context_window_with_stats()
                if context_stats is None:
                    context_stats = stats
                return self._model.stream(
                    context_window,
                    tools=turn_tool_schemas or None,
                )

            async for chunk in stream_with_model_retry(
                _invoke_stream,
                policy=self._model_retry_policy,
            ):
                await self._await_if_paused(stage="stream_chunk")
                self._raise_if_stop_requested(stage="stream_chunk")
                if first_model_latency_ms is None and (
                    chunk.text or chunk.tool_calls is not None or chunk.usage is not None
                ):
                    first_model_latency_ms = max(
                        1,
                        int((time.monotonic() - turn_started_at) * 1000),
                    )
                if chunk.text:
                    iter_text_parts.append(chunk.text)
                    yield chunk.text
                if chunk.tool_calls is not None:
                    final_tool_calls = chunk.tool_calls
                if chunk.usage is not None:
                    final_usage = chunk.usage
                self._raise_if_stop_requested(stage="stream_chunk_processed")

            response_text = "".join(iter_text_parts)
            self._raise_if_stop_requested(stage="stream_iteration_complete")
            if first_model_latency_ms is None and (
                response_text or final_tool_calls is not None or final_usage is not None
            ):
                first_model_latency_ms = max(
                    1,
                    int((time.monotonic() - turn_started_at) * 1000),
                )
            response_tokens = int((final_usage or TokenUsage()).total_tokens or 0)
            if response_tokens <= 0:
                response_tokens = _fallback_response_tokens(
                    response_text,
                    final_tool_calls,
                )
            total_tokens += response_tokens

            if response_text:
                all_text_parts.append(response_text)

            if final_tool_calls:
                batch_signature = self._tool_batch_signature(final_tool_calls)
                response_has_text = bool(str(response_text or "").strip())
                if (
                    batch_signature
                    and not response_has_text
                    and batch_signature == last_tool_batch_signature
                ):
                    identical_tool_batch_streak += 1
                elif batch_signature:
                    identical_tool_batch_streak = 1
                else:
                    identical_tool_batch_streak = 0
                last_tool_batch_signature = batch_signature

                if (
                    identical_tool_batch_streak >= MAX_IDENTICAL_TOOL_BATCH_STREAK
                    and repeated_tool_batch_recovery_hints_used
                    < MAX_IDENTICAL_TOOL_BATCH_RECOVERY_HINTS
                ):
                    repeated_tool_batch_recovery_hints_used += 1
                    self._messages.append({
                        "role": "system",
                        "content": REPEATED_TOOL_BATCH_SYSTEM_HINT,
                    })
                    await self._persist_turn(
                        "system",
                        content=REPEATED_TOOL_BATCH_SYSTEM_HINT,
                    )
                    continue

                if (
                    identical_tool_batch_streak >= MAX_IDENTICAL_TOOL_BATCH_STREAK
                    and repeated_tool_batch_recovery_hints_used
                    >= MAX_IDENTICAL_TOOL_BATCH_RECOVERY_HINTS
                ):
                    fallback = self._build_repeated_tool_batch_fallback(all_tool_events)
                    self._messages.append({"role": "assistant", "content": fallback})
                    await self._persist_turn("assistant", content=fallback)
                    all_text_parts.append(fallback)
                    break

                tc_dicts = self._tool_calls_to_dicts(final_tool_calls)
                self._messages.append({
                    "role": "assistant",
                    "content": response_text or "",
                    "tool_calls": tc_dicts,
                })
                await self._persist_turn(
                    "assistant",
                    content=response_text,
                    tool_calls=tc_dicts,
                )

                ask_user_pending = False
                for tc in final_tool_calls:
                    await self._await_if_paused(stage=f"tool_start:{tc.name}")
                    self._raise_if_stop_requested(stage=f"tool_start:{tc.name}")
                    event = ToolCallEvent(
                        name=tc.name,
                        args=tc.arguments,
                        tool_call_id=str(getattr(tc, "id", "") or ""),
                    )
                    yield event

                    result, elapsed_ms = await self._execute_tool_call(
                        tc.name,
                        tc.arguments,
                        tool_call_id=event.tool_call_id,
                        caller_tool_name=tc.name,
                    )
                    event.result = result
                    event.elapsed_ms = elapsed_ms
                    all_tool_events.append(event)
                    yield event
                    self._raise_if_stop_requested(stage=f"tool_complete:{tc.name}")

                    await self._append_tool_result(tc.id, tc.name, result)

                    if tc.name in _USER_INTERACTION_TOOLS:
                        ask_user_pending = True

                if ask_user_pending:
                    break
            else:
                self._messages.append({"role": "assistant", "content": response_text or ""})
                await self._persist_turn("assistant", content=response_text or "")
                if self._should_retry_with_hybrid_fallback(
                    user_message=user_message,
                    response_text=response_text or "",
                    turn_tool_schemas=turn_tool_schemas,
                    recovery_used=hybrid_recovery_used,
                ):
                    hybrid_recovery_used = True
                    self._messages.append({
                        "role": "system",
                        "content": _HYBRID_RECOVERY_SYSTEM_HINT,
                    })
                    await self._persist_turn(
                        "system",
                        content=_HYBRID_RECOVERY_SYSTEM_HINT,
                    )
                    continue
                break

        interaction_elapsed_ms = max(
            1,
            int((time.monotonic() - turn_started_at) * 1000),
        )
        latency_ms = (
            first_model_latency_ms
            if isinstance(first_model_latency_ms, int) and first_model_latency_ms > 0
            else interaction_elapsed_ms
        )
        tokens_per_second = _tokens_per_second(total_tokens, interaction_elapsed_ms)

        self._maybe_trim()
        self._total_tokens += total_tokens
        self._session_state.turn_count = self._turn_counter
        self._session_state.total_tokens = self._total_tokens

        extract_state_from_tool_events(
            self._session_state, self._turn_counter, all_tool_events,
        )

        # Automatic reflection: analyze this exchange for behavioral patterns
        response_text = "\n\n".join(all_text_parts)
        await self._reflect(user_message, response_text)

        self._update_system_message()

        await self._persist_session_metadata()

        final_context = context_stats or _ContextWindowStats()
        yield CoworkTurn(
            text=response_text,
            tool_calls=all_tool_events,
            tokens_used=total_tokens,
            model=self._model.name,
            latency_ms=latency_ms,
            total_time_ms=interaction_elapsed_ms,
            tokens_per_second=tokens_per_second,
            context_tokens=final_context.context_tokens,
            context_messages=final_context.context_messages,
            omitted_messages=final_context.omitted_messages,
            recall_index_used=final_context.recall_index_used,
        )

    # ------------------------------------------------------------------
    # Session resumption
    # ------------------------------------------------------------------

    async def resume(self, session_id: str) -> None:
        """Resume a previous session from the archive."""
        if self._store is None:
            raise RuntimeError("Cannot resume without a ConversationStore.")

        session = await self._store.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        self._session_id = session_id
        self._total_tokens = session.get("total_tokens", 0)
        self._turn_counter = session.get("turn_count", 0)

        # Restore message counter from DB to avoid collisions
        self._message_counter = await self._store.get_turn_count(session_id)

        # Restore session state
        self._session_state = SessionState.from_json(session.get("session_state"))
        self._session_state.session_id = session_id

        # Load recent turns into in-memory cache
        recent = await self._store.resume_session(session_id)
        self._messages = [
            {"role": "system", "content": self._build_system_content()},
            *self._normalize_resumed_messages(recent),
        ]
        self._maybe_start_memory_indexer()
        await self._hydrate_memory_snapshot()
        if self._memory_indexer is not None and self._message_counter > 0:
            self._memory_indexer.enqueue_up_to_turn(self._message_counter)

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_resumed_messages(messages: list[dict]) -> list[dict]:
        """Normalize DB-restored messages to compact model-facing payloads."""
        normalized: list[dict] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            out = dict(msg)
            if out.get("role") == "tool":
                raw_content = out.get("content")
                if isinstance(raw_content, str):
                    out["content"] = CoworkSession._compact_resumed_tool_content(raw_content)
            normalized.append(out)
        return normalized

    @staticmethod
    def _compact_resumed_tool_content(raw_content: str) -> str:
        """Compact large persisted ToolResult JSON for resumed prompt context."""
        if not raw_content:
            return ""

        try:
            parsed = json.loads(raw_content)
        except (json.JSONDecodeError, TypeError):
            compact, _ = _compact_preview(raw_content, _RESUMED_TOOL_RAW_CHARS)
            return compact

        if not isinstance(parsed, dict):
            compact, _ = _compact_preview(raw_content, _RESUMED_TOOL_RAW_CHARS)
            return compact

        if not any(
            key in parsed
            for key in (
                "success",
                "output",
                "error",
                "data",
                "files_changed",
                "content_blocks",
            )
        ):
            compact, _ = _compact_preview(raw_content, _RESUMED_TOOL_RAW_CHARS)
            return compact

        compact_payload: dict[str, Any] = {
            "success": bool(parsed.get("success", False)),
        }

        output = parsed.get("output")
        if output not in (None, ""):
            preview, truncated = _compact_preview(output, _RESUMED_TOOL_OUTPUT_CHARS)
            compact_payload["output"] = preview
            if truncated:
                compact_payload["output_truncated"] = True

        error = parsed.get("error")
        if error not in (None, ""):
            preview, truncated = _compact_preview(error, _RESUMED_TOOL_ERROR_CHARS)
            compact_payload["error"] = preview
            if truncated:
                compact_payload["error_truncated"] = True

        files_changed = parsed.get("files_changed")
        if isinstance(files_changed, list) and files_changed:
            cleaned = [
                str(path).strip()
                for path in files_changed
                if str(path).strip()
            ]
            if cleaned:
                compact_payload["files_changed_count"] = len(cleaned)
                compact_payload["files_changed_preview"] = cleaned[:_RESUMED_TOOL_FILES_PREVIEW]

        data = parsed.get("data")
        if data not in (None, {}, [], ""):
            if isinstance(data, dict):
                compact_payload["data_keys"] = sorted(str(k) for k in data.keys())[:12]
            data_text, data_truncated = _compact_preview(
                json.dumps(data, ensure_ascii=False, default=str),
                _RESUMED_TOOL_DATA_CHARS,
            )
            compact_payload["data_preview"] = data_text
            if data_truncated:
                compact_payload["data_truncated"] = True

        blocks = parsed.get("content_blocks")
        if isinstance(blocks, list) and blocks:
            block_types = [
                str(block.get("type", "")).strip()
                for block in blocks
                if isinstance(block, dict) and str(block.get("type", "")).strip()
            ]
            compact_payload["content_blocks"] = {
                "count": len(blocks),
                "types": block_types[:8],
            }

        return json.dumps(compact_payload, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _line_within_budget(
        lines: list[str],
        line: str,
        *,
        max_chars: int,
    ) -> bool:
        candidate = [*lines, line]
        return len("\n".join(candidate)) <= max_chars

    def _format_recall_marker_line(self, entry: dict, marker: str) -> str:
        summary, _ = _compact_preview(entry.get("summary", ""), _RECALL_INDEX_LINE_CHARS)
        status = str(entry.get("status", "active") or "active").strip().lower() or "active"
        start = max(0, int(entry.get("source_turn_start", 0) or 0))
        end = max(start, int(entry.get("source_turn_end", start) or start))
        turns = f"turn {start}" if start == end else f"turns {start}-{end}"
        line = f"[{marker}][{status}] {summary} ({turns})"
        compact, _ = _compact_preview(line, _RECALL_INDEX_LINE_CHARS)
        return compact

    def _build_legacy_recall_lines(self, *, omitted_messages: list[dict]) -> list[str]:
        archived_user_topics: list[str] = []
        for msg in reversed(omitted_messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            snippet, _ = _compact_preview(content, _RECALL_INDEX_SNIPPET_CHARS)
            if snippet and snippet not in archived_user_topics:
                archived_user_topics.append(snippet)
            if len(archived_user_topics) >= _RECALL_INDEX_MAX_USER_TOPICS:
                break

        archived_tool_names: list[str] = []
        for msg in reversed(omitted_messages):
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function")
                if isinstance(fn, dict):
                    name = str(fn.get("name", "")).strip()
                else:
                    name = str(call.get("name", "")).strip()
                if name and name not in archived_tool_names:
                    archived_tool_names.append(name)
                if len(archived_tool_names) >= _RECALL_INDEX_MAX_TOOL_NAMES:
                    break
            if len(archived_tool_names) >= _RECALL_INDEX_MAX_TOOL_NAMES:
                break

        legacy: list[str] = []
        if archived_user_topics:
            legacy.append("- Archived user topics: " + " | ".join(archived_user_topics))
        if archived_tool_names:
            legacy.append("- Archived tool activity: " + ", ".join(archived_tool_names))
        return legacy

    def _build_recall_index_message(
        self,
        *,
        omitted_messages: list[dict],
        selected_messages: list[dict],
    ) -> dict | None:
        """Build a compact archive index that steers recall-tool usage."""
        if not omitted_messages:
            return None

        omitted_tool_messages = sum(
            1
            for msg in omitted_messages
            if str(msg.get("role", "")).strip() == "tool"
        )

        lines = [
            "[System: Compact archive index for omitted conversation history.]",
            f"- Omitted older messages: {len(omitted_messages)}",
            f"- Omitted tool-result messages: {omitted_tool_messages}",
            f"- Recent messages kept live: {len(selected_messages)}",
        ]
        if self._session_state.memory_index_last_indexed_turn > 0:
            lines.append(
                f"- Indexed through turn: {self._session_state.memory_index_last_indexed_turn}",
            )

        marker_sections = [
            ("Active DECISION", "DECISION", self._session_state.active_decisions),
            ("Open QUESTION", "OPEN_QUESTION", self._session_state.open_questions),
            ("Open PROPOSAL", "PROPOSAL", self._session_state.active_proposals),
            ("Recent RESEARCH", "RESEARCH", self._session_state.recent_research),
        ]
        has_marker_entries = any(bool(section[2]) for section in marker_sections)
        index_degraded = bool(self._session_state.memory_index_degraded)
        if index_degraded:
            lines.append("- Memory index status: degraded; using legacy archive snippets.")

        if has_marker_entries and not index_degraded:
            for section_title, marker, entries in marker_sections:
                if not entries:
                    continue
                section_header = f"- {section_title}:"
                if self._line_within_budget(
                    lines,
                    section_header,
                    max_chars=self._recall_index_max_chars,
                ):
                    lines.append(section_header)
                else:
                    continue
                for entry in list(entries)[: self._memory_index_section_limit]:
                    rendered = "  * " + self._format_recall_marker_line(entry, marker)
                    if not self._line_within_budget(
                        lines,
                        rendered,
                        max_chars=self._recall_index_max_chars,
                    ):
                        break
                    lines.append(rendered)
        else:
            for legacy_line in self._build_legacy_recall_lines(
                omitted_messages=omitted_messages,
            ):
                if not self._line_within_budget(
                    lines,
                    legacy_line,
                    max_chars=self._recall_index_max_chars,
                ):
                    break
                lines.append(legacy_line)

        action_lines = [
            "- When older details are needed, use conversation_recall first.",
            "- Recall actions:",
            '  * {"action":"decision_context","topic":"<topic>","limit":5}',
            '  * {"action":"entries","entry_type":"decision","status":"active","limit":5}',
            '  * {"action":"open_questions","topic":"<topic>","limit":5}',
            '  * {"action":"source_turns","entry_ids":[<id>],"limit":6}',
            '  * {"action":"search","query":"<keywords>","limit":5}',
        ]
        for line in action_lines:
            if not self._line_within_budget(lines, line, max_chars=self._recall_index_max_chars):
                break
            lines.append(line)

        content = "\n".join(lines)
        compact_content, _ = _compact_preview(content, self._recall_index_max_chars)
        return {"role": "system", "content": compact_content}

    def _context_window(self) -> list[dict]:
        """Return the messages to send to the model."""
        window, _stats = self._context_window_with_stats()
        return window

    def _context_window_with_stats(self) -> tuple[list[dict], _ContextWindowStats]:
        """Return model context plus lightweight telemetry stats."""
        if not self._messages:
            return [], _ContextWindowStats()

        # Always include system prompt
        system = []
        rest = self._messages
        if self._messages[0]["role"] == "system":
            system = [self._messages[0]]
            rest = self._messages[1:]

        system_tokens = sum(_estimate_message_tokens(m) for m in system)
        effective_context_cap = int(self._max_context_tokens)
        budget = max(
            1,
            effective_context_cap - system_tokens - _CONTEXT_OUTPUT_RESERVE_TOKENS,
        )
        recent_message_cap = max(
            8,
            min(int(self._max_context), _CONTEXT_RECENT_MESSAGE_CAP),
        )

        # Walk backward, adding messages until budget is exhausted
        selected: list[dict] = []
        used = 0
        for msg in reversed(rest):
            if len(selected) >= recent_message_cap:
                break
            msg_tokens = _estimate_message_tokens(msg)
            if msg_tokens > budget:
                # One oversized message (typically a large tool result) should
                # not collapse the entire context window.
                continue
            if used + msg_tokens > budget:
                break
            selected.append(msg)
            used += msg_tokens

        # Restore chronological order
        selected.reverse()

        # Ensure we don't start on an orphaned tool result
        while selected and selected[0].get("role") == "tool":
            selected.pop(0)

        omitted_messages = rest[: max(0, len(rest) - len(selected))]
        recall_index_used = False
        recall_index = self._build_recall_index_message(
            omitted_messages=omitted_messages,
            selected_messages=selected,
        )
        if recall_index is not None:
            recall_tokens = _estimate_message_tokens(recall_index)
            while selected and (used + recall_tokens) > budget:
                removed = selected.pop(0)
                used = max(0, used - _estimate_message_tokens(removed))
                while selected and selected[0].get("role") == "tool":
                    removed_tool = selected.pop(0)
                    used = max(0, used - _estimate_message_tokens(removed_tool))

            if recall_tokens <= max(1, budget - used):
                selected = [recall_index, *selected]
                recall_index_used = True

        final_window = self._sanitize_tool_call_sequence(system + selected)
        omitted_count = max(0, len(rest) - (len(selected) - (1 if recall_index_used else 0)))
        context_tokens = sum(_estimate_message_tokens(msg) for msg in final_window)
        stats = _ContextWindowStats(
            context_tokens=context_tokens,
            context_messages=len(final_window),
            omitted_messages=omitted_count,
            recall_index_used=recall_index_used,
        )
        return final_window, stats

    @staticmethod
    def _extract_tool_call_ids(tool_calls: list[dict]) -> list[str]:
        """Extract normalized tool_call IDs from an assistant tool-calls payload."""
        ids: list[str] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id", "") or "").strip()
            if call_id:
                ids.append(call_id)
        return ids

    @classmethod
    def _sanitize_tool_call_sequence(cls, messages: list[dict]) -> list[dict]:
        """Drop malformed tool-call chains that would fail provider validation.

        Providers require assistant messages containing `tool_calls` to be
        immediately followed by tool messages for each declared call ID.
        Interrupted turns can leave dangling assistant tool_calls or orphan tool
        messages in persisted history. This sanitizer repairs context before model
        invocation without mutating archived turns.
        """
        if not messages:
            return []

        sanitized: list[dict] = []
        idx = 0
        total = len(messages)

        while idx < total:
            msg = messages[idx]
            if not isinstance(msg, dict):
                idx += 1
                continue

            role = str(msg.get("role", "")).strip()
            if role == "assistant":
                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    expected_ids = cls._extract_tool_call_ids(tool_calls)
                    next_idx = idx + 1
                    contiguous_tools: list[dict] = []
                    seen_ids: set[str] = set()
                    while next_idx < total:
                        candidate = messages[next_idx]
                        if not isinstance(candidate, dict):
                            break
                        if str(candidate.get("role", "")).strip() != "tool":
                            break
                        contiguous_tools.append(candidate)
                        tool_call_id = str(
                            candidate.get("tool_call_id", "") or "",
                        ).strip()
                        if tool_call_id:
                            seen_ids.add(tool_call_id)
                        next_idx += 1

                    has_complete_chain = (
                        not expected_ids
                        or all(call_id in seen_ids for call_id in expected_ids)
                    )
                    if has_complete_chain:
                        sanitized.append(msg)
                        sanitized.extend(contiguous_tools)
                    else:
                        repaired = dict(msg)
                        repaired.pop("tool_calls", None)
                        sanitized.append(repaired)
                    idx = next_idx
                    continue

                sanitized.append(msg)
                idx += 1
                continue

            if role == "tool":
                # Orphan tool messages without an immediately preceding assistant
                # tool-calls chain break provider validation.
                idx += 1
                continue

            sanitized.append(msg)
            idx += 1

        return sanitized

    async def _execute_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        *,
        tool_call_id: str = "",
        caller_tool_name: str = "",
    ) -> tuple[ToolResult, int]:
        """Execute one tool call with approval + context handling."""
        approval_args = self._prepare_tool_execute_arguments(
            tool_name,
            arguments,
            tool_call_id=tool_call_id,
            caller_tool_name=caller_tool_name or tool_name,
            include_delegate_callback=False,
        )
        if self._approver is not None:
            decision = await self._approver.check(tool_name, approval_args)
            if decision == ApprovalDecision.DENY:
                return ToolResult.fail(f"Tool call '{tool_name}' denied by user."), 0

        execute_args = self._prepare_tool_execute_arguments(
            tool_name,
            arguments,
            tool_call_id=tool_call_id,
            caller_tool_name=caller_tool_name or tool_name,
            include_delegate_callback=True,
        )
        if (
            self._approver is not None
            and bool(execute_args.get("_loom_require_explicit_approval", False))
        ):
            if tool_name == "shell_execute":
                execute_args["_loom_high_risk_confirmed"] = True
            if tool_name == "wp_cli":
                execute_args["confirm_high_risk"] = True
        start = time.monotonic()
        result = await self._tools.execute(
            tool_name,
            execute_args,
            workspace=self._workspace,
            scratch_dir=self._scratch_dir,
            auth_context=self._auth_context,
            allow_internal_args=True,
            execution_surface="tui",
        )
        if (
            not result.success
            and self._tool_exposure_mode == "hybrid"
            and isinstance(result.error, str)
            and result.error.startswith("Unknown tool:")
            and "list_tools" not in result.error
        ):
            result = ToolResult(
                success=False,
                output=result.output,
                content_blocks=result.content_blocks,
                data=result.data,
                files_changed=list(result.files_changed),
                error=(
                    f"{result.error} Use list_tools to discover available tools, "
                    "then call run_tool with the selected name and arguments."
                ),
            )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return result, elapsed_ms

    def _prepare_tool_execute_arguments(
        self,
        tool_name: str,
        arguments: dict,
        *,
        tool_call_id: str = "",
        caller_tool_name: str = "",
        include_delegate_callback: bool = True,
    ) -> dict:
        """Inject runtime-only knobs used by selected tools."""
        execute_args = self._sanitize_runtime_arguments(tool_name, dict(arguments or {}))
        normalized_call_id = str(tool_call_id or "").strip()
        caller_name = str(caller_tool_name or tool_name or "").strip()

        if tool_name == "run_tool":
            delegated_args = execute_args.get("arguments")
            if isinstance(delegated_args, dict):
                delegated_tool_name = str(execute_args.get("name", "") or "").strip()
                delegated_copy = self._sanitize_runtime_arguments(
                    delegated_tool_name,
                    delegated_args,
                )
                if normalized_call_id:
                    delegated_copy["_loom_parent_tool_call_id"] = normalized_call_id
                    delegated_copy["_loom_parent_tool_name"] = caller_name or "run_tool"
                execute_args["arguments"] = delegated_copy

        if (
            tool_name == "delegate_task"
            and include_delegate_callback
            and callable(self._delegate_progress_callback)
        ):
            existing_callback = execute_args.get("_progress_callback")
            delegate_callback = self._delegate_progress_callback

            def _wrapped_progress_callback(raw_payload: dict | None) -> None:
                payload = dict(raw_payload) if isinstance(raw_payload, dict) else {}
                if normalized_call_id:
                    payload.setdefault("tool_call_id", normalized_call_id)
                payload.setdefault("tool_name", "delegate_task")
                if caller_name:
                    payload.setdefault("caller_tool_name", caller_name)
                try:
                    maybe = delegate_callback(payload)
                    if inspect.isawaitable(maybe):
                        asyncio.create_task(maybe)
                except Exception:
                    logger.debug("Delegate progress callback failed", exc_info=True)

                if callable(existing_callback):
                    try:
                        maybe_existing = existing_callback(payload)
                        if inspect.isawaitable(maybe_existing):
                            asyncio.create_task(maybe_existing)
                    except Exception:
                        logger.debug(
                            "Existing delegate progress callback failed",
                            exc_info=True,
                        )

            execute_args["_progress_callback"] = _wrapped_progress_callback

        if tool_name in {"web_fetch", "web_fetch_html"}:
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

        if tool_name == "shell_execute":
            command = str(execute_args.get("command", "") or "")
            risk_info = high_risk_command_metadata(command)
            if isinstance(risk_info, dict):
                execute_args["_loom_require_explicit_approval"] = True
                execute_args["_loom_risk_info"] = risk_info

        if tool_name == "wp_cli":
            group = str(execute_args.get("group", "") or "").strip().lower()
            action = str(execute_args.get("action", "") or "").strip().lower()
            wp_args = execute_args.get("args")
            assessment = assess_wp_cli_risk(
                group=group,
                action=action,
                args=wp_args if isinstance(wp_args, dict) else {},
            )
            if assessment is not None and self._wp_high_risk_confirmation_enabled():
                execute_args["_loom_require_explicit_approval"] = True
                execute_args["_loom_risk_info"] = format_wp_risk_info(assessment)
        return execute_args

    @staticmethod
    def _sanitize_runtime_arguments(tool_name: str, raw_args: dict) -> dict:
        """Remove internal runtime controls from model/tool-provided arguments."""
        cleaned: dict = {}
        for key, value in raw_args.items():
            key_text = str(key or "")
            if key_text.startswith(_RUNTIME_INTERNAL_PREFIX):
                if (
                    str(tool_name or "").strip() == "run_tool"
                    and key_text in _RUN_TOOL_PARENT_KEYS
                ):
                    cleaned[key_text] = value
                continue
            cleaned[key] = value

        # High-risk confirmation for wp_cli is set internally only after approval.
        if str(tool_name or "").strip() == "wp_cli":
            cleaned.pop("confirm_high_risk", None)
        return cleaned

    def _wp_high_risk_confirmation_enabled(self) -> bool:
        """Check wp_cli runtime policy toggle from configured tool instance."""
        tool = self._tools.get("wp_cli")
        if tool is None:
            return True
        return bool(getattr(tool, "high_risk_requires_confirmation", True))

    def _maybe_trim(self) -> None:
        """Trim in-memory message cache if very large.

        This does NOT lose data — all messages are persisted in the
        conversation store.  This just keeps RAM usage bounded.
        Ensures trim doesn't break message sequences.
        """
        max_cache = self._max_context * 3
        if len(self._messages) <= max_cache:
            return

        system = []
        if self._messages and self._messages[0]["role"] == "system":
            system = [self._messages[0]]

        kept = self._messages[-(self._max_context * 2):]
        # Don't start on an orphaned tool result
        while kept and kept[0].get("role") == "tool":
            kept.pop(0)

        self._messages = system + kept

    # ------------------------------------------------------------------
    # Dangling reference detection
    # ------------------------------------------------------------------

    def _maybe_recall_hint(self, user_message: str) -> str | None:
        """If the user references earlier context not in the window, nudge the model."""
        lower = user_message.lower()
        if any(phrase in lower for phrase in _DANGLING_REF_INDICATORS):
            return (
                "[System: The user may be referencing earlier conversation "
                "that is no longer in your context window. Use the "
                "conversation_recall tool to search for relevant context "
                "before proceeding.]"
            )
        return None

    @staticmethod
    def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in keywords)

    @staticmethod
    def _is_small_talk(tokens: set[str]) -> bool:
        if not tokens:
            return False
        if len(tokens) > 3:
            return False
        return tokens.issubset(_GREETING_TOKENS)

    @staticmethod
    def _mcp_tools_for_message(
        message: str,
        *,
        tokens: set[str],
        available: set[str],
    ) -> list[str]:
        """Return MCP tools that are likely relevant to the user message."""
        if not message:
            return []
        has_mcp_hint = "mcp" in tokens
        matches: list[str] = []
        for name in sorted(available):
            if not name.startswith("mcp."):
                continue
            lowered = name.lower()
            if lowered in message:
                matches.append(name)
                continue

            parts = lowered.split(".")
            if len(parts) < 3:
                continue
            alias = parts[1]
            tool_tokens = {
                token
                for token in re.split(r"[._/-]+", ".".join(parts[2:]))
                if token
            }
            if alias in tokens:
                matches.append(name)
                continue
            if has_mcp_hint and tokens.intersection(tool_tokens):
                matches.append(name)
        return matches

    def _should_retry_with_hybrid_fallback(
        self,
        *,
        user_message: str,
        response_text: str,
        turn_tool_schemas: list[dict],
        recovery_used: bool,
    ) -> bool:
        """Detect hybrid tool stall and inject one fallback-nudge retry."""
        if self._tool_exposure_mode != "hybrid" or recovery_used:
            return False

        schema_names = {
            str(schema.get("name", "")).strip()
            for schema in turn_tool_schemas
            if isinstance(schema, dict)
        }
        if not _HYBRID_FALLBACK_TOOLS.issubset(schema_names):
            return False

        message = " ".join(str(user_message or "").lower().split())
        response = " ".join(str(response_text or "").lower().split())
        if not message or not response:
            return False

        tokens = set(re.findall(r"[a-z0-9_./+-]+", message))
        available = set(
            self._tools.list_tools(
                auth_context=self._auth_context,
                execution_surface="tui",
            ),
        )
        mcp_aliases = {
            parts[1]
            for name in available
            if name.startswith("mcp.")
            for parts in [name.lower().split(".")]
            if len(parts) >= 3 and parts[1]
        }
        response_has_integration_signal = (
            "mcp" in response
            or "tool set" in response
            or "direct tool" in response
        )
        has_integration_hint = (
            "mcp" in tokens
            or bool(tokens.intersection(mcp_aliases))
            or response_has_integration_signal
        )
        if not has_integration_hint:
            return False

        return any(marker in response for marker in _HYBRID_TOOL_STALL_MARKERS)

    def _tool_names_for_turn(self, user_message: str) -> list[str]:
        """Choose intent-aware typed tool names for this turn (no fallback lane)."""
        available = set(
            self._tools.list_tools(
                auth_context=self._auth_context,
                execution_surface="tui",
            ),
        )
        if not available:
            return []

        message = " ".join(str(user_message or "").lower().split())
        tokens = set(re.findall(r"[a-z0-9_./+-]+", message))

        selected: set[str] = set(_CORE_TOOL_NAMES)

        if self._contains_any(message, _CODING_KEYWORDS):
            selected.update(_CODING_TOOL_NAMES)
        if self._contains_any(message, _WEB_KEYWORDS):
            selected.update(_WEB_TOOL_NAMES)
        if self._contains_any(message, _WRITING_KEYWORDS):
            selected.update(_WRITING_TOOL_NAMES)
        if self._contains_any(message, _FINANCE_KEYWORDS):
            selected.update(_FINANCE_TOOL_NAMES)
        if self._contains_any(message, _SPREADSHEET_KEYWORDS):
            selected.add("spreadsheet")
        if self._contains_any(message, _MATH_KEYWORDS):
            selected.add("calculator")

        explicit_mentions: list[str] = []
        # Honor explicit tool-name mentions from the user.
        for name in sorted(available):
            if name.lower() in message:
                selected.add(name)
                explicit_mentions.append(name)

        mcp_mentions = self._mcp_tools_for_message(
            message,
            tokens=tokens,
            available=available,
        )
        if mcp_mentions:
            selected.update(mcp_mentions)

        # If the message is not obvious small talk and no category matched,
        # include a compact general-purpose set instead of all tools.
        if selected == set(_CORE_TOOL_NAMES) and not self._is_small_talk(tokens):
            selected.update(_GENERAL_TOOL_NAMES)

        ordered_candidates = [
            *_CORE_TOOL_NAMES,
            *_GENERAL_TOOL_NAMES,
            *_CODING_TOOL_NAMES,
            *_WEB_TOOL_NAMES,
            *_WRITING_TOOL_NAMES,
            *_FINANCE_TOOL_NAMES,
            "spreadsheet",
            "calculator",
        ]

        selected_available = {name for name in selected if name in available}
        ordered: list[str] = []
        seen: set[str] = set()

        priority_mentions = [
            name
            for name in [*explicit_mentions, *mcp_mentions]
            if name in selected_available
        ]
        for name in priority_mentions:
            if name not in seen:
                ordered.append(name)
                seen.add(name)

        for name in [*_CORE_TOOL_NAMES, *ordered_candidates]:
            if name in selected_available and name not in seen:
                ordered.append(name)
                seen.add(name)

        for name in sorted(selected_available):
            if name not in seen:
                ordered.append(name)
                seen.add(name)
        return ordered

    def _tool_schemas_for_turn(self, user_message: str) -> list[dict]:
        """Return tool schemas for this turn according to exposure mode."""
        if self._tool_exposure_mode == "full":
            return self._tools.all_schemas(
                auth_context=self._auth_context,
                execution_surface="tui",
            )

        all_schemas = self._tools.all_schemas(
            auth_context=self._auth_context,
            execution_surface="tui",
        )
        if not all_schemas:
            return []

        schema_by_name: dict[str, dict] = {}
        for schema in all_schemas:
            name = str(schema.get("name", "")).strip()
            if not name:
                continue
            schema_by_name[name] = schema

        ordered_typed_names = self._tool_names_for_turn(user_message)
        typed_names: list[str] = []
        typed_bytes = 0
        for name in ordered_typed_names:
            schema = schema_by_name.get(name)
            if schema is None:
                continue
            if len(typed_names) >= _TYPED_TOOL_SCHEMA_CAP:
                break
            schema_size = len(
                json.dumps(
                    schema,
                    ensure_ascii=False,
                    default=str,
                ).encode("utf-8", errors="replace"),
            )
            if typed_names and (typed_bytes + schema_size) > _TYPED_TOOL_SCHEMA_BYTE_BUDGET:
                break
            typed_names.append(name)
            typed_bytes += schema_size

        if not typed_names:
            for fallback in _CORE_TOOL_NAMES:
                if fallback in schema_by_name:
                    typed_names.append(fallback)
                if len(typed_names) >= min(_TYPED_TOOL_SCHEMA_CAP, 4):
                    break

        final_names: list[str] = list(typed_names)
        if self._tool_exposure_mode == "hybrid":
            for fallback in _FALLBACK_TOOL_NAMES:
                if fallback in schema_by_name and fallback not in final_names:
                    final_names.append(fallback)

        schemas = [schema_by_name[name] for name in final_names if name in schema_by_name]
        if schemas:
            logger.debug(
                (
                    "cowork_tool_schema_selection mode=%s selected=%d typed=%d "
                    "total_available=%d names=%s"
                ),
                self._tool_exposure_mode,
                len(schemas),
                len(typed_names),
                len(schema_by_name),
                ",".join(final_names),
            )
            return schemas
        return all_schemas

    def _bind_hybrid_tools(self) -> None:
        """Bind session-aware callbacks for hybrid fallback tools."""
        list_tools_tool = self._tools.get("list_tools")
        if list_tools_tool is not None and hasattr(list_tools_tool, "bind"):
            try:
                list_tools_tool.bind(self._tool_catalog_rows)
            except Exception as e:
                logger.debug("Failed binding list_tools callback: %s", e)

        run_tool_tool = self._tools.get("run_tool")
        if run_tool_tool is not None and hasattr(run_tool_tool, "bind"):
            try:
                run_tool_tool.bind(self._dispatch_run_tool)
            except Exception as e:
                logger.debug("Failed binding run_tool callback: %s", e)

    @staticmethod
    def _tool_category(name: str) -> str:
        if name in _CORE_TOOL_NAMES or name in _FALLBACK_TOOL_NAMES:
            return "core"
        if name in _CODING_TOOL_NAMES or name in {
            "read_file",
            "write_file",
            "edit_file",
            "move_file",
            "delete_file",
            "analyze_code",
        }:
            return "coding"
        if name in _WEB_TOOL_NAMES:
            return "web"
        if name in _WRITING_TOOL_NAMES:
            return "writing"
        if name in _FINANCE_TOOL_NAMES:
            return "finance"
        if name.startswith("mcp."):
            return "mcp"
        return "other"

    def _tool_catalog_rows(
        self,
        auth_context: Any | None = None,
        execution_surface: str = "tui",
    ) -> list[dict]:
        """Build a compact list-tools catalog from currently available schemas."""
        schemas = self._tools.all_schemas(
            auth_context=auth_context,
            execution_surface=execution_surface,
        )
        rows: list[dict] = []
        for schema in schemas:
            name = str(schema.get("name", "")).strip()
            if not name:
                continue
            tool = self._tools.get(name)
            description = str(schema.get("description", "") or "").strip()
            parameters = schema.get("parameters", {})
            auth_requirements = getattr(tool, "auth_requirements", [])
            if not isinstance(auth_requirements, list):
                auth_requirements = []
            rows.append({
                "name": name,
                "summary": " ".join(description.split()),
                "description": description,
                "parameters": parameters if isinstance(parameters, dict) else {},
                "mutates": bool(getattr(tool, "is_mutating", False)),
                "auth_mode": normalize_tool_auth_mode(
                    getattr(tool, "auth_mode", "no_auth"),
                ),
                "auth_required": tool_auth_required(tool),
                "auth_requirements": list(auth_requirements),
                "category": self._tool_category(name),
                "execution_surfaces": list(normalize_tool_execution_surfaces(
                    schema.get("x_supported_execution_surfaces", []),
                )),
            })
        rows.sort(key=lambda item: str(item.get("name", "")))
        return rows

    async def _dispatch_run_tool(
        self,
        tool_name: str,
        arguments: dict,
        ctx: Any,
    ) -> ToolResult:
        """Execute a delegated tool call from run_tool with safety parity."""
        target = str(tool_name or "").strip()
        if not target:
            return ToolResult.fail("run_tool requires a non-empty tool name.")
        if target == "run_tool":
            return ToolResult.fail("run_tool cannot invoke itself.")
        if target in _USER_INTERACTION_TOOLS:
            return ToolResult.fail(
                f"Tool '{target}' must be called directly and cannot be delegated via run_tool.",
            )
        if not isinstance(arguments, dict):
            return ToolResult.fail("run_tool 'arguments' must be an object.")
        execute_input = dict(arguments)
        parent_tool_call_id = str(
            execute_input.pop("_loom_parent_tool_call_id", "") or "",
        ).strip()
        parent_tool_name = str(
            execute_input.pop("_loom_parent_tool_name", "") or "",
        ).strip()

        auth_context = getattr(ctx, "auth_context", self._auth_context)
        if not self._tools.has(
            target,
            auth_context=auth_context,
            execution_surface="tui",
        ):
            return ToolResult.fail(f"Unknown tool: {target}")

        approval_args = self._prepare_tool_execute_arguments(
            target,
            execute_input,
            tool_call_id=parent_tool_call_id,
            caller_tool_name=parent_tool_name or "run_tool",
            include_delegate_callback=False,
        )
        if self._approver is not None:
            decision = await self._approver.check(target, approval_args)
            if decision == ApprovalDecision.DENY:
                return ToolResult.fail(f"Tool call '{target}' denied by user.")

        execute_args = self._prepare_tool_execute_arguments(
            target,
            execute_input,
            tool_call_id=parent_tool_call_id,
            caller_tool_name=parent_tool_name or "run_tool",
            include_delegate_callback=True,
        )
        if (
            self._approver is not None
            and bool(execute_args.get("_loom_require_explicit_approval", False))
        ):
            if target == "shell_execute":
                execute_args["_loom_high_risk_confirmed"] = True
            if target == "wp_cli":
                execute_args["confirm_high_risk"] = True
        return await self._tools.execute(
            target,
            execute_args,
            workspace=self._workspace,
            scratch_dir=self._scratch_dir,
            auth_context=auth_context,
            allow_internal_args=True,
            execution_surface="tui",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_content(self) -> str:
        """Build the full system prompt with session state and learned behaviors."""
        parts = [self._static_system_prompt]
        if self._behaviors_section:
            parts.append(f"\n{self._behaviors_section}")
        if self._session_state and self._session_state.turn_count > 0:
            parts.append(f"\n## Session State\n{self._session_state.to_yaml()}")
        return "\n".join(parts)

    async def _reflect(self, user_message: str, assistant_response: str) -> None:
        """Run gap analysis on this user-assistant exchange.

        Best-effort: failures are logged but never surface to the user.
        Updates the behaviors section for system prompt injection.
        """
        if self._reflection is None:
            return
        try:
            await self._reflection.on_turn_complete(
                user_message=user_message,
                assistant_response=assistant_response,
                session_id=self._session_id,
            )
            # Refresh the behaviors section from all accumulated patterns
            await self._load_behaviors()
        except Exception as e:
            logger.debug("Gap analysis failed (non-fatal): %s", e)

    async def _load_behaviors(self) -> None:
        """Load learned behavioral patterns and format for prompt injection."""
        if self._reflection is None:
            return
        try:
            from loom.learning.reflection import (
                format_behaviors_for_prompt,
                get_learned_behaviors,
            )
            patterns = await get_learned_behaviors(self._reflection._learning)
            self._behaviors_section = format_behaviors_for_prompt(patterns)
        except Exception as e:
            logger.debug("Failed to load behaviors: %s", e)

    def _update_system_message(self) -> None:
        """Update the system message in-place with current session state."""
        if self._messages and self._messages[0]["role"] == "system":
            self._messages[0]["content"] = self._build_system_content()

    async def _append_tool_result(
        self, tool_call_id: str, tool_name: str, result: ToolResult,
    ) -> None:
        """Append a tool result message and persist it."""
        content = await self._serialize_tool_result_for_model(tool_name, result)
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })
        persisted_content = result.to_json()
        await self._persist_turn(
            "tool", content=persisted_content,
            tool_call_id=tool_call_id, tool_name=tool_name,
        )

    @staticmethod
    def _tool_batch_signature(tool_calls: list[ToolCall] | None) -> str:
        """Return a stable signature for one assistant tool-call batch."""
        if not tool_calls:
            return ""
        normalized: list[dict[str, Any]] = []
        for tc in tool_calls:
            normalized.append({
                "name": str(getattr(tc, "name", "") or "").strip(),
                "arguments": dict(getattr(tc, "arguments", {}) or {}),
            })
        return json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def _build_repeated_tool_batch_fallback(
        self,
        tool_events: list[ToolCallEvent],
    ) -> str:
        """Return a deterministic user-facing fallback when tool loops stall."""
        last_completed = next(
            (
                event
                for event in reversed(tool_events)
                if event.result is not None
            ),
            None,
        )
        if last_completed and last_completed.result:
            output_preview, _ = _compact_preview(
                last_completed.result.output,
                280,
            )
            if output_preview:
                return (
                    "I stopped because the model repeated identical tool calls without "
                    "making progress. "
                    f"Latest {last_completed.name} result: {output_preview}"
                )
            if last_completed.result.error:
                error_preview, _ = _compact_preview(
                    last_completed.result.error,
                    220,
                )
                return (
                    "I stopped because the model repeated identical tool calls without "
                    "making progress. "
                    f"Latest {last_completed.name} error: {error_preview}"
                )
        return (
            "I stopped because the model repeated identical tool calls without making "
            "progress. I can continue once we change the query or tool arguments."
        )

    @staticmethod
    def _tool_output_limit(tool_name: str) -> int:
        if tool_name in _HEAVY_OUTPUT_TOOLS:
            return HEAVY_TOOL_RESULT_OUTPUT_CHARS
        return DEFAULT_TOOL_RESULT_OUTPUT_CHARS

    async def _compact_text(self, text: str, *, max_chars: int, label: str) -> str:
        return await self._compactor.compact(
            str(text or ""),
            max_chars=max_chars,
            label=label,
        )

    async def _summarize_tool_data(self, data: dict | None) -> dict | None:
        if not isinstance(data, dict) or not data:
            return None

        if len(data) > 12:
            packed = json.dumps(data, ensure_ascii=False, default=str)
            summary_text = await self._compact_text(
                packed,
                max_chars=900,
                label="cowork tool data payload",
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
                    max_chars=180,
                    label=f"cowork tool data {key}",
                )
            elif isinstance(value, (int, float, bool)) or value is None:
                summary[key] = value
            elif isinstance(value, (list, dict)):
                packed = json.dumps(value, ensure_ascii=False, default=str)
                summary[key] = await self._compact_text(
                    packed,
                    max_chars=220,
                    label=f"cowork tool data {key}",
                )
            else:
                summary[key] = str(type(value).__name__)
        return summary or None

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
                        label=f"cowork content block {key}",
                    )
            serialized_blocks.append(compact)
        return serialized_blocks or None

    async def _serialize_tool_result_for_model(
        self,
        tool_name: str,
        result: ToolResult,
    ) -> str:
        limit = self._tool_output_limit(tool_name)
        output_text = await self._compact_text(
            result.output,
            max_chars=limit,
            label=f"cowork {tool_name} tool output",
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
                label=f"cowork {tool_name} files changed",
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

    @staticmethod
    def _tool_calls_to_dicts(tool_calls: list[ToolCall]) -> list[dict]:
        """Convert ToolCall objects to OpenAI message format dicts."""
        return [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in tool_calls
        ]

    async def _persist_turn(
        self,
        role: str,
        content: str | None = None,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
    ) -> None:
        """Write-through: persist a turn to the conversation store."""
        if self._store is None or not self._session_id:
            return
        try:
            async with self._counter_lock:
                self._message_counter += 1
                counter = self._message_counter
            await self._store.append_turn(
                session_id=self._session_id,
                turn_number=counter,
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
            if self._memory_indexer is not None:
                self._memory_indexer.enqueue_up_to_turn(counter)
        except Exception as e:
            logger.warning("Persist turn failed: %s", e)

    async def _persist_session_metadata(self) -> None:
        """Persist session state and token counts."""
        if self._store is None or not self._session_id:
            return
        try:
            await self._store.update_session(
                session_id=self._session_id,
                total_tokens=self._total_tokens,
                turn_count=self._turn_counter,
                session_state=self._session_state.to_dict(),
            )
        except Exception as e:
            logger.warning("Persist metadata failed: %s", e)


def build_cowork_system_prompt(workspace: Path | None = None) -> str:
    """Build the system prompt for cowork mode."""
    workspace_info = f"Workspace: {workspace}" if workspace else "No workspace set."

    return f"""\
You are a collaborative assistant for complex tasks, working interactively with
the user. You can support software development, research, analysis, planning,
and operations tasks.
You have access to tools for reading, writing, editing, and searching files,
running shell commands, executing git operations, and fetching web content.

{workspace_info}

GUIDELINES:
- Start by understanding the user's requested outcome and constraints.
- For coding tasks, read files before editing them and understand existing code.
- For coding tasks, use targeted edits (edit_file) rather than rewriting files.
- Keep changes minimal and focused on what was requested.
- If something is unclear, use the ask_user tool to ask for clarification.
- Show your work: explain what you're doing and why.
- When you encounter errors, investigate and fix them rather than giving up.
- Do NOT modify files outside the workspace directory unless explicitly directed.
- Do NOT fabricate file contents or tool outputs.

TOOL USAGE:
- Use glob_find to discover files by pattern (fast).
- Use ripgrep_search for content search (much faster than search_files).
- Use web_search to find documentation, solutions, or package information online.
- Use web_fetch to read a specific URL's content.
- Use web_fetch_html when you explicitly need raw page source markup.
- Use shell_execute for running tests, builds, linters, etc.
- Use git_command for version control operations (including push).
- Use task_tracker to organize multi-step work and show progress.
- Use ask_user when you need the developer's input or decision.
- In hybrid mode when a needed tool is not directly typed:
  1) call list_tools with {{"detail":"compact"}} to discover tool names
  2) call list_tools with {{"detail":"schema","query":"<tool name>"}} for argument schema
  3) call run_tool with the exact tool name and JSON arguments
  Do not use list_tools schema globally; always narrow with query/category/filters.

CONVERSATION HISTORY:
- Your context window contains only recent turns. The full session history
  is preserved in the archive and is never lost.
- Use conversation_recall to search for earlier context when:
  - The user references something discussed earlier
  - You need information from a previous tool call
  - You're unsure about a prior decision or constraint
- When your context doesn't contain enough information to act confidently,
  search the archive before guessing or asking the user to repeat themselves.

TASK DELEGATION:
- Use delegate_task for complex multi-step work that benefits from planning,
  decomposition, verification, or parallel execution.
- delegate_task submits work to the orchestration engine which will plan,
  execute subtasks, verify results, and return a summary.
- Use delegate_task when: multi-file refactoring, research pipelines,
  cross-source analysis, strategy/report generation, or any work that benefits
  from structured decomposition.
- Use direct tools when: reading/editing a single file, running a command,
  quick checks, focused research, or anything simple and straightforward."""
