"""Conversation recall tool: search and retrieve from the session archive.

The model calls this to access past conversation that has fallen out of
the context window. The full history is in SQLite and never discarded.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from loom.engine.semantic_compactor import SemanticCompactor
from loom.tools.registry import Tool, ToolContext, ToolResult

if TYPE_CHECKING:
    from loom.state.conversation_store import ConversationStore


MAX_OUTPUT_CHARS = 32_000
MAX_TURN_CONTENT_CHARS = 2_200
MAX_MEMORY_ENTRY_FIELD_CHARS = 260
MAX_MEMORY_RECALL_LIMIT = 50
_V2_ACTIONS = frozenset({
    "entries",
    "decision_context",
    "timeline",
    "open_questions",
    "source_turns",
})
logger = logging.getLogger(__name__)


class ConversationRecallTool(Tool):
    """Search and retrieve past conversation/messages from session archive."""

    name = "conversation_recall"
    description = (
        "Search conversation history and typed cowork memory. "
        "Actions: search, range, tool_calls, summary, entries, "
        "decision_context, timeline, open_questions, source_turns."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "search",
                    "range",
                    "tool_calls",
                    "summary",
                    "entries",
                    "decision_context",
                    "timeline",
                    "open_questions",
                    "source_turns",
                ],
                "description": (
                    "search/range/tool_calls/summary are legacy actions. "
                    "entries/decision_context/timeline/open_questions/source_turns "
                    "return typed cowork memory context."
                ),
            },
            "query": {
                "type": "string",
                "description": "Keyword query (search or entries).",
            },
            "tool_name": {
                "type": "string",
                "description": "Tool name for tool_calls action.",
            },
            "start_turn": {
                "type": "integer",
                "description": "Start of turn range (range/source_turns).",
            },
            "end_turn": {
                "type": "integer",
                "description": "End of turn range, inclusive (range/source_turns).",
            },
            "entry_type": {
                "type": "string",
                "description": "Typed memory filter (entries action).",
            },
            "status": {
                "type": "string",
                "description": "Typed memory status filter (entries action).",
            },
            "topic": {
                "type": "string",
                "description": "Topic filter (entries/decision_context/timeline/open_questions).",
            },
            "entry_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Memory entry ids (source_turns).",
            },
            "include_resolved": {
                "type": "boolean",
                "description": "Include non-active decisions/open questions.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum rows to return.",
            },
        },
        "required": ["action"],
    }

    @property
    def timeout_seconds(self) -> int:
        # Larger recall slices can trigger semantic compaction, so the default
        # 30s tool timeout is too aggressive for legitimate requests.
        return 90

    def __init__(
        self,
        store: ConversationStore | None = None,
        session_id: str = "",
        session_state: object | None = None,
        compactor: SemanticCompactor | None = None,
        *,
        v2_actions_enabled: bool = True,
        force_fts: bool = False,
    ):
        self._store = store
        self._session_id = session_id
        self._session_state = session_state
        self._compactor = compactor
        self._v2_actions_enabled = bool(v2_actions_enabled)
        self._force_fts = bool(force_fts)

    def bind(
        self,
        store: ConversationStore,
        session_id: str,
        session_state: object | None = None,
        compactor: SemanticCompactor | None = None,
        *,
        v2_actions_enabled: bool | None = None,
        force_fts: bool | None = None,
    ) -> None:
        """Bind to a specific session after construction."""
        self._store = store
        self._session_id = session_id
        self._session_state = session_state
        self._compactor = compactor
        if v2_actions_enabled is not None:
            self._v2_actions_enabled = bool(v2_actions_enabled)
        if force_fts is not None:
            self._force_fts = bool(force_fts)

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if self._store is None or not self._session_id:
            return ToolResult.fail(
                "Conversation recall is not available (no active session)."
            )

        action = str(args.get("action", "") or "").strip()
        limit = self._safe_limit(args.get("limit", 10), default=10)
        logger.info(
            "cowork_memory_recall_query session=%s action=%s limit=%s",
            self._session_id,
            action or "<empty>",
            limit,
        )

        if action in _V2_ACTIONS and not self._v2_actions_enabled:
            return ToolResult.fail(
                "Recall v2 actions are disabled for this session. "
                "Use search/range/tool_calls/summary instead."
            )

        result: ToolResult
        if action == "search":
            result = await self._search(args.get("query", ""), limit)
        elif action == "range":
            result = await self._range(
                args.get("start_turn", 0),
                args.get("end_turn", 0),
            )
        elif action == "tool_calls":
            result = await self._tool_calls(args.get("tool_name", ""), limit)
        elif action == "summary":
            result = self._summary()
        elif action == "entries":
            result = await self._entries(args, limit)
        elif action == "decision_context":
            result = await self._decision_context(args, limit)
        elif action == "timeline":
            result = await self._timeline(args, limit)
        elif action == "open_questions":
            result = await self._open_questions(args, limit)
        elif action == "source_turns":
            result = await self._source_turns(args, limit)
        else:
            result = ToolResult.fail(f"Unknown action: {action}")

        fallback = bool(result.output and "raw_turns_fallback" in result.output)
        logger.info(
            (
                "cowork_memory_recall_result session=%s action=%s "
                "success=%s output_chars=%s fallback=%s"
            ),
            self._session_id,
            action or "<empty>",
            result.success,
            len(result.output or ""),
            fallback,
        )
        if fallback:
            logger.info(
                "cowork_memory_recall_fallback session=%s action=%s",
                self._session_id,
                action or "<empty>",
            )
        return result

    async def _search(self, query: str, limit: int) -> ToolResult:
        query_text = str(query or "").strip()
        if not query_text:
            return ToolResult.fail("'query' parameter is required for search.")

        rows = await self._store.search_turns(
            self._session_id,
            query_text,
            limit=min(limit, MAX_MEMORY_RECALL_LIMIT),
        )
        if not rows:
            return ToolResult.ok(f"No messages found matching '{query_text}'.")
        return ToolResult.ok(await self._format_turns(rows))

    async def _range(self, start: int, end: int) -> ToolResult:
        start_turn = self._safe_int(start, 0)
        end_turn = self._safe_int(end, 0)
        if end_turn < start_turn:
            return ToolResult.fail("end_turn must be >= start_turn.")

        rows = await self._store.get_turn_range(self._session_id, start_turn, end_turn)
        if not rows:
            return ToolResult.ok(f"No turns found in range {start_turn}-{end_turn}.")
        return ToolResult.ok(await self._format_turns(rows))

    async def _tool_calls(self, tool_name: str, limit: int) -> ToolResult:
        normalized = str(tool_name or "").strip()
        if not normalized:
            return ToolResult.fail("'tool_name' parameter is required for tool_calls.")

        rows = await self._store.search_tool_calls(
            self._session_id,
            normalized,
            limit=min(limit, MAX_MEMORY_RECALL_LIMIT),
        )
        if not rows:
            return ToolResult.ok(f"No calls to '{normalized}' found in archive.")
        return ToolResult.ok(await self._format_turns(rows))

    def _summary(self) -> ToolResult:
        if self._session_state and hasattr(self._session_state, "to_yaml"):
            return ToolResult.ok(self._session_state.to_yaml())
        return ToolResult.ok("No session state available.")

    async def _entries(self, args: dict, limit: int) -> ToolResult:
        query = str(args.get("query", "") or "").strip()
        entry_type = str(args.get("entry_type", "") or "").strip()
        status = str(args.get("status", "") or "").strip()
        topic = str(args.get("topic", "") or "").strip()

        rows = await self._store.search_cowork_memory_entries(
            self._session_id,
            query=query,
            entry_type=entry_type,
            status=status,
            topic=topic,
            limit=min(limit, MAX_MEMORY_RECALL_LIMIT),
            force_fts=self._force_fts,
        )
        if rows:
            rendered = await self._format_entries(rows, source="index")
            return ToolResult.ok(rendered)

        if query:
            fallback_turns = await self._store.search_turns(
                self._session_id,
                query,
                limit=min(limit, 8),
            )
            if fallback_turns:
                rendered_turns = await self._format_turns(fallback_turns)
                return ToolResult.ok(
                    "No typed index entries matched. Source=raw_turns_fallback.\n\n"
                    f"{rendered_turns}"
                )
        return ToolResult.ok(
            "No typed entries matched. Try broader filters or query. "
            "Suggested actions: entries(topic=<term>), decision_context(topic=<term>), "
            "search(query=<keywords>)."
        )

    async def _decision_context(self, args: dict, limit: int) -> ToolResult:
        topic = str(args.get("topic", "") or "").strip()
        include_resolved = bool(args.get("include_resolved", False))
        status = "" if include_resolved else "active"
        rows = await self._store.search_cowork_memory_entries(
            self._session_id,
            query=topic,
            entry_type="decision",
            status=status,
            topic=topic,
            limit=min(limit, MAX_MEMORY_RECALL_LIMIT),
            force_fts=self._force_fts,
        )
        if rows:
            return ToolResult.ok(await self._format_entries(rows, source="index"))

        if topic:
            fallback_turns = await self._store.search_turns(
                self._session_id,
                topic,
                limit=min(limit, 10),
            )
            if fallback_turns:
                rendered_turns = await self._format_turns(fallback_turns)
                return ToolResult.ok(
                    "No indexed decisions matched; source=raw_turns_fallback.\n\n"
                    f"{rendered_turns}"
                )

        return ToolResult.ok(
            "No decision context found. Try entries(entry_type=proposal|decision) "
            "or timeline(topic=<term>)."
        )

    async def _timeline(self, args: dict, limit: int) -> ToolResult:
        topic = str(args.get("topic", "") or "").strip()
        rows = await self._store.get_cowork_memory_timeline(
            self._session_id,
            topic=topic,
            limit=min(limit, MAX_MEMORY_RECALL_LIMIT),
        )
        if not rows:
            return ToolResult.ok(
                "No memory timeline entries found. "
                "Try search(query=<keywords>) or entries(topic=<term>)."
            )
        return ToolResult.ok(await self._format_entries(rows, source="index"))

    async def _open_questions(self, args: dict, limit: int) -> ToolResult:
        topic = str(args.get("topic", "") or "").strip()
        include_resolved = bool(args.get("include_resolved", False))
        status = "" if include_resolved else "active"
        questions = await self._store.search_cowork_memory_entries(
            self._session_id,
            query=topic,
            entry_type="open_question",
            status=status,
            topic=topic,
            limit=min(limit, MAX_MEMORY_RECALL_LIMIT),
            force_fts=self._force_fts,
        )
        if not questions:
            return ToolResult.ok(
                "No open questions found. "
                "Try entries(entry_type=proposal,status=active) or decision_context()."
            )

        rendered: list[str] = []
        for item in questions:
            rendered.append(self._render_entry_line(item, source="index"))
            related = await self._store.search_cowork_memory_entries(
                self._session_id,
                query=item.get("topic", ""),
                entry_type="proposal",
                status="active",
                topic=item.get("topic", ""),
                limit=2,
                force_fts=self._force_fts,
            )
            for proposal in related:
                rendered.append("  RELATED " + self._render_entry_line(proposal, source="index"))
        content = "\n".join(rendered)
        return ToolResult.ok(await self._finalize_output(content))

    async def _source_turns(self, args: dict, limit: int) -> ToolResult:
        entry_ids_raw = args.get("entry_ids", [])
        entry_ids = self._coerce_int_list(entry_ids_raw)
        start_turn = self._safe_int(args.get("start_turn", 0), 0)
        end_turn = self._safe_int(args.get("end_turn", 0), 0)

        rows: list[dict] = []
        if entry_ids:
            entries = await self._store.get_cowork_memory_entries_by_ids(
                self._session_id,
                entry_ids=entry_ids,
                limit=min(limit, MAX_MEMORY_RECALL_LIMIT),
            )
            if not entries:
                return ToolResult.ok("No indexed entries found for supplied entry_ids.")
            for entry in entries:
                start = max(0, self._safe_int(entry.get("source_turn_start", 0), 0))
                end = max(start, self._safe_int(entry.get("source_turn_end", start), start))
                turns = await self._store.get_turn_range(self._session_id, start, end)
                rows.extend(turns)
        else:
            if end_turn < start_turn:
                return ToolResult.fail("end_turn must be >= start_turn.")
            if start_turn <= 0 and end_turn <= 0:
                return ToolResult.fail(
                    "source_turns requires entry_ids or start_turn/end_turn."
                )
            rows = await self._store.get_turn_range(self._session_id, start_turn, end_turn)

        if not rows:
            return ToolResult.ok("No source turns found for requested references.")
        return ToolResult.ok(await self._format_turns(rows[: max(1, min(limit * 2, 120))]))

    @staticmethod
    def _safe_limit(value: object, *, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(1, min(parsed, MAX_MEMORY_RECALL_LIMIT))

    @staticmethod
    def _safe_int(value: object, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _coerce_int_list(value: object) -> list[int]:
        if not isinstance(value, list):
            return []
        normalized: list[int] = []
        for item in value:
            try:
                val = int(item)
            except (TypeError, ValueError):
                continue
            if val > 0 and val not in normalized:
                normalized.append(val)
        return normalized

    @staticmethod
    def _normalize_entry_text(
        value: object,
        *,
        max_chars: int = MAX_MEMORY_ENTRY_FIELD_CHARS,
    ) -> str:
        compact = " ".join(str(value or "").split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3].rstrip() + "..."

    def _render_entry_line(self, entry: dict, *, source: str) -> str:
        entry_id = int(entry.get("id", 0) or 0)
        entry_type = str(entry.get("entry_type", "research") or "research").upper()
        status = str(entry.get("status", "active") or "active").lower()
        summary = self._normalize_entry_text(entry.get("summary", ""))
        start = max(0, self._safe_int(entry.get("source_turn_start", 0), 0))
        end = max(start, self._safe_int(entry.get("source_turn_end", start), start))
        turn_ref = f"turn {start}" if start == end else f"turns {start}-{end}"
        try:
            confidence_raw = float(entry.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence_raw = 0.0
        confidence = max(0.0, min(1.0, confidence_raw))

        line = (
            f"[{entry_type}][{status}] id={entry_id} "
            f"{summary} ({turn_ref}) confidence={confidence:.2f} source={source}"
        )
        rationale = self._normalize_entry_text(entry.get("rationale", ""), max_chars=180)
        evidence = self._normalize_entry_text(entry.get("evidence_excerpt", ""), max_chars=180)
        parts = [line]
        if rationale:
            parts.append(f"  RATIONALE: {rationale}")
        if evidence:
            parts.append(f"  EVIDENCE: {evidence}")
        return "\n".join(parts)

    async def _format_entries(self, rows: list[dict], *, source: str) -> str:
        rendered = "\n".join(self._render_entry_line(row, source=source) for row in rows)
        return await self._finalize_output(rendered)

    async def _format_turns(self, rows: list[dict]) -> str:
        rendered = await _format_turns(rows, compactor=self._compactor)
        return await self._finalize_output(rendered)

    async def _finalize_output(self, text: str) -> str:
        rendered = str(text or "")
        if self._compactor and len(rendered) > MAX_OUTPUT_CHARS:
            return await self._compactor.compact(
                rendered,
                max_chars=MAX_OUTPUT_CHARS,
                label="conversation recall results",
            )
        return rendered


async def _format_turns(
    rows: list[dict],
    *,
    compactor: SemanticCompactor | None,
) -> str:
    """Format turn rows as compact text for the model."""
    parts = []

    for row in rows:
        turn = row.get("turn_number", "?")
        role = row.get("role", "?")
        content = row.get("content", "") or ""
        tool_name = row.get("tool_name")
        tool_calls_raw = row.get("tool_calls")

        if role == "tool" and tool_name:
            header = f"[Turn {turn}] tool:{tool_name}"
        elif role == "assistant" and tool_calls_raw:
            try:
                calls = json.loads(tool_calls_raw)
                call_names = [c.get("function", {}).get("name", "?") for c in calls]
                header = f"[Turn {turn}] assistant (calls: {', '.join(call_names)})"
            except (json.JSONDecodeError, TypeError):
                header = f"[Turn {turn}] {role}"
        else:
            header = f"[Turn {turn}] {role}"

        if compactor and len(content) > MAX_TURN_CONTENT_CHARS:
            content = await compactor.compact(
                content,
                max_chars=MAX_TURN_CONTENT_CHARS,
                label=f"conversation recall turn {turn}",
            )

        parts.append(f"{header}:\n  {content}" if content else header)

    return "\n\n".join(parts)
