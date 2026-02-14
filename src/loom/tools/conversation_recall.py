"""Conversation recall tool: search and retrieve from the session archive.

The model calls this to access past conversation that has fallen out of
the context window.  The full history is in SQLite — nothing is lost.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loom.tools.registry import Tool, ToolContext, ToolResult

if TYPE_CHECKING:
    from loom.state.conversation_store import ConversationStore


# Max output size in characters (~8K tokens at 4 chars/token).
MAX_OUTPUT_CHARS = 32_000


class ConversationRecallTool(Tool):
    """Search and retrieve past conversation messages from the session archive.

    Use this when you need to recall:
    - What the user said earlier about a topic
    - The output of a previous tool call
    - A decision or discussion from earlier in the session
    - Code that was read or written earlier

    The full conversation history is preserved — nothing is ever lost.
    """

    name = "conversation_recall"
    description = (
        "Search the conversation archive for past messages. "
        "Actions: 'search' (full-text by keyword), "
        "'range' (get turns by number range), "
        "'tool_calls' (find past calls to a specific tool), "
        "'summary' (session overview and stats)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "range", "tool_calls", "summary"],
                "description": (
                    "search: full-text search across all messages. "
                    "range: get turns by number range. "
                    "tool_calls: find past calls to a specific tool. "
                    "summary: session overview and stats."
                ),
            },
            "query": {
                "type": "string",
                "description": "Search query (for 'search' action).",
            },
            "tool_name": {
                "type": "string",
                "description": "Tool name to filter by (for 'tool_calls' action).",
            },
            "start_turn": {
                "type": "integer",
                "description": "Start of turn range (for 'range' action).",
            },
            "end_turn": {
                "type": "integer",
                "description": "End of turn range, inclusive (for 'range' action).",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return. Default 10.",
            },
        },
        "required": ["action"],
    }

    def __init__(
        self,
        store: ConversationStore | None = None,
        session_id: str = "",
        session_state: object | None = None,
    ):
        self._store = store
        self._session_id = session_id
        self._session_state = session_state

    def bind(
        self,
        store: ConversationStore,
        session_id: str,
        session_state: object | None = None,
    ) -> None:
        """Bind to a specific session after construction."""
        self._store = store
        self._session_id = session_id
        self._session_state = session_state

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if self._store is None or not self._session_id:
            return ToolResult.fail(
                "Conversation recall is not available (no active session)."
            )

        action = args.get("action", "")
        limit = args.get("limit", 10)

        if action == "search":
            return await self._search(args.get("query", ""), limit)
        elif action == "range":
            return await self._range(
                args.get("start_turn", 0),
                args.get("end_turn", 0),
            )
        elif action == "tool_calls":
            return await self._tool_calls(args.get("tool_name", ""), limit)
        elif action == "summary":
            return self._summary()
        else:
            return ToolResult.fail(f"Unknown action: {action}")

    async def _search(self, query: str, limit: int) -> ToolResult:
        if not query:
            return ToolResult.fail("'query' parameter is required for search.")

        rows = await self._store.search_turns(self._session_id, query, limit=limit)
        if not rows:
            return ToolResult.ok(f"No messages found matching '{query}'.")

        return ToolResult.ok(_format_turns(rows))

    async def _range(self, start: int, end: int) -> ToolResult:
        if end < start:
            return ToolResult.fail("end_turn must be >= start_turn.")

        rows = await self._store.get_turn_range(self._session_id, start, end)
        if not rows:
            return ToolResult.ok(f"No turns found in range {start}-{end}.")

        return ToolResult.ok(_format_turns(rows))

    async def _tool_calls(self, tool_name: str, limit: int) -> ToolResult:
        if not tool_name:
            return ToolResult.fail("'tool_name' parameter is required for tool_calls.")

        rows = await self._store.search_tool_calls(
            self._session_id, tool_name, limit=limit,
        )
        if not rows:
            return ToolResult.ok(f"No calls to '{tool_name}' found in archive.")

        return ToolResult.ok(_format_turns(rows))

    def _summary(self) -> ToolResult:
        if self._session_state and hasattr(self._session_state, "to_yaml"):
            return ToolResult.ok(self._session_state.to_yaml())
        return ToolResult.ok("No session state available.")


def _format_turns(rows: list[dict]) -> str:
    """Format turn rows as compact text for the model."""
    parts = []
    total_len = 0

    for row in rows:
        turn = row.get("turn_number", "?")
        role = row.get("role", "?")
        content = row.get("content", "") or ""
        tool_name = row.get("tool_name")
        tool_calls_raw = row.get("tool_calls")

        if role == "tool" and tool_name:
            header = f"[Turn {turn}] tool:{tool_name}"
        elif role == "assistant" and tool_calls_raw:
            # Show tool call names
            try:
                calls = json.loads(tool_calls_raw)
                call_names = [c.get("function", {}).get("name", "?") for c in calls]
                header = f"[Turn {turn}] assistant (calls: {', '.join(call_names)})"
            except (json.JSONDecodeError, TypeError):
                header = f"[Turn {turn}] {role}"
        else:
            header = f"[Turn {turn}] {role}"

        # Truncate very large content (e.g. file reads)
        if len(content) > 2000:
            content = content[:2000] + f"\n  [... truncated, {len(content)} chars total]"

        entry = f"{header}:\n  {content}" if content else header
        entry_len = len(entry)

        if total_len + entry_len > MAX_OUTPUT_CHARS:
            parts.append(f"[... {len(rows) - len(parts)} more turns truncated]")
            break

        parts.append(entry)
        total_len += entry_len

    return "\n\n".join(parts)
