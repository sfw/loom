"""Append-only conversation persistence for cowork sessions.

Every message sent or received in a CoworkSession is written here
synchronously (write-through).  The in-memory message list in
CoworkSession is a cache of the most recent turns; this store holds
everything.  Nothing is ever deleted or compacted.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

from loom.state.memory import Database
from loom.utils.tokens import estimate_tokens as _estimate_tokens


class ConversationStore:
    """Append-only persistence for cowork conversation history."""

    MAX_QUERY_LIMIT = 1000

    def __init__(self, db: Database):
        self._db = db

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def create_session(
        self,
        workspace: str,
        model_name: str,
        system_prompt: str = "",
    ) -> str:
        """Create a new cowork session and return its ID."""
        session_id = uuid.uuid4().hex  # full 128-bit entropy
        now = datetime.now().isoformat()
        await self._db.execute(
            """INSERT INTO cowork_sessions
               (id, workspace_path, model_name, system_prompt, started_at, last_active_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, workspace, model_name, system_prompt, now, now),
        )
        return session_id

    async def get_session(self, session_id: str) -> dict | None:
        """Retrieve session metadata."""
        return await self._db.query_one(
            "SELECT * FROM cowork_sessions WHERE id = ?", (session_id,),
        )

    async def list_sessions(
        self, workspace: str | None = None, active_only: bool = False,
    ) -> list[dict]:
        """List sessions, optionally filtered by workspace."""
        conditions = []
        params: list = []
        if workspace:
            conditions.append("workspace_path = ?")
            params.append(workspace)
        if active_only:
            conditions.append("is_active = 1")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        return await self._db.query(
            f"SELECT * FROM cowork_sessions {where} ORDER BY last_active_at DESC",
            tuple(params),
        )

    async def update_session(
        self,
        session_id: str,
        total_tokens: int | None = None,
        turn_count: int | None = None,
        session_state: dict | None = None,
        is_active: bool | None = None,
    ) -> None:
        """Update session metadata."""
        updates = ["last_active_at = ?"]
        params: list = [datetime.now().isoformat()]

        if total_tokens is not None:
            updates.append("total_tokens = ?")
            params.append(total_tokens)
        if turn_count is not None:
            updates.append("turn_count = ?")
            params.append(turn_count)
        if session_state is not None:
            updates.append("session_state = ?")
            params.append(json.dumps(session_state))
        if is_active is not None:
            updates.append("is_active = ?")
            params.append(int(is_active))

        params.append(session_id)
        await self._db.execute(
            f"UPDATE cowork_sessions SET {', '.join(updates)} WHERE id = ?",
            tuple(params),
        )

    # ------------------------------------------------------------------
    # Turn persistence
    # ------------------------------------------------------------------

    async def append_turn(
        self,
        session_id: str,
        turn_number: int,
        role: str,
        content: str | None = None,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
    ) -> int:
        """Append a conversation turn.  Returns the row ID."""
        token_count = _estimate_tokens(content or "")
        if tool_calls:
            token_count += _estimate_tokens(json.dumps(tool_calls))

        row_id = await self._db.execute_returning_id(
            """INSERT INTO conversation_turns
               (session_id, turn_number, role, content, tool_calls,
                tool_call_id, tool_name, token_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                turn_number,
                role,
                content,
                json.dumps(tool_calls) if tool_calls else None,
                tool_call_id,
                tool_name,
                token_count,
            ),
        )
        return row_id

    async def get_turns(
        self,
        session_id: str,
        offset: int = 0,
        limit: int = 50,
    ) -> list[dict]:
        """Get turns for a session, ordered by turn number."""
        limit = min(limit, self.MAX_QUERY_LIMIT)
        return await self._db.query(
            """SELECT * FROM conversation_turns
               WHERE session_id = ?
               ORDER BY turn_number ASC
               LIMIT ? OFFSET ?""",
            (session_id, limit, offset),
        )

    async def get_recent_turns(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[dict]:
        """Get the most recent turns (for session resumption)."""
        limit = min(limit, self.MAX_QUERY_LIMIT)
        rows = await self._db.query(
            """SELECT * FROM conversation_turns
               WHERE session_id = ?
               ORDER BY turn_number DESC
               LIMIT ?""",
            (session_id, limit),
        )
        return list(reversed(rows))

    async def get_turn_count(self, session_id: str) -> int:
        """Get the number of turns in a session."""
        row = await self._db.query_one(
            "SELECT COUNT(*) as cnt FROM conversation_turns WHERE session_id = ?",
            (session_id,),
        )
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Search / retrieval
    # ------------------------------------------------------------------

    async def search_turns(
        self,
        session_id: str,
        query: str,
        limit: int = 10,
    ) -> list[dict]:
        """Full-text search across turn content."""
        limit = min(limit, self.MAX_QUERY_LIMIT)
        # Escape LIKE wildcards in user-supplied query to prevent injection
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return await self._db.query(
            """SELECT * FROM conversation_turns
               WHERE session_id = ? AND content LIKE ? ESCAPE '\\'
               ORDER BY turn_number DESC
               LIMIT ?""",
            (session_id, f"%{escaped}%", limit),
        )

    async def search_tool_calls(
        self,
        session_id: str,
        tool_name: str,
        limit: int = 10,
    ) -> list[dict]:
        """Find past calls to a specific tool."""
        limit = min(limit, self.MAX_QUERY_LIMIT)
        return await self._db.query(
            """SELECT * FROM conversation_turns
               WHERE session_id = ? AND tool_name = ?
               ORDER BY turn_number DESC
               LIMIT ?""",
            (session_id, tool_name, limit),
        )

    async def get_turn_range(
        self,
        session_id: str,
        start: int,
        end: int,
    ) -> list[dict]:
        """Get turns within a turn-number range (inclusive)."""
        return await self._db.query(
            """SELECT * FROM conversation_turns
               WHERE session_id = ? AND turn_number BETWEEN ? AND ?
               ORDER BY turn_number ASC""",
            (session_id, start, end),
        )

    # ------------------------------------------------------------------
    # Helpers for session resume
    # ------------------------------------------------------------------

    async def resume_session(
        self,
        session_id: str,
        recent_limit: int = 100,
    ) -> list[dict]:
        """Load recent turns as message dicts for session resumption.

        Returns messages in the OpenAI format ready to be loaded
        into ``CoworkSession._messages``.
        """
        rows = await self.get_recent_turns(session_id, limit=recent_limit)
        messages = []
        for row in rows:
            msg: dict = {"role": row["role"]}
            if row["content"] is not None:
                msg["content"] = row["content"]
            if row["tool_calls"]:
                msg["tool_calls"] = json.loads(row["tool_calls"])
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            messages.append(msg)
        return messages
