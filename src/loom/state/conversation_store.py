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
from typing import Any

from loom.state.memory import Database
from loom.utils.tokens import estimate_tokens as _estimate_tokens


class ConversationStore:
    """Append-only persistence for cowork conversation history."""

    MAX_QUERY_LIMIT = 1000
    MAX_CHAT_EVENT_LIMIT = 500

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

    # ------------------------------------------------------------------
    # Chat replay events (UI transcript journal)
    # ------------------------------------------------------------------

    async def append_chat_event(
        self,
        session_id: str,
        event_type: str,
        payload: dict | str,
        *,
        seq: int | None = None,
    ) -> int:
        """Append one chat transcript event and return its sequence number."""
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        event_type = str(event_type or "").strip()
        if not event_type:
            raise ValueError("event_type is required")

        if isinstance(payload, str):
            payload_json = payload
        else:
            payload_json = json.dumps(payload, ensure_ascii=False, default=str)

        now = datetime.now().isoformat()
        if seq is not None:
            seq = int(seq)
            if seq <= 0:
                raise ValueError("seq must be > 0")
            await self._db.execute(
                """INSERT INTO cowork_chat_events
                   (session_id, seq, event_type, payload, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, seq, event_type, payload_json, now),
            )
            return seq

        # Allocate sequence in one SQL statement so SELECT(max)+INSERT are
        # serialized together under SQLite's write lock.
        row_id = await self._db.execute_returning_id(
            """INSERT INTO cowork_chat_events
               (session_id, seq, event_type, payload, created_at)
               SELECT ?, COALESCE(MAX(seq), 0) + 1, ?, ?, ?
               FROM cowork_chat_events
               WHERE session_id = ?""",
            (session_id, event_type, payload_json, now, session_id),
        )
        row = await self._db.query_one(
            "SELECT seq FROM cowork_chat_events WHERE id = ?",
            (row_id,),
        )
        if not row:
            raise RuntimeError("Inserted chat event not found.")
        return int(row.get("seq", 0) or 0)

    async def get_last_chat_seq(self, session_id: str) -> int:
        """Return the last known chat-event sequence for a session."""
        row = await self._db.query_one(
            "SELECT COALESCE(MAX(seq), 0) AS max_seq "
            "FROM cowork_chat_events WHERE session_id = ?",
            (session_id,),
        )
        if not row:
            return 0
        return int(row.get("max_seq", 0) or 0)

    async def get_chat_events(
        self,
        session_id: str,
        *,
        before_seq: int | None = None,
        after_seq: int | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """Get chat replay events in chronological order."""
        safe_limit = max(1, min(int(limit), self.MAX_CHAT_EVENT_LIMIT))
        conditions = ["session_id = ?"]
        params: list[Any] = [session_id]
        if before_seq is not None:
            conditions.append("seq < ?")
            params.append(int(before_seq))
        if after_seq is not None:
            conditions.append("seq > ?")
            params.append(int(after_seq))

        where = " AND ".join(conditions)
        rows = await self._db.query(
            f"""SELECT * FROM cowork_chat_events
                WHERE {where}
                ORDER BY seq DESC
                LIMIT ?""",
            tuple([*params, safe_limit]),
        )
        rows = list(reversed(rows))
        return [self._normalize_chat_event_row(row) for row in rows]

    @staticmethod
    def _normalize_chat_event_row(row: dict) -> dict:
        payload_raw = row.get("payload", "{}")
        payload: dict | str
        payload_parse_error = False
        try:
            parsed = json.loads(payload_raw)
            payload = parsed if isinstance(parsed, dict) else {"value": parsed}
        except (json.JSONDecodeError, TypeError):
            payload = {"raw": str(payload_raw)}
            payload_parse_error = True

        return {
            "id": int(row.get("id", 0) or 0),
            "session_id": str(row.get("session_id", "") or ""),
            "seq": int(row.get("seq", 0) or 0),
            "event_type": str(row.get("event_type", "") or ""),
            "payload": payload,
            "payload_parse_error": payload_parse_error,
            "created_at": row.get("created_at"),
        }

    async def synthesize_chat_events_from_turns(
        self,
        session_id: str,
        *,
        before_turn: int | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """Best-effort UI transcript synthesis from persisted conversation turns."""
        safe_limit = max(1, min(int(limit), self.MAX_CHAT_EVENT_LIMIT))
        conditions = ["session_id = ?"]
        params: list[Any] = [session_id]
        if before_turn is not None:
            conditions.append("turn_number < ?")
            params.append(int(before_turn))

        where = " AND ".join(conditions)
        rows = await self._db.query(
            f"""SELECT * FROM conversation_turns
                WHERE {where}
                ORDER BY turn_number DESC
                LIMIT ?""",
            tuple([*params, safe_limit]),
        )
        if not rows:
            return []
        rows = list(reversed(rows))

        events: list[dict] = []
        for row in rows:
            role = str(row.get("role", "") or "").strip().lower()
            turn_number = int(row.get("turn_number", 0) or 0)
            created_at = row.get("created_at")

            if role == "user":
                text = str(row.get("content", "") or "")
                events.append({
                    "event_type": "user_message",
                    "payload": {"text": text},
                    "turn_number": turn_number,
                    "created_at": created_at,
                })
                continue

            if role == "assistant":
                content = str(row.get("content", "") or "")
                tool_calls = self._decode_tool_calls(row.get("tool_calls"))
                if content:
                    events.append({
                        "event_type": "assistant_text",
                        "payload": {"text": content, "markup": False},
                        "turn_number": turn_number,
                        "created_at": created_at,
                    })
                for call in tool_calls:
                    payload = self._payload_for_tool_call_start(call)
                    if payload is None:
                        continue
                    events.append({
                        "event_type": "tool_call_started",
                        "payload": payload,
                        "turn_number": turn_number,
                        "created_at": created_at,
                    })
                continue

            if role == "tool":
                payload = self._payload_for_tool_completion(row)
                events.append({
                    "event_type": "tool_call_completed",
                    "payload": payload,
                    "turn_number": turn_number,
                    "created_at": created_at,
                })
                blocks = payload.get("content_blocks")
                if isinstance(blocks, list) and blocks:
                    events.append({
                        "event_type": "content_indicator",
                        "payload": {"content_blocks": blocks},
                        "turn_number": turn_number,
                        "created_at": created_at,
                    })
                continue

            # Skip most persisted system messages from model context hints.
            if role == "system":
                continue

        return events

    @staticmethod
    def _decode_tool_calls(raw: Any) -> list[dict]:
        if raw in (None, "", []):
            return []
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return []
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        return []

    @staticmethod
    def _payload_for_tool_call_start(call: dict) -> dict | None:
        call_id = str(call.get("id", "") or "").strip()
        fn = call.get("function")
        if not isinstance(fn, dict):
            return None
        tool_name = str(fn.get("name", "") or "").strip()
        raw_args = fn.get("arguments")
        args: dict = {}
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, dict):
                    args = parsed
            except json.JSONDecodeError:
                args = {"raw": raw_args}
        elif isinstance(raw_args, dict):
            args = dict(raw_args)

        if not tool_name:
            return None
        return {
            "tool_name": tool_name,
            "args": args,
            "tool_call_id": call_id,
        }

    @staticmethod
    def _payload_for_tool_completion(row: dict) -> dict:
        from loom.tools.registry import ToolResult

        raw_content = str(row.get("content", "") or "")
        result = ToolResult.from_json(raw_content)
        from loom.content import serialize_block

        blocks = []
        if result.content_blocks:
            for block in result.content_blocks:
                try:
                    blocks.append(serialize_block(block))
                except Exception:
                    continue

        output = str(result.output or "")
        error = str(result.error or "")
        if (
            raw_content.strip()
            and not output
            and error.lower().startswith("invalid json")
        ):
            output = ConversationStore._summarize_raw_payload(raw_content)
            error = "Malformed tool result payload"

        payload: dict[str, Any] = {
            "tool_name": str(row.get("tool_name", "") or "").strip(),
            "tool_call_id": str(row.get("tool_call_id", "") or "").strip(),
            "success": bool(result.success),
            "elapsed_ms": 0,
            "output": output,
            "error": error,
        }
        if blocks:
            payload["content_blocks"] = blocks
        return payload

    @staticmethod
    def _summarize_raw_payload(raw: str, *, max_chars: int = 280) -> str:
        text = " ".join(str(raw or "").split())
        if len(text) <= max_chars:
            return text
        return f"{text[: max_chars - 3]}..."
