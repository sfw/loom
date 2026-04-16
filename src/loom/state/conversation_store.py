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

_COWORK_MEMORY_ENTRY_TYPES = frozenset({
    "decision",
    "proposal",
    "research",
    "rationale",
    "constraint",
    "risk",
    "open_question",
    "action_item",
})
_COWORK_MEMORY_STATUSES = frozenset({
    "active",
    "superseded",
    "resolved",
    "rejected",
})
_SESSION_STATE_SEMANTIC_KEYS = frozenset({
    "session_id",
    "workspace",
    "model_name",
    "turn_count",
    "total_tokens",
    "files_touched",
    "key_decisions",
    "current_focus",
    "errors_resolved",
    "active_decisions",
    "active_proposals",
    "recent_research",
    "open_questions",
    "memory_index_last_indexed_turn",
    "memory_index_degraded",
    "memory_index_failure_count",
    "memory_index_last_error",
})
_SESSION_STATE_NON_SEMANTIC_KEYS = frozenset({
    "title",
    "ui_state",
})
_UNSET = object()


class ConversationStore:
    """Append-only persistence for cowork conversation history."""

    MAX_QUERY_LIMIT = 1000
    MAX_CHAT_EVENT_LIMIT = 500
    MAX_MEMORY_QUERY_LIMIT = 200

    def __init__(self, db: Database):
        self._db = db

    @staticmethod
    def _decode_turn_metadata(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return dict(raw)
        if not isinstance(raw, str):
            return {}
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}

    @staticmethod
    def _decode_session_state_blob(raw: object) -> dict[str, Any]:
        if isinstance(raw, dict):
            return dict(raw)
        if not isinstance(raw, str):
            return {}
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}

    @staticmethod
    def _split_session_state_domains(
        state: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        semantic: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        for key, value in dict(state or {}).items():
            if key in _SESSION_STATE_SEMANTIC_KEYS:
                semantic[key] = value
            else:
                metadata[key] = value
        return semantic, metadata

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

    async def get_sessions_by_ids(self, session_ids: list[str]) -> dict[str, dict]:
        """Return session rows keyed by session ID."""
        clean_ids = sorted({
            str(session_id or "").strip()
            for session_id in session_ids
            if str(session_id or "").strip()
        })
        if not clean_ids:
            return {}
        placeholders = ", ".join("?" for _ in clean_ids)
        rows = await self._db.query(
            f"SELECT * FROM cowork_sessions WHERE id IN ({placeholders})",
            tuple(clean_ids),
        )
        return {
            str(row.get("id", "") or "").strip(): dict(row)
            for row in rows
            if str(row.get("id", "") or "").strip()
        }

    async def delete_session(self, session_id: str) -> None:
        """Delete a session and all related data."""
        await self._db.execute(
            "DELETE FROM cowork_chat_events WHERE session_id = ?", (session_id,),
        )
        await self._db.execute(
            "DELETE FROM conversation_run_links WHERE session_id = ?", (session_id,),
        )
        await self._db.execute(
            "DELETE FROM conversation_turns WHERE session_id = ?", (session_id,),
        )
        await self._db.execute(
            "DELETE FROM cowork_sessions WHERE id = ?", (session_id,),
        )

    async def update_session(
        self,
        session_id: str,
        total_tokens: int | None = None,
        turn_count: int | None = None,
        session_state: dict | None = None,
        session_state_through_turn: int | None = None,
        chat_journal_through_turn: int | None = None,
        chat_journal_through_seq: int | None = None,
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
        if session_state_through_turn is not None:
            updates.append("session_state_through_turn = ?")
            params.append(max(0, int(session_state_through_turn)))
        if chat_journal_through_turn is not None:
            updates.append("chat_journal_through_turn = ?")
            params.append(max(0, int(chat_journal_through_turn)))
        if chat_journal_through_seq is not None:
            updates.append("chat_journal_through_seq = ?")
            params.append(max(0, int(chat_journal_through_seq)))
        if is_active is not None:
            updates.append("is_active = ?")
            params.append(int(is_active))

        params.append(session_id)
        await self._db.execute(
            f"UPDATE cowork_sessions SET {', '.join(updates)} WHERE id = ?",
            tuple(params),
        )

    async def write_session_checkpoint(
        self,
        session_id: str,
        *,
        session_state: dict,
        through_turn: int,
        is_active: bool | None = None,
    ) -> None:
        """Persist a session checkpoint derived from committed canonical turns."""
        now = datetime.now().isoformat()
        safe_through_turn = max(0, int(through_turn))
        safe_is_active = None if is_active is None else int(bool(is_active))

        async def _callback(conn) -> None:
            session_row = await (
                await conn.execute(
                    "SELECT session_state FROM cowork_sessions WHERE id = ?",
                    (session_id,),
                )
            ).fetchone()
            existing_state = self._decode_session_state_blob(
                session_row[0] if session_row is not None else "",
            )
            _, metadata = self._split_session_state_domains(existing_state)
            new_semantic, _ = self._split_session_state_domains(session_state or {})
            merged_state = {
                **metadata,
                **new_semantic,
            }

            metrics_row = await (
                await conn.execute(
                    """
                    SELECT
                        COALESCE(SUM(token_count), 0) AS total_tokens,
                        COALESCE(SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END), 0) AS user_turns,
                        COALESCE(MAX(turn_number), 0) AS latest_turn
                    FROM conversation_turns
                    WHERE session_id = ? AND turn_number <= ?
                    """,
                    (session_id, safe_through_turn),
                )
            ).fetchone()
            derived_total_tokens = int((metrics_row[0] if metrics_row is not None else 0) or 0)
            derived_turn_count = int((metrics_row[1] if metrics_row is not None else 0) or 0)
            committed_through_turn = int((metrics_row[2] if metrics_row is not None else 0) or 0)
            merged_state["turn_count"] = derived_turn_count
            merged_state["total_tokens"] = derived_total_tokens
            updates = [
                "last_active_at = ?",
                "total_tokens = ?",
                "turn_count = ?",
                "session_state = ?",
                "session_state_through_turn = ?",
            ]
            params: list[Any] = [
                now,
                derived_total_tokens,
                derived_turn_count,
                json.dumps(merged_state),
                committed_through_turn,
            ]
            if safe_is_active is not None:
                updates.append("is_active = ?")
                params.append(safe_is_active)
            params.append(session_id)
            await conn.execute(
                f"UPDATE cowork_sessions SET {', '.join(updates)} WHERE id = ?",
                tuple(params),
            )

        await self._db.run_write_transaction(_callback)

    async def patch_session_state_metadata(
        self,
        session_id: str,
        *,
        title: str | None | object = _UNSET,
        ui_state: dict | object = _UNSET,
        is_active: bool | None = None,
    ) -> None:
        """Patch non-semantic session-state metadata without rewriting checkpoints."""
        now = datetime.now().isoformat()
        safe_is_active = None if is_active is None else int(bool(is_active))

        async def _callback(conn) -> None:
            row = await (
                await conn.execute(
                    "SELECT session_state FROM cowork_sessions WHERE id = ?",
                    (session_id,),
                )
            ).fetchone()
            existing_state = self._decode_session_state_blob(row[0] if row is not None else "")
            semantic, metadata = self._split_session_state_domains(existing_state)

            if title is not _UNSET:
                clean_title = str(title or "").strip()
                if clean_title:
                    metadata["title"] = clean_title
                else:
                    metadata.pop("title", None)
            if ui_state is not _UNSET:
                metadata["ui_state"] = dict(ui_state) if isinstance(ui_state, dict) else {}

            merged_state = {
                **metadata,
                **semantic,
            }

            updates = [
                "last_active_at = ?",
                "session_state = ?",
            ]
            params: list[Any] = [
                now,
                json.dumps(merged_state),
            ]
            if safe_is_active is not None:
                updates.append("is_active = ?")
                params.append(safe_is_active)
            params.append(session_id)
            await conn.execute(
                f"UPDATE cowork_sessions SET {', '.join(updates)} WHERE id = ?",
                tuple(params),
            )

        await self._db.run_write_transaction(_callback)

    # ------------------------------------------------------------------
    # Turn persistence
    # ------------------------------------------------------------------

    async def append_turn(
        self,
        session_id: str,
        turn_number: int,
        role: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
    ) -> int:
        """Append a conversation turn.  Returns the row ID."""
        token_count = _estimate_tokens(content or "")
        if metadata:
            token_count += _estimate_tokens(
                json.dumps(metadata, ensure_ascii=False, default=str),
            )
        if tool_calls:
            token_count += _estimate_tokens(json.dumps(tool_calls))

        row_id = await self._db.execute_returning_id(
            """INSERT INTO conversation_turns
               (session_id, turn_number, role, content, metadata, tool_calls,
                tool_call_id, tool_name, token_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                turn_number,
                role,
                content,
                json.dumps(metadata, ensure_ascii=False, default=str) if metadata else None,
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

    async def get_turns_before(
        self,
        session_id: str,
        *,
        before_turn: int,
        limit: int = 50,
    ) -> list[dict]:
        """Get turns older than ``before_turn``, ordered by turn number."""
        limit = min(limit, self.MAX_QUERY_LIMIT)
        rows = await self._db.query(
            """SELECT * FROM conversation_turns
               WHERE session_id = ? AND turn_number < ?
               ORDER BY turn_number DESC
               LIMIT ?""",
            (session_id, int(before_turn), limit),
        )
        return list(reversed(rows))

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

    async def get_last_turn_number(self, session_id: str) -> int:
        """Return the latest persisted conversation turn number."""
        row = await self._db.query_one(
            "SELECT COALESCE(MAX(turn_number), 0) AS max_turn "
            "FROM conversation_turns WHERE session_id = ?",
            (session_id,),
        )
        return int((row or {}).get("max_turn", 0) or 0)

    async def get_user_turn_count(self, session_id: str) -> int:
        """Return the number of user-authored turns in a session."""
        row = await self._db.query_one(
            "SELECT COUNT(*) AS cnt FROM conversation_turns WHERE session_id = ? AND role = 'user'",
            (session_id,),
        )
        return int((row or {}).get("cnt", 0) or 0)

    async def get_turns_after(
        self,
        session_id: str,
        *,
        after_turn: int,
        limit: int = 1000,
    ) -> list[dict]:
        """Load persisted turns strictly after a turn boundary."""
        safe_limit = min(max(1, int(limit)), self.MAX_QUERY_LIMIT)
        return await self._db.query(
            """SELECT * FROM conversation_turns
               WHERE session_id = ? AND turn_number > ?
               ORDER BY turn_number ASC
               LIMIT ?""",
            (session_id, max(0, int(after_turn)), safe_limit),
        )

    async def link_run(
        self,
        session_id: str,
        run_id: str,
        *,
        link_type: str = "origin",
    ) -> None:
        """Persist a conversation-to-run linkage for workspace navigation."""
        clean_session_id = str(session_id or "").strip()
        clean_run_id = str(run_id or "").strip()
        clean_link_type = str(link_type or "").strip() or "origin"
        if not clean_session_id or not clean_run_id:
            return
        await self._db.execute(
            """
            INSERT INTO conversation_run_links (session_id, run_id, link_type, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id, run_id, link_type) DO NOTHING
            """,
            (clean_session_id, clean_run_id, clean_link_type, datetime.now().isoformat()),
        )

    async def list_linked_runs(self, session_id: str) -> list[dict]:
        """Return run links for a conversation/session."""
        rows = await self._db.query(
            """
            SELECT *
            FROM conversation_run_links
            WHERE session_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (str(session_id or "").strip(),),
        )
        return [dict(row) for row in rows]

    async def list_linked_runs_for_sessions(self, session_ids: list[str]) -> dict[str, list[dict]]:
        """Return run links grouped by session ID."""
        clean_ids = sorted({
            str(session_id or "").strip()
            for session_id in session_ids
            if str(session_id or "").strip()
        })
        if not clean_ids:
            return {}
        placeholders = ", ".join("?" for _ in clean_ids)
        rows = await self._db.query(
            f"""
            SELECT *
            FROM conversation_run_links
            WHERE session_id IN ({placeholders})
            ORDER BY created_at ASC, id ASC
            """,
            tuple(clean_ids),
        )
        grouped: dict[str, list[dict]] = {session_id: [] for session_id in clean_ids}
        for row in rows:
            session_id = str(row.get("session_id", "") or "").strip()
            if not session_id:
                continue
            grouped.setdefault(session_id, []).append(dict(row))
        return grouped

    async def list_linked_conversations(self, run_id: str) -> list[dict]:
        """Return conversation links for a run."""
        rows = await self._db.query(
            """
            SELECT *
            FROM conversation_run_links
            WHERE run_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (str(run_id or "").strip(),),
        )
        return [dict(row) for row in rows]

    async def list_linked_conversations_for_runs(self, run_ids: list[str]) -> dict[str, list[dict]]:
        """Return conversation links grouped by run/task ID."""
        clean_ids = sorted({
            str(run_id or "").strip()
            for run_id in run_ids
            if str(run_id or "").strip()
        })
        if not clean_ids:
            return {}
        placeholders = ", ".join("?" for _ in clean_ids)
        rows = await self._db.query(
            f"""
            SELECT *
            FROM conversation_run_links
            WHERE run_id IN ({placeholders})
            ORDER BY created_at ASC, id ASC
            """,
            tuple(clean_ids),
        )
        grouped: dict[str, list[dict]] = {run_id: [] for run_id in clean_ids}
        for row in rows:
            run_id = str(row.get("run_id", "") or "").strip()
            if not run_id:
                continue
            grouped.setdefault(run_id, []).append(dict(row))
        return grouped

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
    # Cowork memory index (typed recall layer)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_memory_entry_type(value: object) -> str:
        text = str(value or "").strip().lower()
        return text if text in _COWORK_MEMORY_ENTRY_TYPES else "research"

    @staticmethod
    def _normalize_memory_status(value: object) -> str:
        text = str(value or "").strip().lower()
        return text if text in _COWORK_MEMORY_STATUSES else "active"

    @staticmethod
    def _coerce_text_list(value: object) -> list[str]:
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return []
            return [candidate]
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if not text:
                continue
            if text not in cleaned:
                cleaned.append(text)
        return cleaned

    @staticmethod
    def _safe_int(value: object, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _status_priority_sql(alias: str = "c") -> str:
        return (
            f"CASE {alias}.status "
            "WHEN 'active' THEN 0 "
            "WHEN 'resolved' THEN 1 "
            "WHEN 'superseded' THEN 2 "
            "WHEN 'rejected' THEN 3 "
            "ELSE 4 END"
        )

    @staticmethod
    def _decode_json_column(value: object, *, default: object) -> object:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return default
            try:
                parsed = json.loads(text)
            except Exception:
                return default
            return parsed
        return default

    @classmethod
    def _normalize_memory_row(cls, row: dict) -> dict:
        normalized = dict(row)
        normalized["entry_type"] = cls._normalize_memory_entry_type(
            normalized.get("entry_type"),
        )
        normalized["status"] = cls._normalize_memory_status(normalized.get("status"))
        normalized["confidence"] = float(normalized.get("confidence", 0.0) or 0.0)
        normalized["tags"] = cls._decode_json_column(
            normalized.get("tags_json"),
            default=[],
        )
        normalized["entities"] = cls._decode_json_column(
            normalized.get("entities_json"),
            default=[],
        )
        normalized["source_roles"] = cls._decode_json_column(
            normalized.get("source_roles_json"),
            default=[],
        )
        return normalized

    async def get_cowork_memory_index_state(self, session_id: str) -> dict:
        row = await self._db.query_one(
            """SELECT * FROM cowork_memory_index_state
               WHERE session_id = ?""",
            (session_id,),
        )
        if not row:
            return {
                "session_id": session_id,
                "last_indexed_turn": 0,
                "index_version": 1,
                "index_degraded": False,
                "last_indexed_at": "",
                "last_error": "",
                "failure_count": 0,
            }
        state = dict(row)
        state["last_indexed_turn"] = int(state.get("last_indexed_turn", 0) or 0)
        state["index_version"] = int(state.get("index_version", 1) or 1)
        state["failure_count"] = int(state.get("failure_count", 0) or 0)
        state["index_degraded"] = bool(state.get("index_degraded", 0))
        return state

    async def upsert_cowork_memory_index_state(
        self,
        session_id: str,
        *,
        last_indexed_turn: int,
        index_degraded: bool = False,
        last_error: str = "",
        failure_count: int = 0,
        index_version: int = 1,
    ) -> None:
        now = datetime.now().isoformat()
        await self._db.execute(
            """INSERT INTO cowork_memory_index_state
               (session_id, last_indexed_turn, index_version, index_degraded,
                last_indexed_at, last_error, failure_count, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                   last_indexed_turn=excluded.last_indexed_turn,
                   index_version=excluded.index_version,
                   index_degraded=excluded.index_degraded,
                   last_indexed_at=excluded.last_indexed_at,
                   last_error=excluded.last_error,
                   failure_count=excluded.failure_count,
                   updated_at=excluded.updated_at""",
            (
                str(session_id or "").strip(),
                max(0, int(last_indexed_turn)),
                max(1, int(index_version)),
                1 if bool(index_degraded) else 0,
                now,
                str(last_error or "").strip(),
                max(0, int(failure_count)),
                now,
                now,
            ),
        )

    async def upsert_cowork_memory_entry(
        self,
        session_id: str,
        entry: dict,
    ) -> int:
        clean_session = str(session_id or "").strip()
        if not clean_session:
            raise ValueError("session_id is required")
        if not isinstance(entry, dict):
            raise ValueError("entry must be a dict")

        now = datetime.now().isoformat()
        tags = self._coerce_text_list(entry.get("tags", entry.get("tags_json", [])))
        entities = self._coerce_text_list(
            entry.get("entities", entry.get("entities_json", []))
        )
        source_roles = self._coerce_text_list(
            entry.get("source_roles", entry.get("source_roles_json", []))
        )
        fingerprint = str(entry.get("fingerprint", "") or "").strip()
        if not fingerprint:
            raise ValueError("entry fingerprint is required")

        await self._db.execute(
            """INSERT INTO cowork_memory_entries
               (session_id, entry_type, status, summary, rationale, topic,
                tags_json, tags_text, entities_json, entities_text,
                source_turn_start, source_turn_end, source_roles_json,
                evidence_excerpt, supersedes_entry_id, confidence, fingerprint,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id, fingerprint) DO UPDATE SET
                   entry_type=excluded.entry_type,
                   status=excluded.status,
                   summary=excluded.summary,
                   rationale=excluded.rationale,
                   topic=excluded.topic,
                   tags_json=excluded.tags_json,
                   tags_text=excluded.tags_text,
                   entities_json=excluded.entities_json,
                   entities_text=excluded.entities_text,
                   source_turn_start=excluded.source_turn_start,
                   source_turn_end=excluded.source_turn_end,
                   source_roles_json=excluded.source_roles_json,
                   evidence_excerpt=excluded.evidence_excerpt,
                   supersedes_entry_id=excluded.supersedes_entry_id,
                   confidence=excluded.confidence,
                   updated_at=excluded.updated_at""",
            (
                clean_session,
                self._normalize_memory_entry_type(entry.get("entry_type")),
                self._normalize_memory_status(entry.get("status")),
                str(entry.get("summary", "") or "").strip(),
                str(entry.get("rationale", "") or "").strip(),
                str(entry.get("topic", "") or "").strip(),
                json.dumps(tags, ensure_ascii=False),
                ", ".join(tags),
                json.dumps(entities, ensure_ascii=False),
                ", ".join(entities),
                max(0, self._safe_int(entry.get("source_turn_start", 0), 0)),
                max(
                    max(0, self._safe_int(entry.get("source_turn_start", 0), 0)),
                    self._safe_int(entry.get("source_turn_end", 0), 0),
                ),
                json.dumps(source_roles, ensure_ascii=False),
                str(entry.get("evidence_excerpt", "") or "").strip(),
                (
                    self._safe_int(entry.get("supersedes_entry_id"), 0)
                    if entry.get("supersedes_entry_id") not in (None, "", 0, "0")
                    else None
                ),
                self._safe_float(entry.get("confidence", 0.0), 0.0),
                fingerprint,
                now,
                now,
            ),
        )
        row = await self._db.query_one(
            """SELECT id FROM cowork_memory_entries
               WHERE session_id = ? AND fingerprint = ?""",
            (clean_session, fingerprint),
        )
        return int(row.get("id", 0) or 0) if row else 0

    async def upsert_cowork_memory_entries(
        self,
        session_id: str,
        entries: list[dict],
    ) -> int:
        inserted = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            row_id = await self.upsert_cowork_memory_entry(session_id, entry)
            if row_id > 0:
                inserted += 1
        return inserted

    async def _has_cowork_memory_fts(self) -> bool:
        row = await self._db.query_one(
            """SELECT 1 AS present
               FROM sqlite_master
               WHERE type='table' AND name='cowork_memory_fts'""",
        )
        return bool(row)

    async def rebuild_cowork_memory_fts(self) -> None:
        if not await self._has_cowork_memory_fts():
            return
        await self._db.execute(
            "INSERT INTO cowork_memory_fts(cowork_memory_fts) VALUES ('rebuild')",
        )

    async def search_cowork_memory_entries(
        self,
        session_id: str,
        *,
        query: str = "",
        entry_type: str = "",
        status: str = "",
        topic: str = "",
        limit: int = 20,
        force_fts: bool = False,
    ) -> list[dict]:
        safe_limit = max(1, min(int(limit), self.MAX_MEMORY_QUERY_LIMIT))
        clean_session = str(session_id or "").strip()
        clean_query = str(query or "").strip()
        clean_topic = str(topic or "").strip()
        clean_type = self._normalize_memory_entry_type(entry_type) if entry_type else ""
        clean_status = self._normalize_memory_status(status) if status else ""

        if clean_query and (await self._has_cowork_memory_fts()):
            where = ["c.session_id = ?", "cowork_memory_fts MATCH ?"]
            params: list[Any] = [clean_session, clean_query]
            if clean_type:
                where.append("c.entry_type = ?")
                params.append(clean_type)
            if clean_status:
                where.append("c.status = ?")
                params.append(clean_status)
            if clean_topic:
                escaped = clean_topic.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                where.append("c.topic LIKE ? ESCAPE '\\'")
                params.append(f"%{escaped}%")
            try:
                rows = await self._db.query(
                    f"""SELECT c.*, bm25(cowork_memory_fts) AS rank_score
                        FROM cowork_memory_fts
                        JOIN cowork_memory_entries c ON c.id = cowork_memory_fts.rowid
                        WHERE {' AND '.join(where)}
                        ORDER BY {self._status_priority_sql('c')}, rank_score ASC, c.updated_at DESC
                        LIMIT ?""",
                    tuple([*params, safe_limit]),
                )
                return [self._normalize_memory_row(row) for row in rows]
            except Exception:
                if force_fts:
                    return []

        if force_fts and clean_query:
            return []

        where = ["session_id = ?"]
        params = [clean_session]
        if clean_type:
            where.append("entry_type = ?")
            params.append(clean_type)
        if clean_status:
            where.append("status = ?")
            params.append(clean_status)
        if clean_topic:
            escaped_topic = (
                clean_topic.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            where.append("topic LIKE ? ESCAPE '\\'")
            params.append(f"%{escaped_topic}%")
        if clean_query:
            escaped_query = (
                clean_query.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            like = f"%{escaped_query}%"
            where.append(
                "(summary LIKE ? ESCAPE '\\' OR rationale LIKE ? ESCAPE '\\' "
                "OR evidence_excerpt LIKE ? ESCAPE '\\')",
            )
            params.extend([like, like, like])

        rows = await self._db.query(
            f"""SELECT * FROM cowork_memory_entries
                WHERE {' AND '.join(where)}
                ORDER BY {self._status_priority_sql('cowork_memory_entries')}, updated_at DESC
                LIMIT ?""",
            tuple([*params, safe_limit]),
        )
        return [self._normalize_memory_row(row) for row in rows]

    async def get_cowork_memory_entries_by_ids(
        self,
        session_id: str,
        *,
        entry_ids: list[int],
        limit: int = 50,
    ) -> list[dict]:
        clean_ids: list[int] = []
        for item in entry_ids:
            value = self._safe_int(item, 0)
            if value > 0:
                clean_ids.append(value)
        if not clean_ids:
            return []
        safe_limit = max(1, min(int(limit), self.MAX_MEMORY_QUERY_LIMIT))
        placeholders = ", ".join("?" for _ in clean_ids)
        rows = await self._db.query(
            f"""SELECT * FROM cowork_memory_entries
                WHERE session_id = ? AND id IN ({placeholders})
                ORDER BY {self._status_priority_sql('cowork_memory_entries')}, updated_at DESC
                LIMIT ?""",
            tuple([session_id, *clean_ids, safe_limit]),
        )
        return [self._normalize_memory_row(row) for row in rows]

    async def get_cowork_memory_timeline(
        self,
        session_id: str,
        *,
        topic: str = "",
        limit: int = 50,
    ) -> list[dict]:
        safe_limit = max(1, min(int(limit), self.MAX_MEMORY_QUERY_LIMIT))
        where = ["session_id = ?"]
        params: list[Any] = [session_id]
        clean_topic = str(topic or "").strip()
        if clean_topic:
            escaped = clean_topic.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            where.append("topic LIKE ? ESCAPE '\\'")
            params.append(f"%{escaped}%")
        rows = await self._db.query(
            f"""SELECT * FROM cowork_memory_entries
                WHERE {' AND '.join(where)}
                ORDER BY source_turn_start ASC, source_turn_end ASC, id ASC
                LIMIT ?""",
            tuple([*params, safe_limit]),
        )
        return [self._normalize_memory_row(row) for row in rows]

    async def get_cowork_memory_active_snapshot(
        self,
        session_id: str,
        *,
        max_decisions: int = 4,
        max_proposals: int = 4,
        max_research: int = 4,
        max_questions: int = 4,
    ) -> dict[str, list[dict]]:
        decisions = await self.search_cowork_memory_entries(
            session_id,
            entry_type="decision",
            status="active",
            limit=max_decisions,
        )
        proposals = await self.search_cowork_memory_entries(
            session_id,
            entry_type="proposal",
            status="active",
            limit=max_proposals,
        )
        research = await self.search_cowork_memory_entries(
            session_id,
            entry_type="research",
            status="active",
            limit=max_research,
        )
        open_questions = await self.search_cowork_memory_entries(
            session_id,
            entry_type="open_question",
            status="active",
            limit=max_questions,
        )
        return {
            "active_decisions": decisions,
            "active_proposals": proposals,
            "recent_research": research,
            "open_questions": open_questions,
        }

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
            metadata = self._decode_turn_metadata(row.get("metadata"))
            for key in (
                "workspace_paths",
                "workspace_files",
                "workspace_directories",
                "content_blocks",
            ):
                value = metadata.get(key)
                if isinstance(value, list) and value:
                    msg[key] = value
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
        journal_through_turn: int | None = None,
        journal_through_seq: int | None = None,
    ) -> int:
        """Append one chat transcript event and return its sequence number."""
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        event_type = str(event_type or "").strip()
        if not event_type:
            raise ValueError("event_type is required")

        payload_dict = payload if isinstance(payload, dict) else None
        if journal_through_turn is not None and isinstance(payload_dict, dict):
            payload_dict = dict(payload_dict)
            payload_dict.setdefault("journal_through_turn", max(0, int(journal_through_turn)))
        if isinstance(payload, str):
            payload_json = payload
        else:
            payload_json = json.dumps(
                payload_dict if payload_dict is not None else payload,
                ensure_ascii=False,
                default=str,
            )

        now = datetime.now().isoformat()

        async def _callback(conn) -> int:
            assigned_seq: int
            if seq is not None:
                explicit_seq = int(seq)
                if explicit_seq <= 0:
                    raise ValueError("seq must be > 0")
                await conn.execute(
                    """INSERT INTO cowork_chat_events
                       (session_id, seq, event_type, payload, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (session_id, explicit_seq, event_type, payload_json, now),
                )
                assigned_seq = explicit_seq
            else:
                cursor = await conn.execute(
                    """INSERT INTO cowork_chat_events
                       (session_id, seq, event_type, payload, created_at)
                       SELECT ?, COALESCE(MAX(seq), 0) + 1, ?, ?, ?
                       FROM cowork_chat_events
                       WHERE session_id = ?""",
                    (session_id, event_type, payload_json, now, session_id),
                )
                row = await (
                    await conn.execute(
                        "SELECT seq FROM cowork_chat_events WHERE id = ?",
                        (cursor.lastrowid,),
                    )
                ).fetchone()
                if row is None:
                    raise RuntimeError("Inserted chat event not found.")
                assigned_seq = int(row[0] or 0)

            if journal_through_turn is not None:
                through_turn = max(0, int(journal_through_turn))
                through_seq = max(
                    0,
                    int(
                        assigned_seq
                        if journal_through_seq is None
                        else journal_through_seq
                    ),
                )
                await conn.execute(
                    """
                    UPDATE cowork_sessions
                    SET last_active_at = ?,
                        chat_journal_through_turn = CASE
                            WHEN chat_journal_through_turn > ? THEN chat_journal_through_turn
                            ELSE ?
                        END,
                        chat_journal_through_seq = CASE
                            WHEN chat_journal_through_seq > ? THEN chat_journal_through_seq
                            ELSE ?
                        END
                    WHERE id = ?
                    """,
                    (
                        now,
                        through_turn,
                        through_turn,
                        through_seq,
                        through_seq,
                        session_id,
                    ),
                )
            return assigned_seq

        return int(await self._db.run_write_transaction(_callback))

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

    async def backfill_chat_journal_coverage(self, session_id: str) -> tuple[int, int]:
        """Persist explicit journal coverage recovered from durable replay rows."""
        async def _callback(conn) -> tuple[int, int]:
            row = await (
                await conn.execute(
                    """
                    SELECT chat_journal_through_turn, chat_journal_through_seq
                    FROM cowork_sessions
                    WHERE id = ?
                    """,
                    (session_id,),
                )
            ).fetchone()
            if row is None:
                return (0, 0)
            current_turn = int(row[0] or 0)
            current_seq = int(row[1] or 0)
            if current_turn > 0 and current_seq > 0:
                return (current_turn, current_seq)
            rows = await (
                await conn.execute(
                    """
                    SELECT seq, payload
                    FROM cowork_chat_events
                    WHERE session_id = ? AND event_type = 'turn_separator'
                    ORDER BY seq DESC
                    LIMIT 32
                    """,
                    (session_id,),
                )
            ).fetchall()
            recovered_turn = 0
            recovered_seq = 0
            for seq, payload_raw in rows:
                payload = self._decode_session_state_blob(payload_raw)
                candidate_turn = int(payload.get("journal_through_turn", 0) or 0)
                if candidate_turn <= 0:
                    continue
                recovered_turn = candidate_turn
                recovered_seq = int(seq or 0)
                break
            if recovered_turn <= 0 or recovered_seq <= 0:
                return (0, 0)
            await conn.execute(
                """
                UPDATE cowork_sessions
                SET last_active_at = ?,
                    chat_journal_through_turn = ?,
                    chat_journal_through_seq = ?
                WHERE id = ?
                """,
                (
                    datetime.now().isoformat(),
                    recovered_turn,
                    recovered_seq,
                    session_id,
                ),
            )
            return (recovered_turn, recovered_seq)

        result = await self._db.run_write_transaction(_callback)
        return (int(result[0]), int(result[1]))

    async def get_chat_events(
        self,
        session_id: str,
        *,
        before_seq: int | None = None,
        after_seq: int | None = None,
        through_seq: int | None = None,
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
        if through_seq is not None:
            conditions.append("seq <= ?")
            params.append(int(through_seq))

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

    async def get_transcript_page(
        self,
        session_id: str,
        *,
        before_seq: int | None = None,
        before_turn: int | None = None,
        after_seq: int = 0,
        limit: int = 200,
    ) -> list[dict]:
        """Load a transcript page using journal coverage rather than journal presence."""
        safe_limit = max(1, min(int(limit), self.MAX_CHAT_EVENT_LIMIT))
        session = await self.get_session(session_id)
        if session is None:
            return []

        if after_seq > 0 and before_seq is None:
            # Incremental consumers already have the historical transcript. For
            # live catch-up, return only durable chat-event rows so a just-sent
            # turn is not replayed once from the journal and again from
            # synthesized conversation turns while journal coverage lags.
            return await self.get_chat_events(
                session_id,
                after_seq=max(0, int(after_seq)),
                limit=safe_limit,
            )

        covered_turn = max(0, int(session.get("chat_journal_through_turn", 0) or 0))
        covered_seq = max(0, int(session.get("chat_journal_through_seq", 0) or 0))
        if covered_turn <= 0 or covered_seq <= 0:
            covered_turn, covered_seq = await self.backfill_chat_journal_coverage(session_id)
        latest_turn = await self.get_last_turn_number(session_id)

        if before_turn is not None and before_seq is None:
            return await self.synthesize_chat_events_from_turns(
                session_id,
                before_turn=max(1, int(before_turn)),
                limit=safe_limit,
            )

        if covered_turn <= 0 or covered_seq <= 0:
            if latest_turn <= 0:
                return await self.get_chat_events(
                    session_id,
                    before_seq=before_seq,
                    after_seq=(None if after_seq <= 0 else after_seq),
                    limit=safe_limit,
                )
            if before_turn is not None:
                return await self.synthesize_chat_events_from_turns(
                    session_id,
                    before_turn=max(1, int(before_turn)),
                    limit=safe_limit,
                )
            if before_seq is not None:
                return await self.synthesize_chat_events_from_turns(
                    session_id,
                    before_turn=max(1, int((int(before_seq) + 99) / 100)),
                    limit=safe_limit,
                )
            if after_seq > 0:
                return []
            return await self._synthesized_transcript_suffix(
                session_id,
                after_turn=max(0, int(after_seq / 100)),
                before_turn=before_turn,
                limit=safe_limit,
            )

        if before_seq is not None:
            rows = await self.get_chat_events(
                session_id,
                before_seq=min(max(1, int(before_seq)), covered_seq + 1),
                through_seq=covered_seq,
                limit=safe_limit,
            )
            if rows:
                return rows
            synth_before_turn = (
                max(1, int(before_turn))
                if before_turn is not None
                else max(1, int((int(before_seq) + 99) / 100))
            )
            return await self.synthesize_chat_events_from_turns(
                session_id,
                before_turn=synth_before_turn,
                limit=safe_limit,
            )

        if after_seq > 0:
            journal_rows = await self.get_chat_events(
                session_id,
                after_seq=min(max(0, int(after_seq)), covered_seq),
                through_seq=covered_seq,
                limit=safe_limit,
            )
            if covered_turn >= latest_turn:
                return journal_rows
            synth_after_turn = (
                covered_turn
                if after_seq <= covered_seq
                else max(covered_turn, int(after_seq / 100))
            )
            synth_rows = await self._synthesized_transcript_suffix(
                session_id,
                after_turn=synth_after_turn,
                limit=safe_limit,
            )
            return [*journal_rows, *synth_rows][-safe_limit:]

        if covered_turn >= latest_turn:
            return await self.get_chat_events(
                session_id,
                through_seq=covered_seq,
                limit=safe_limit,
            )

        journal_rows = await self.get_chat_events(
            session_id,
            through_seq=covered_seq,
            limit=safe_limit,
        )
        synth_rows = await self._synthesized_transcript_suffix(
            session_id,
            after_turn=covered_turn,
            limit=safe_limit,
        )
        return [*journal_rows, *synth_rows][-safe_limit:]

    async def _synthesized_transcript_suffix(
        self,
        session_id: str,
        *,
        after_turn: int = 0,
        before_turn: int | None = None,
        limit: int = 200,
    ) -> list[dict]:
        safe_limit = max(1, min(int(limit), self.MAX_CHAT_EVENT_LIMIT))
        conditions = ["session_id = ?"]
        params: list[Any] = [session_id]
        if after_turn > 0:
            conditions.append("turn_number > ?")
            params.append(int(after_turn))
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
        return self._synthesize_chat_events_from_turn_rows(session_id, rows)

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
        return self._synthesize_chat_events_from_turn_rows(session_id, rows)

    def _synthesize_chat_events_from_turn_rows(
        self,
        session_id: str,
        rows: list[dict],
    ) -> list[dict]:
        events: list[dict] = []
        for row in rows:
            role = str(row.get("role", "") or "").strip().lower()
            turn_number = int(row.get("turn_number", 0) or 0)
            created_at = row.get("created_at")
            seq_base = turn_number * 100
            seq_index = 0

            def _append(event_type: str, payload: dict[str, Any]) -> None:
                nonlocal seq_index
                events.append({
                    "id": 0,
                    "session_id": session_id,
                    "seq": seq_base + seq_index,
                    "event_type": event_type,
                    "payload": payload,
                    "payload_parse_error": False,
                    "turn_number": turn_number,
                    "created_at": created_at,
                })
                seq_index += 1

            if role == "user":
                text = str(row.get("content", "") or "")
                metadata = self._decode_turn_metadata(row.get("metadata"))
                payload: dict[str, Any] = {"text": text}
                attachment_payload = {
                    key: metadata.get(key)
                    for key in (
                        "workspace_paths",
                        "workspace_files",
                        "workspace_directories",
                        "content_blocks",
                    )
                    if isinstance(metadata.get(key), list) and metadata.get(key)
                }
                payload.update(attachment_payload)
                _append("user_message", payload)
                if attachment_payload:
                    _append("content_indicator", attachment_payload)
                continue

            if role == "assistant":
                content = str(row.get("content", "") or "")
                tool_calls = self._decode_tool_calls(row.get("tool_calls"))
                if content:
                    _append("assistant_text", {"text": content, "markup": False})
                for call in tool_calls:
                    payload = self._payload_for_tool_call_start(call)
                    if payload is None:
                        continue
                    _append("tool_call_started", payload)
                continue

            if role == "tool":
                payload = self._payload_for_tool_completion(row)
                _append("tool_call_completed", payload)
                blocks = payload.get("content_blocks")
                if isinstance(blocks, list) and blocks:
                    _append("content_indicator", {"content_blocks": blocks})
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
    def _ask_user_question_payload(result: object) -> dict | None:
        from loom.tools.ask_user import normalize_ask_user_args
        from loom.tools.registry import ToolResult

        if not isinstance(result, ToolResult):
            return None
        data = result.data
        if not isinstance(data, dict):
            return None
        candidate = dict(data)
        options_v2 = candidate.get("options_v2")
        if isinstance(options_v2, list) and options_v2:
            candidate["options"] = options_v2
        normalized = normalize_ask_user_args(candidate)
        question = str(normalized.get("question", "") or "").strip()
        if not question:
            return None
        return normalized

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
        if isinstance(result.data, dict):
            payload["data"] = dict(result.data)
        if payload["tool_name"] == "ask_user":
            question_payload = ConversationStore._ask_user_question_payload(result)
            if question_payload is not None:
                payload["question_payload"] = question_payload
        if blocks:
            payload["content_blocks"] = blocks
        return payload

    @staticmethod
    def _summarize_raw_payload(raw: str, *, max_chars: int = 280) -> str:
        text = " ".join(str(raw or "").split())
        if len(text) <= max_chars:
            return text
        return f"{text[: max_chars - 3]}..."
