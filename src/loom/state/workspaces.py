"""Workspace registry and workspace-scoped metadata persistence."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loom.state.memory import Database


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def canonicalize_workspace_path(raw_path: object) -> str:
    """Return a normalized absolute path for a workspace-like input."""
    text = str(raw_path or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).expanduser().resolve(strict=False))
    except Exception:
        return str(Path(text).expanduser().absolute())


def _decode_json_object(
    raw_value: object,
    *,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fallback = dict(default or {})
    if isinstance(raw_value, dict):
        return dict(raw_value)
    if not isinstance(raw_value, str):
        return fallback
    text = raw_value.strip()
    if not text:
        return fallback
    try:
        parsed = json.loads(text)
    except Exception:
        return fallback
    return dict(parsed) if isinstance(parsed, dict) else fallback


class WorkspaceRegistry:
    """Persisted workspace registry with lightweight discovery from existing data."""

    def __init__(self, db: Database):
        self._db = db

    @staticmethod
    def _default_display_name(path_text: str) -> str:
        path = Path(path_text)
        return path.name or path_text

    @staticmethod
    def _workspace_payload_from_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return None
        payload = dict(row)
        payload["metadata"] = _decode_json_object(payload.get("metadata"))
        payload["is_archived"] = bool(payload.get("is_archived", 0))
        return payload

    async def sync_known_workspaces_from_sources(self) -> int:
        """Ensure task/session workspace paths exist in the registry."""
        inserted = 0
        task_rows = await self._db.query(
            """
            SELECT workspace_path, metadata
            FROM tasks
            WHERE TRIM(COALESCE(workspace_path, '')) <> ''
            """,
        )
        session_rows = await self._db.query(
            """
            SELECT workspace_path
            FROM cowork_sessions
            WHERE TRIM(COALESCE(workspace_path, '')) <> ''
            """,
        )

        seen_paths: set[str] = set()
        for row in task_rows:
            metadata = _decode_json_object(row.get("metadata"))
            canonical_path = canonicalize_workspace_path(
                metadata.get("source_workspace_root") or row.get("workspace_path"),
            )
            if canonical_path:
                seen_paths.add(canonical_path)
        for row in session_rows:
            canonical_path = canonicalize_workspace_path(row.get("workspace_path"))
            if not canonical_path:
                continue
            seen_paths.add(canonical_path)

        for canonical_path in sorted(seen_paths):
            created = await self.ensure_workspace(canonical_path)
            if created is not None:
                inserted += 1
        return inserted

    async def ensure_workspace(
        self,
        workspace_path: object,
        *,
        display_name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Create a workspace row if one does not already exist."""
        canonical_path = canonicalize_workspace_path(workspace_path)
        if not canonical_path:
            return None
        existing = await self.get_by_path(canonical_path)
        if existing is not None:
            return None

        now = _now_iso()
        workspace_id = f"ws-{uuid.uuid4().hex[:12]}"
        resolved_display = str(display_name or "").strip() or self._default_display_name(
            canonical_path
        )
        await self._db.execute(
            """
            INSERT INTO workspaces
                (id, canonical_path, display_name, workspace_type, sort_order,
                 last_opened_at, is_archived, metadata, created_at, updated_at)
            VALUES (?, ?, ?, 'local', 0, ?, 0, ?, ?, ?)
            """,
            (
                workspace_id,
                canonical_path,
                resolved_display,
                now,
                json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                now,
                now,
            ),
        )
        return await self.get(workspace_id)

    async def list(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        await self.sync_known_workspaces_from_sources()
        params: tuple[Any, ...] = ()
        where = ""
        if not include_archived:
            where = "WHERE is_archived = 0"
        rows = await self._db.query(
            f"""
            SELECT *
            FROM workspaces
            {where}
            ORDER BY sort_order ASC, COALESCE(last_opened_at, '') DESC, display_name ASC
            """,
            params,
        )
        return [payload for row in rows if (payload := self._workspace_payload_from_row(row))]

    async def get(self, workspace_id: str) -> dict[str, Any] | None:
        row = await self._db.query_one(
            "SELECT * FROM workspaces WHERE id = ?",
            (str(workspace_id or "").strip(),),
        )
        return self._workspace_payload_from_row(row)

    async def get_by_path(self, workspace_path: object) -> dict[str, Any] | None:
        canonical_path = canonicalize_workspace_path(workspace_path)
        if not canonical_path:
            return None
        row = await self._db.query_one(
            "SELECT * FROM workspaces WHERE canonical_path = ?",
            (canonical_path,),
        )
        return self._workspace_payload_from_row(row)

    async def update(
        self,
        workspace_id: str,
        *,
        display_name: str | None = None,
        sort_order: int | None = None,
        metadata: dict[str, Any] | None = None,
        last_opened_at: str | None = None,
        is_archived: bool | None = None,
    ) -> dict[str, Any] | None:
        clean_id = str(workspace_id or "").strip()
        existing = await self.get(clean_id)
        if existing is None:
            return None

        updates: list[str] = ["updated_at = ?"]
        params: list[Any] = [_now_iso()]
        if display_name is not None:
            updates.append("display_name = ?")
            params.append(str(display_name or "").strip() or existing["display_name"])
        if sort_order is not None:
            updates.append("sort_order = ?")
            params.append(int(sort_order))
        if metadata is not None:
            merged = dict(existing.get("metadata", {}))
            merged.update(dict(metadata))
            updates.append("metadata = ?")
            params.append(json.dumps(merged, ensure_ascii=False, sort_keys=True))
        if last_opened_at is not None:
            updates.append("last_opened_at = ?")
            params.append(str(last_opened_at or "").strip())
        if is_archived is not None:
            updates.append("is_archived = ?")
            params.append(1 if bool(is_archived) else 0)

        params.append(clean_id)
        await self._db.execute(
            f"UPDATE workspaces SET {', '.join(updates)} WHERE id = ?",
            tuple(params),
        )
        return await self.get(clean_id)

    async def archive(self, workspace_id: str) -> bool:
        updated = await self.update(workspace_id, is_archived=True)
        return updated is not None

    async def get_workspace_settings(self, workspace_id: str) -> dict[str, Any]:
        row = await self._db.query_one(
            "SELECT * FROM workspace_settings WHERE workspace_id = ?",
            (str(workspace_id or "").strip(),),
        )
        if not row:
            return {
                "workspace_id": str(workspace_id or "").strip(),
                "overrides": {},
                "created_at": "",
                "updated_at": "",
            }
        return {
            "workspace_id": str(row.get("workspace_id", "") or ""),
            "overrides": _decode_json_object(row.get("settings_json")),
            "created_at": str(row.get("created_at", "") or ""),
            "updated_at": str(row.get("updated_at", "") or ""),
        }

    async def patch_workspace_settings(
        self,
        workspace_id: str,
        *,
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        clean_id = str(workspace_id or "").strip()
        current = await self.get_workspace_settings(clean_id)
        merged = dict(current.get("overrides", {}))
        merged.update(dict(overrides or {}))
        now = _now_iso()
        await self._db.execute(
            """
            INSERT INTO workspace_settings (workspace_id, settings_json, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(workspace_id) DO UPDATE SET
                settings_json = excluded.settings_json,
                updated_at = excluded.updated_at
            """,
            (
                clean_id,
                json.dumps(merged, ensure_ascii=False, sort_keys=True),
                now,
                now,
            ),
        )
        return await self.get_workspace_settings(clean_id)
