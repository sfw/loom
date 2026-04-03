"""Authoritative SQLite-backed state for auth-free search provider pacing."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from loom.state.memory import Database


@dataclass(frozen=True, slots=True)
class SearchProviderPolicy:
    name: str
    priority: int
    min_interval_seconds: float
    cooldown_seconds: float
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class ProviderDispatchDecision:
    status: str
    retry_at: float = 0.0
    reason: str = ""


class SearchProviderStateStore:
    """Single-authority provider pacing state backed by SQLite."""

    def __init__(
        self,
        db: Database,
        *,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self._db = db
        self._now = now_fn or time.time

    @classmethod
    async def from_database_path(
        cls,
        db_path: str | Path,
        *,
        now_fn: Callable[[], float] | None = None,
    ) -> SearchProviderStateStore:
        db = Database(str(db_path))
        await db.initialize()
        return cls(db, now_fn=now_fn)

    async def sync_policies(
        self,
        policies: Sequence[SearchProviderPolicy],
    ) -> None:
        rows = [
            (
                policy.name,
                int(policy.enabled),
                int(policy.priority),
                max(0.0, float(policy.min_interval_seconds)),
                float(self._now()),
            )
            for policy in policies
        ]
        if not rows:
            return
        await self._db.execute_many(
            """
            INSERT INTO search_provider_state (
                provider,
                enabled,
                priority,
                min_interval_seconds,
                updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(provider) DO UPDATE SET
                enabled=excluded.enabled,
                priority=excluded.priority,
                min_interval_seconds=excluded.min_interval_seconds,
                updated_at=excluded.updated_at
            """,
            rows,
        )

    async def ordered_provider_names(self) -> list[str]:
        rows = await self._db.query(
            """
            SELECT provider
            FROM search_provider_state
            WHERE enabled = 1
            ORDER BY priority DESC, provider ASC
            """,
        )
        return [str(row["provider"]) for row in rows]

    async def request_dispatch(
        self,
        policy: SearchProviderPolicy,
        *,
        lease_owner: str,
        lease_ttl_seconds: float,
    ) -> ProviderDispatchDecision:
        async def _callback(db) -> ProviderDispatchDecision:
            now = float(self._now())
            await db.execute(
                """
                INSERT INTO search_provider_state (
                    provider,
                    enabled,
                    priority,
                    min_interval_seconds,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(provider) DO UPDATE SET
                    enabled=excluded.enabled,
                    priority=excluded.priority,
                    min_interval_seconds=excluded.min_interval_seconds,
                    updated_at=excluded.updated_at
                """,
                (
                    policy.name,
                    int(policy.enabled),
                    int(policy.priority),
                    max(0.0, float(policy.min_interval_seconds)),
                    now,
                ),
            )
            cursor = await db.execute(
                """
                SELECT enabled,
                       next_allowed_at,
                       cooldown_until,
                       lease_owner,
                       lease_expires_at
                FROM search_provider_state
                WHERE provider = ?
                """,
                (policy.name,),
            )
            row = await cursor.fetchone()
            if row is None:
                return ProviderDispatchDecision(status="disabled", reason="missing")

            enabled = bool(int(row["enabled"] or 0))
            if not enabled:
                return ProviderDispatchDecision(status="disabled", reason="disabled")

            cooldown_until = max(0.0, float(row["cooldown_until"] or 0.0))
            if cooldown_until > now:
                return ProviderDispatchDecision(
                    status="cooldown",
                    retry_at=cooldown_until,
                    reason="cooldown",
                )

            active_lease_owner = str(row["lease_owner"] or "")
            lease_expires_at = max(0.0, float(row["lease_expires_at"] or 0.0))
            if (
                active_lease_owner
                and active_lease_owner != lease_owner
                and lease_expires_at > now
            ):
                return ProviderDispatchDecision(
                    status="wait",
                    retry_at=lease_expires_at,
                    reason="lease",
                )

            next_allowed_at = max(0.0, float(row["next_allowed_at"] or 0.0))
            if next_allowed_at > now:
                return ProviderDispatchDecision(
                    status="wait",
                    retry_at=next_allowed_at,
                    reason="interval",
                )

            await db.execute(
                """
                UPDATE search_provider_state
                SET lease_owner = ?,
                    lease_expires_at = ?,
                    next_allowed_at = ?,
                    last_started_at = ?,
                    updated_at = ?
                WHERE provider = ?
                """,
                (
                    lease_owner,
                    now + max(1.0, float(lease_ttl_seconds)),
                    now + max(0.0, float(policy.min_interval_seconds)),
                    now,
                    now,
                    policy.name,
                ),
            )
            return ProviderDispatchDecision(status="dispatch_now")

        result = await self._db.run_write_transaction(_callback)
        return result if isinstance(result, ProviderDispatchDecision) else ProviderDispatchDecision(
            status="disabled",
            reason="unexpected",
        )

    async def mark_success(
        self,
        provider_name: str,
        *,
        lease_owner: str,
        status_code: int = 200,
    ) -> None:
        now = float(self._now())
        await self._db.execute(
            """
            UPDATE search_provider_state
            SET cooldown_until = 0,
                lease_owner = CASE WHEN lease_owner = ? THEN '' ELSE lease_owner END,
                lease_expires_at = CASE WHEN lease_owner = ? THEN 0 ELSE lease_expires_at END,
                consecutive_failures = 0,
                soft_block_count = 0,
                last_status_code = ?,
                last_finished_at = ?,
                last_success_at = ?,
                updated_at = ?
            WHERE provider = ?
            """,
            (
                lease_owner,
                lease_owner,
                int(status_code),
                now,
                now,
                now,
                provider_name,
            ),
        )

    async def mark_failure(
        self,
        policy: SearchProviderPolicy,
        *,
        lease_owner: str,
        status_code: int | None,
        soft_block: bool,
    ) -> None:
        async def _callback(db) -> None:
            now = float(self._now())
            cursor = await db.execute(
                """
                SELECT consecutive_failures, soft_block_count
                FROM search_provider_state
                WHERE provider = ?
                """,
                (policy.name,),
            )
            row = await cursor.fetchone()
            failures = int(row["consecutive_failures"] or 0) if row is not None else 0
            soft_blocks = int(row["soft_block_count"] or 0) if row is not None else 0
            failures += 1
            if soft_block:
                soft_blocks += 1
            cooldown_until = 0.0
            if soft_block or status_code is None or status_code in {403, 429, 500, 502, 503, 504}:
                cooldown_until = now + max(0.0, float(policy.cooldown_seconds))
            await db.execute(
                """
                UPDATE search_provider_state
                SET cooldown_until = CASE
                        WHEN ? > cooldown_until THEN ?
                        ELSE cooldown_until
                    END,
                    lease_owner = CASE WHEN lease_owner = ? THEN '' ELSE lease_owner END,
                    lease_expires_at = CASE WHEN lease_owner = ? THEN 0 ELSE lease_expires_at END,
                    consecutive_failures = ?,
                    soft_block_count = ?,
                    last_status_code = ?,
                    last_finished_at = ?,
                    updated_at = ?
                WHERE provider = ?
                """,
                (
                    cooldown_until,
                    cooldown_until,
                    lease_owner,
                    lease_owner,
                    failures,
                    soft_blocks,
                    status_code,
                    now,
                    now,
                    policy.name,
                ),
            )

        await self._db.run_write_transaction(_callback)

    async def snapshot(self) -> list[dict]:
        return await self._db.query(
            """
            SELECT provider,
                   enabled,
                   priority,
                   min_interval_seconds,
                   next_allowed_at,
                   cooldown_until,
                   lease_owner,
                   lease_expires_at,
                   consecutive_failures,
                   soft_block_count,
                   last_status_code,
                   last_started_at,
                   last_finished_at,
                   last_success_at,
                   updated_at
            FROM search_provider_state
            ORDER BY priority DESC, provider ASC
            """,
        )

    @staticmethod
    def new_lease_owner() -> str:
        return uuid.uuid4().hex
