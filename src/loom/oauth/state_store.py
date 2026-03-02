"""Thread-safe OAuth pending-state store with replay/expiry protection."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

_MARKER_TTL_SECONDS = 900.0


class OAuthStateStoreError(Exception):
    """Raised when OAuth pending state is invalid or unavailable."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class OAuthPendingState:
    """Immutable snapshot for one pending OAuth state."""

    state: str
    code_verifier: str
    redirect_uri: str
    created_at: float
    expires_at: float
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class _PendingEntry:
    state: str
    code_verifier: str
    redirect_uri: str
    created_at: float
    expires_at: float
    metadata: dict[str, str]
    callback_payload: dict[str, str] | None = None


class OAuthStateStore:
    """Single-process OAuth state registry keyed strictly by OAuth state."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._pending: dict[str, _PendingEntry] = {}
        self._used_until: dict[str, float] = {}
        self._expired_until: dict[str, float] = {}
        self._cancelled_until: dict[str, float] = {}

    @property
    def pending_count(self) -> int:
        with self._lock:
            self._cleanup_locked()
            return len(self._pending)

    def create_pending(
        self,
        *,
        state: str,
        code_verifier: str,
        redirect_uri: str,
        ttl_seconds: int,
        metadata: dict[str, str] | None = None,
    ) -> OAuthPendingState:
        clean_state = str(state or "").strip()
        if not clean_state:
            raise OAuthStateStoreError("missing_state", "OAuth state cannot be empty.")
        clean_verifier = str(code_verifier or "").strip()
        if not clean_verifier:
            raise OAuthStateStoreError(
                "missing_verifier",
                "OAuth code verifier cannot be empty.",
            )
        clean_redirect_uri = str(redirect_uri or "").strip()
        if not clean_redirect_uri:
            raise OAuthStateStoreError(
                "missing_redirect_uri",
                "OAuth redirect URI cannot be empty.",
            )
        ttl = max(1, int(ttl_seconds))
        now = time.monotonic()
        with self._lock:
            self._cleanup_locked(now)
            if clean_state in self._used_until:
                raise OAuthStateStoreError(
                    "state_replayed",
                    "OAuth state has already been consumed.",
                )
            if clean_state in self._expired_until:
                raise OAuthStateStoreError(
                    "state_expired",
                    "OAuth state is no longer valid.",
                )
            if clean_state in self._cancelled_until:
                raise OAuthStateStoreError(
                    "state_cancelled",
                    "OAuth state was cancelled.",
                )
            if clean_state in self._pending:
                raise OAuthStateStoreError(
                    "state_conflict",
                    "OAuth state is already pending.",
                )
            entry = _PendingEntry(
                state=clean_state,
                code_verifier=clean_verifier,
                redirect_uri=clean_redirect_uri,
                created_at=now,
                expires_at=now + float(ttl),
                metadata=dict(metadata or {}),
            )
            self._pending[clean_state] = entry
            self._condition.notify_all()
            return self._snapshot(entry)

    def register_callback(
        self,
        *,
        state: str,
        payload: dict[str, str],
    ) -> str:
        clean_state = str(state or "").strip()
        if not clean_state:
            return "missing_state"
        now = time.monotonic()
        with self._lock:
            self._cleanup_locked(now)
            if clean_state in self._used_until:
                return "state_replayed"
            if clean_state in self._cancelled_until:
                return "state_cancelled"
            if clean_state in self._expired_until:
                return "state_expired"
            entry = self._pending.get(clean_state)
            if entry is None:
                return "state_unknown"
            if now >= entry.expires_at:
                self._pending.pop(clean_state, None)
                self._mark_expired_locked(clean_state, now)
                self._condition.notify_all()
                return "state_expired"
            if entry.callback_payload is not None:
                return "state_replayed"
            entry.callback_payload = dict(payload)
            self._condition.notify_all()
            return "ok"

    def await_callback(
        self,
        *,
        state: str,
        timeout_seconds: int,
    ) -> dict[str, str]:
        clean_state = str(state or "").strip()
        if not clean_state:
            raise OAuthStateStoreError("missing_state", "OAuth state cannot be empty.")
        deadline = time.monotonic() + max(1, int(timeout_seconds))
        with self._lock:
            while True:
                now = time.monotonic()
                self._cleanup_locked(now)
                entry = self._pending.get(clean_state)
                if entry is None:
                    if clean_state in self._cancelled_until:
                        raise OAuthStateStoreError(
                            "auth_cancelled",
                            "OAuth login was cancelled.",
                        )
                    if clean_state in self._expired_until:
                        raise OAuthStateStoreError(
                            "state_expired",
                            "OAuth state has expired.",
                        )
                    if clean_state in self._used_until:
                        raise OAuthStateStoreError(
                            "state_replayed",
                            "OAuth state has already been consumed.",
                        )
                    raise OAuthStateStoreError(
                        "state_unknown",
                        "OAuth state is not pending.",
                    )
                if entry.callback_payload is not None:
                    return dict(entry.callback_payload)
                if now >= entry.expires_at or now >= deadline:
                    self._pending.pop(clean_state, None)
                    self._mark_expired_locked(clean_state, now)
                    self._condition.notify_all()
                    raise OAuthStateStoreError(
                        "callback_timeout",
                        "Timed out waiting for OAuth callback.",
                    )
                wait_for = min(entry.expires_at, deadline) - now
                self._condition.wait(timeout=max(0.05, wait_for))

    def get_pending(self, *, state: str) -> OAuthPendingState:
        clean_state = str(state or "").strip()
        if not clean_state:
            raise OAuthStateStoreError("missing_state", "OAuth state cannot be empty.")
        now = time.monotonic()
        with self._lock:
            self._cleanup_locked(now)
            entry = self._pending.get(clean_state)
            if entry is None:
                if clean_state in self._expired_until:
                    raise OAuthStateStoreError(
                        "state_expired",
                        "OAuth state has expired.",
                    )
                if clean_state in self._used_until:
                    raise OAuthStateStoreError(
                        "state_replayed",
                        "OAuth state has already been consumed.",
                    )
                if clean_state in self._cancelled_until:
                    raise OAuthStateStoreError(
                        "state_cancelled",
                        "OAuth state was cancelled.",
                    )
                raise OAuthStateStoreError(
                    "state_unknown",
                    "OAuth state is not pending.",
                )
            if now >= entry.expires_at:
                self._pending.pop(clean_state, None)
                self._mark_expired_locked(clean_state, now)
                self._condition.notify_all()
                raise OAuthStateStoreError(
                    "state_expired",
                    "OAuth state has expired.",
                )
            return self._snapshot(entry)

    def complete(self, *, state: str) -> OAuthPendingState | None:
        clean_state = str(state or "").strip()
        if not clean_state:
            return None
        now = time.monotonic()
        with self._lock:
            self._cleanup_locked(now)
            entry = self._pending.pop(clean_state, None)
            self._condition.notify_all()
            if entry is None:
                return None
            self._mark_used_locked(clean_state, now)
            return self._snapshot(entry)

    def cancel(self, *, state: str) -> bool:
        clean_state = str(state or "").strip()
        if not clean_state:
            return False
        now = time.monotonic()
        with self._lock:
            self._cleanup_locked(now)
            entry = self._pending.pop(clean_state, None)
            self._condition.notify_all()
            if entry is None:
                return False
            self._mark_cancelled_locked(clean_state, now)
            return True

    def clear_all(self) -> None:
        with self._lock:
            self._pending.clear()
            self._condition.notify_all()

    @staticmethod
    def _snapshot(entry: _PendingEntry) -> OAuthPendingState:
        return OAuthPendingState(
            state=entry.state,
            code_verifier=entry.code_verifier,
            redirect_uri=entry.redirect_uri,
            created_at=entry.created_at,
            expires_at=entry.expires_at,
            metadata=dict(entry.metadata),
        )

    def _mark_used_locked(self, state: str, now: float) -> None:
        self._used_until[state] = now + _MARKER_TTL_SECONDS

    def _mark_expired_locked(self, state: str, now: float) -> None:
        self._expired_until[state] = now + _MARKER_TTL_SECONDS

    def _mark_cancelled_locked(self, state: str, now: float) -> None:
        self._cancelled_until[state] = now + _MARKER_TTL_SECONDS

    def _cleanup_locked(self, now: float | None = None) -> None:
        ts = time.monotonic() if now is None else now
        expired_pending = [
            state
            for state, entry in self._pending.items()
            if ts >= entry.expires_at
        ]
        for state in expired_pending:
            self._pending.pop(state, None)
            self._mark_expired_locked(state, ts)

        self._used_until = {
            state: until for state, until in self._used_until.items() if until > ts
        }
        self._expired_until = {
            state: until for state, until in self._expired_until.items() if until > ts
        }
        self._cancelled_until = {
            state: until
            for state, until in self._cancelled_until.items()
            if until > ts
        }
