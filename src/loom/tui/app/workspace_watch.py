"""Workspace watch polling and debounced refresh helpers."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

from loom.tui.widgets import Sidebar

from .constants import _WORKSPACE_SCAN_EXCLUDE_DIRS

logger = logging.getLogger(__name__)

def _compute_workspace_signature(
    self,
) -> tuple[tuple[int, int, int] | None, bool]:
    """Return a bounded filesystem signature for change detection."""
    max_entries = self._tui_workspace_scan_max_entries()
    try:
        root = self._workspace.resolve()
    except OSError:
        return None, False

    file_count = 0
    newest_mtime_ns = 0
    total_size = 0
    overflow = False

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [
            name
            for name in sorted(dirnames)
            if name not in _WORKSPACE_SCAN_EXCLUDE_DIRS
            and not name.endswith(".egg-info")
        ]
        for filename in sorted(filenames):
            if file_count >= max_entries:
                overflow = True
                break
            path = Path(dirpath) / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            file_count += 1
            size = int(getattr(stat, "st_size", 0) or 0)
            mtime_ns = int(
                getattr(
                    stat,
                    "st_mtime_ns",
                    int(float(getattr(stat, "st_mtime", 0.0)) * 1_000_000_000),
                ),
            )
            total_size += size
            if mtime_ns > newest_mtime_ns:
                newest_mtime_ns = mtime_ns
        if overflow:
            break

    return (file_count, newest_mtime_ns, total_size), overflow

def _start_workspace_watch(self) -> None:
    """Start realtime workspace polling when enabled."""
    self._stop_workspace_watch()
    if not self._tui_realtime_refresh_enabled():
        return
    if not self.is_running:
        return

    # Avoid blocking startup on a full workspace walk; first poll seeds this.
    self._workspace_signature = None
    self._workspace_scan_overflow_notified = False

    backend = self._tui_workspace_watch_backend()
    if backend == "native":
        logger.info(
            "tui_workspace_watch: native backend requested; using poll fallback",
        )
    interval = self._tui_workspace_poll_interval_seconds()
    self._workspace_poll_timer = self.set_interval(
        interval,
        self._on_workspace_poll_tick,
    )

def _stop_workspace_watch(self) -> None:
    """Stop realtime workspace polling and clear pending refresh timers."""
    timer = self._workspace_poll_timer
    if timer is not None:
        try:
            timer.stop()
        except Exception:
            pass
    self._workspace_poll_timer = None
    self._workspace_poll_inflight = False
    self._cancel_workspace_refresh_timer()

def _on_workspace_poll_tick(self) -> None:
    """Poll workspace signature and schedule a debounced refresh on change."""
    if not self._tui_realtime_refresh_enabled():
        return
    if self._workspace_poll_inflight:
        return
    self._workspace_poll_inflight = True

    async def _scan() -> None:
        try:
            signature, overflow = await asyncio.to_thread(
                self._compute_workspace_signature,
            )
            if overflow and not self._workspace_scan_overflow_notified:
                self._workspace_scan_overflow_notified = True
                logger.info(
                    "tui_workspace_watch: scan cap reached at %s entries",
                    self._tui_workspace_scan_max_entries(),
                )
            elif not overflow:
                self._workspace_scan_overflow_notified = False

            if signature is None:
                return
            if self._workspace_signature is None:
                self._workspace_signature = signature
                return
            if signature != self._workspace_signature:
                self._workspace_signature = signature
                self._request_workspace_refresh("watch-poll")
        finally:
            self._workspace_poll_inflight = False

    self.run_worker(
        _scan(),
        group="workspace-watch-scan",
        exclusive=False,
    )

def _cancel_workspace_refresh_timer(self) -> None:
    timer = self._workspace_refresh_timer
    if timer is not None:
        try:
            timer.stop()
        except Exception:
            pass
    self._workspace_refresh_timer = None
    self._workspace_refresh_timer_pending = False

def _request_workspace_refresh(
    self,
    reason: str,
    *,
    immediate: bool = False,
) -> None:
    """Coalesce workspace refresh requests to avoid thrashing the UI."""
    clean_reason = str(reason or "").strip() or "unspecified"
    if not self.is_running:
        self._workspace_refresh_pending_reasons.clear()
        self._workspace_refresh_first_request_at = 0.0
        self._refresh_workspace_tree()
        return
    now = time.monotonic()
    if not self._workspace_refresh_pending_reasons:
        self._workspace_refresh_first_request_at = now
    self._workspace_refresh_pending_reasons.add(clean_reason)

    if immediate:
        self._flush_workspace_refresh_requests()
        return

    debounce = self._tui_workspace_refresh_debounce_seconds()
    max_wait = self._tui_workspace_refresh_max_wait_seconds()
    first = self._workspace_refresh_first_request_at or now
    elapsed = max(0.0, now - first)
    if elapsed >= max_wait:
        self._flush_workspace_refresh_requests()
        return

    self._cancel_workspace_refresh_timer()
    delay = min(debounce, max(0.01, max_wait - elapsed))
    self._workspace_refresh_timer_pending = True

    def _fire() -> None:
        self._workspace_refresh_timer_pending = False
        self._workspace_refresh_timer = None
        self._flush_workspace_refresh_requests()

    self._workspace_refresh_timer = self.set_timer(delay, _fire)

def _flush_workspace_refresh_requests(self) -> None:
    if not self._workspace_refresh_pending_reasons:
        return
    self._cancel_workspace_refresh_timer()
    reasons = sorted(self._workspace_refresh_pending_reasons)
    self._workspace_refresh_pending_reasons.clear()
    self._workspace_refresh_first_request_at = 0.0
    self._refresh_workspace_tree()
    self._refresh_process_command_index(background=True)
    logger.debug("tui_workspace_refresh reasons=%s", ",".join(reasons))

def _refresh_workspace_tree(self) -> None:
    """Reload sidebar workspace tree to pick up new files."""
    try:
        sidebar = self.query_one("#sidebar", Sidebar)
    except Exception:
        return
    sidebar.refresh_workspace_tree()

