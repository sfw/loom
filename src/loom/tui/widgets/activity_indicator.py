"""Animated activity indicator for header liveness state."""

from __future__ import annotations

import asyncio
import time

from textual.widgets import Static


class ActivityIndicator(Static):
    """Render dim static dots when idle and a moving head while active."""

    _GLYPH_IDLE = "[#4f6787]■[/]"
    _GLYPH_TRAIL = "[#7dcfff]■[/]"
    _GLYPH_HEAD = "[bold #00f0ff]■[/]"
    _GLYPH_IDLE_TOP = "[#4f6787]▄[/]"
    _GLYPH_TRAIL_TOP = "[#7dcfff]▄[/]"
    _GLYPH_HEAD_TOP = "[bold #00f0ff]▄[/]"
    _GLYPH_IDLE_MID = "[#4f6787]█[/]"
    _GLYPH_TRAIL_MID = "[#7dcfff]█[/]"
    _GLYPH_HEAD_MID = "[bold #00f0ff]█[/]"
    _GLYPH_IDLE_BOT = "[#4f6787]▀[/]"
    _GLYPH_TRAIL_BOT = "[#7dcfff]▀[/]"
    _GLYPH_HEAD_BOT = "[bold #00f0ff]▀[/]"

    DEFAULT_CSS = """
    ActivityIndicator {
        width: 8;
        min-width: 8;
        max-width: 8;
        height: 1;
        content-align: center middle;
        background: $panel;
        color: $foreground;
        text-opacity: 85%;
    }
    """

    def __init__(
        self,
        *,
        dot_count: int = 8,
        frame_interval_ms: int = 90,
        idle_hold_ms: int = 300,
        **kwargs,
    ) -> None:
        super().__init__("", **kwargs)
        self._dot_count = max(2, int(dot_count))
        self._frame_interval_s = max(0.04, int(frame_interval_ms) / 1000.0)
        self._idle_hold_s = max(0.0, int(idle_hold_ms) / 1000.0)
        self._frame_index = 0
        self._direction = 1
        self._active_requested = False
        self._hold_until = 0.0
        self._timer = None

    def on_unmount(self) -> None:
        self._stop_timer()

    def set_active(self, active: bool) -> None:
        """Set desired active state; idle transitions use a short hold window."""
        now = time.monotonic()
        requested = bool(active)
        was_requested = self._active_requested

        if requested:
            self._active_requested = True
            self._hold_until = 0.0
            self._ensure_timer()
            self.refresh()
            return

        self._active_requested = False

        # Already idle and not in a hold window: stay static.
        if not was_requested and not self._is_visually_active(now=now):
            self._hold_until = 0.0
            self._frame_index = 0
            self._direction = 1
            self._stop_timer()
            self.refresh()
            return

        # Transitioning from active -> idle gets one short hold window.
        if was_requested:
            if self._idle_hold_s <= 0:
                self._hold_until = 0.0
                self._frame_index = 0
                self._direction = 1
                self._stop_timer()
                self.refresh()
                return
            self._hold_until = now + self._idle_hold_s

        # If we're already in hold, do not extend it on repeated idle syncs.
        self._ensure_timer()
        self.refresh()

    def _is_visually_active(self, *, now: float | None = None) -> bool:
        ts = time.monotonic() if now is None else float(now)
        if self._active_requested:
            return True
        return ts < self._hold_until

    def _ensure_timer(self) -> None:
        if self._timer is not None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Unit tests may exercise the widget outside a live Textual loop.
            return
        self._timer = self.set_interval(self._frame_interval_s, self._on_tick)

    def _stop_timer(self) -> None:
        timer = self._timer
        if timer is None:
            return
        self._timer = None
        try:
            timer.stop()
        except Exception:
            pass

    def _advance_frame(self) -> None:
        if self._dot_count <= 1:
            self._frame_index = 0
            self._direction = 1
            return
        next_index = self._frame_index + self._direction
        if next_index >= self._dot_count:
            self._direction = -1
            next_index = self._dot_count - 2
        elif next_index < 0:
            self._direction = 1
            next_index = 1
        self._frame_index = max(0, min(next_index, self._dot_count - 1))

    def _on_tick(self) -> None:
        if not self._is_visually_active():
            self._frame_index = 0
            self._direction = 1
            self._stop_timer()
            self.refresh()
            return
        self._advance_frame()
        self.refresh()

    def _render_line(
        self,
        *,
        idle_glyph: str,
        trail_glyph: str,
        head_glyph: str,
    ) -> str:
        dots = [idle_glyph for _ in range(self._dot_count)]
        if self._is_visually_active():
            index = max(0, min(self._frame_index, self._dot_count - 1))
            left = index - 1
            right = index + 1
            if left >= 0:
                dots[left] = trail_glyph
            if right < self._dot_count:
                dots[right] = trail_glyph
            dots[index] = head_glyph
        return "".join(dots)

    def render(self) -> str:
        line = self._render_line(
            idle_glyph=self._GLYPH_IDLE,
            trail_glyph=self._GLYPH_TRAIL,
            head_glyph=self._GLYPH_HEAD,
        )
        height = max(1, int(getattr(self.size, "height", 1) or 1))
        if height <= 1:
            return line
        rows: list[str] = []
        for row_idx in range(height):
            if row_idx == 0:
                rows.append(
                    self._render_line(
                        idle_glyph=self._GLYPH_IDLE_TOP,
                        trail_glyph=self._GLYPH_TRAIL_TOP,
                        head_glyph=self._GLYPH_HEAD_TOP,
                    )
                )
                continue
            if row_idx == (height - 1):
                rows.append(
                    self._render_line(
                        idle_glyph=self._GLYPH_IDLE_BOT,
                        trail_glyph=self._GLYPH_TRAIL_BOT,
                        head_glyph=self._GLYPH_HEAD_BOT,
                    )
                )
                continue
            rows.append(
                self._render_line(
                    idle_glyph=self._GLYPH_IDLE_MID,
                    trail_glyph=self._GLYPH_TRAIL_MID,
                    head_glyph=self._GLYPH_HEAD_MID,
                )
            )
        return "\n".join(rows)
