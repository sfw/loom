"""Steering queue and chat-stop helpers."""

from __future__ import annotations

import asyncio
import logging
import textwrap
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static

from loom.tui.screens import ProcessRunCloseScreen
from loom.tui.widgets import ChatLog, EventPanel, StatusBar
from loom.utils.latency import log_latency_event

from ..constants import (
    _DEFAULT_TUI_CHAT_STOP_COOPERATIVE_WAIT_SECONDS,
    _DEFAULT_TUI_CHAT_STOP_SETTLE_TIMEOUT_SECONDS,
)
from ..models import SteeringDirective

logger = logging.getLogger(__name__)


def _now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


def new_steering_directive(
    *,
    kind: str,
    text: str,
    source: str,
    id_factory: Callable[[], str],
) -> SteeringDirective:
    """Build a normalized steering directive instance."""
    return SteeringDirective(
        id=id_factory(),
        kind=str(kind or "").strip().lower() or "inject",
        text=str(text or "").strip(),
        source=str(source or "").strip().lower() or "slash",
    )


def pop_pending_inject_directive(
    self,
    *,
    clear_session: bool = True,
) -> SteeringDirective | None:
    return self._remove_pending_inject_directive_at(
        0,
        clear_session=clear_session,
    )


def clear_pending_inject_directives(
    self,
    *,
    clear_session: bool = True,
) -> list[SteeringDirective]:
    previous = list(self._pending_inject_directives)
    self._pending_inject_directives = []
    if clear_session and self._session is not None:
        clear_pending = getattr(
            self._session,
            "clear_pending_inject_instruction",
            None,
        )
        if callable(clear_pending):
            clear_pending()
    self._refresh_hint_panel()
    self._sync_chat_stop_control()
    return previous


def pending_inject_directive_index(self, directive_id: str) -> int:
    clean = str(directive_id or "").strip()
    if not clean:
        return -1
    for idx, directive in enumerate(self._pending_inject_directives):
        if str(directive.id or "").strip() == clean:
            return idx
    return -1


def sync_session_pending_inject_queue(self) -> None:
    if self._session is None:
        return
    clear_pending = getattr(
        self._session,
        "clear_pending_inject_instruction",
        None,
    )
    if callable(clear_pending):
        clear_pending()
    queue_inject = getattr(self._session, "queue_inject_instruction", None)
    if not callable(queue_inject):
        return
    for directive in self._pending_inject_directives:
        text = str(getattr(directive, "text", "") or "").strip()
        if text:
            queue_inject(text)


def remove_pending_inject_directive_at(
    self,
    index: int,
    *,
    clear_session: bool = True,
) -> SteeringDirective | None:
    if index < 0 or index >= len(self._pending_inject_directives):
        return None
    directive = self._pending_inject_directives.pop(index)
    if clear_session:
        self._sync_session_pending_inject_queue()
    self._refresh_hint_panel()
    self._sync_chat_stop_control()
    return directive


def pop_next_queued_followup_directive(self) -> SteeringDirective | None:
    """Pop the next queued directive eligible for auto follow-up dispatch."""
    if not self._pending_inject_directives:
        return None
    directive = self._pending_inject_directives[0]
    source = str(getattr(directive, "source", "") or "").strip().lower()
    if source not in {"enter", "button"}:
        return None
    return self._remove_pending_inject_directive_at(
        0,
        clear_session=True,
    )


def start_queued_followup_turn(self, message: str) -> None:
    """Launch a queued follow-up prompt once chat is idle."""
    clean = str(message or "").strip()
    if not clean or self._session is None:
        return
    if self._chat_busy or self._chat_redirect_inflight or self._chat_stop_requested:
        return
    self._chat_turn_worker = self._run_turn(clean)


def take_input_text_for_steering(self) -> str:
    """Consume current input text and clear the input box."""
    input_widget = self.query_one("#user-input", Input)
    text = str(input_widget.value or "").strip()
    input_widget.value = ""
    self._reset_slash_tab_cycle()
    self._reset_input_history_navigation()
    self._sync_chat_stop_control()
    self._set_slash_hint("")
    return text


def render_steer_queue_status(self) -> str:
    """Render `/steer queue` summary for cowork steering state."""
    lines = ["[bold #7dcfff]Steering Queue[/]"]
    paused = bool(self._session and self._session.pause_requested)
    lines.append(f"  [bold]Pause:[/] {'requested' if paused else 'not requested'}")
    if not self._pending_inject_directives:
        lines.append("  [bold]Inject:[/] empty")
        return "\n".join(lines)
    age_seconds = int(self._pending_inject_age_seconds())
    lines.append(f"  [bold]Inject:[/] {self._pending_inject_count()} queued")
    lines.append(f"  [bold]Oldest:[/] {age_seconds}s ago")
    for idx, directive in enumerate(self._pending_inject_directives, start=1):
        preview = textwrap.shorten(
            " ".join(str(directive.text or "").split()),
            width=140,
            placeholder="…",
        )
        lines.append(f"  [bold]{idx}.[/] {self._escape_markup(preview)}")
    return "\n".join(lines)


def mark_cowork_tool_inflight(self, tool_name: str) -> None:
    clean = str(tool_name or "").strip()
    if not clean:
        return
    current = int(self._cowork_inflight_tool_counts.get(clean, 0) or 0)
    self._cowork_inflight_tool_counts[clean] = max(1, current + 1)


def clear_cowork_tool_inflight(self, tool_name: str) -> None:
    clean = str(tool_name or "").strip()
    if not clean:
        return
    current = int(self._cowork_inflight_tool_counts.get(clean, 0) or 0)
    if current <= 1:
        self._cowork_inflight_tool_counts.pop(clean, None)
        return
    self._cowork_inflight_tool_counts[clean] = current - 1


def cowork_inflight_mutating_tool_name(self) -> str:
    for tool_name, count in self._cowork_inflight_tool_counts.items():
        if int(count or 0) <= 0:
            continue
        if self._is_mutating_tool(tool_name):
            return tool_name
    return ""


async def record_steering_event(
    self,
    event_type: str,
    *,
    message: str = "",
    directive: SteeringDirective | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {}
    if message:
        payload["text"] = message
        payload["markup"] = True
        try:
            chat = self.query_one("#chat-log", ChatLog)
            chat.add_info(message)
        except Exception:
            pass
    if directive is not None:
        payload.update({
            "directive_id": directive.id,
            "directive_kind": directive.kind,
            "directive_text": directive.text,
            "source": directive.source,
            "status": directive.status,
            "created_at_monotonic": float(directive.created_at),
        })
    if extra:
        payload.update(extra)
    await self._append_chat_replay_event(
        str(event_type or "steer_failed").strip(),
        payload,
        render=False,
    )
    try:
        events_panel = self.query_one("#events-panel", EventPanel)
        label = str(event_type or "steer").replace("steer_", "steer:")
        detail = textwrap.shorten(
            " ".join(str(payload.get("text", "")).split()) or label,
            width=120,
            placeholder="…",
        )
        events_panel.add_event(_now_str(), "steer", f"{label} {detail}".strip())
    except Exception:
        pass


async def queue_chat_inject_instruction(
    self,
    text: str,
    *,
    source: str,
) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return False
    if self._session is None:
        await self._record_steering_event(
            "steer_failed",
            message="No active cowork chat session for inject.",
            extra={"result": "failed", "reason": "no_session", "source": source},
        )
        return False
    if not self._is_cowork_stop_visible():
        await self._record_steering_event(
            "steer_failed",
            message="No active cowork chat execution to inject into.",
            extra={"result": "failed", "reason": "not_busy", "source": source},
        )
        return False

    directive = self._new_steering_directive(
        kind="inject",
        text=clean,
        source=source,
    )
    self._pending_inject_directives.append(directive)
    self._session.queue_inject_instruction(clean)
    await self._record_steering_event(
        "steer_inject_queued",
        message=(
            "Queued inject instruction for next safe boundary. "
            f"Queue size: {self._pending_inject_count()}."
        ),
        directive=directive,
        extra={
            "result": "queued",
            "apply_mode": "safe_boundary",
            "queue_size": self._pending_inject_count(),
        },
    )
    log_latency_event(
        logger,
        event="steer_inject_queued",
        duration_seconds=0.0,
        fields={
            "session_id": self._active_session_id(),
            "turn_number": int(getattr(self._session, "_turn_counter", 0) or 0),
            "directive_id": directive.id,
            "directive_kind": directive.kind,
            "source": directive.source,
            "result": "queued",
            "queue_size": self._pending_inject_count(),
        },
    )
    self._sync_chat_stop_control()
    self._refresh_hint_panel()
    return True


async def sync_pending_inject_apply_state(self) -> None:
    if not self._pending_inject_directives or self._session is None:
        return
    local_count = self._pending_inject_count()
    count_value = getattr(self._session, "pending_inject_instruction_count", None)
    if count_value is not None:
        try:
            pending_count = int(count_value)
        except Exception:
            pending_count = (
                local_count
                if bool(getattr(self._session, "has_pending_inject_instruction", False))
                else 0
            )
    else:
        pending_count = (
            local_count
            if bool(getattr(self._session, "has_pending_inject_instruction", False))
            else 0
        )
    if pending_count >= local_count:
        return
    apply_count = max(0, local_count - pending_count)
    if apply_count <= 0:
        return
    applied: list[SteeringDirective] = []
    for _ in range(apply_count):
        if not self._pending_inject_directives:
            break
        directive = self._pending_inject_directives.pop(0)
        directive.status = "applied"
        applied.append(directive)
    for directive in applied:
        self._last_applied_directive_id = directive.id
        queued_ms = int(max(0.0, time.monotonic() - directive.created_at) * 1000)
        await self._record_steering_event(
            "steer_inject_applied",
            message="Applied queued inject instruction.",
            directive=directive,
            extra={
                "result": "applied",
                "queued_ms": queued_ms,
                "apply_mode": "safe_boundary",
                "queue_size": self._pending_inject_count(),
            },
        )
        log_latency_event(
            logger,
            event="steer_inject_applied",
            duration_seconds=max(0.0, float(queued_ms) / 1000.0),
            fields={
                "session_id": self._active_session_id(),
                "turn_number": int(getattr(self._session, "_turn_counter", 0) or 0),
                "directive_id": directive.id,
                "directive_kind": directive.kind,
                "source": directive.source,
                "queued_ms": queued_ms,
                "result": "applied",
                "queue_size": self._pending_inject_count(),
            },
        )
    self._sync_chat_stop_control()
    self._refresh_hint_panel()


async def request_chat_pause(self, *, source: str) -> bool:
    if self._session is None:
        await self._record_steering_event(
            "steer_failed",
            message="No active cowork chat session.",
            extra={"result": "failed", "reason": "no_session", "source": source},
        )
        return False
    if not self._is_cowork_stop_visible():
        await self._record_steering_event(
            "steer_failed",
            message="No active cowork chat execution to pause.",
            extra={"result": "failed", "reason": "not_busy", "source": source},
        )
        return False
    if self._session.pause_requested:
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info("Cowork chat is already paused.")
        return True
    self._session.request_pause()
    await self._record_steering_event(
        "steer_pause_requested",
        message="Pause requested. Cowork chat will pause at the next safe boundary.",
        extra={"result": "queued", "source": source},
    )
    log_latency_event(
        logger,
        event="steer_pause_requested",
        duration_seconds=0.0,
        fields={
            "session_id": self._active_session_id(),
            "turn_number": int(getattr(self._session, "_turn_counter", 0) or 0),
            "source": source,
            "result": "queued",
        },
    )
    self._sync_chat_stop_control()
    return True


async def request_chat_resume(self, *, source: str) -> bool:
    if self._session is None:
        await self._record_steering_event(
            "steer_failed",
            message="No active cowork chat session.",
            extra={"result": "failed", "reason": "no_session", "source": source},
        )
        return False
    if not self._session.pause_requested:
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info("Cowork chat is not paused.")
        return True
    self._session.request_resume()
    await self._record_steering_event(
        "steer_resumed",
        message="Cowork chat resumed.",
        extra={"result": "applied", "source": source},
    )
    log_latency_event(
        logger,
        event="steer_resumed",
        duration_seconds=0.0,
        fields={
            "session_id": self._active_session_id(),
            "turn_number": int(getattr(self._session, "_turn_counter", 0) or 0),
            "source": source,
            "result": "applied",
        },
    )
    self._sync_chat_stop_control()
    return True


async def clear_chat_steering(self, *, source: str) -> bool:
    had_pending = self._has_pending_inject()
    had_paused = bool(self._session and self._session.pause_requested)
    if had_pending:
        cleared = self._clear_pending_inject_directives(clear_session=True)
        for directive in cleared:
            directive.status = "dismissed"
        await self._record_steering_event(
            "steer_inject_dismissed",
            message=f"Cleared {len(cleared)} queued inject instruction(s).",
            extra={
                "result": "dismissed",
                "source": source,
                "queue_cleared_count": len(cleared),
            },
        )
        log_latency_event(
            logger,
            event="steer_inject_dismissed",
            duration_seconds=0.0,
            fields={
                "session_id": self._active_session_id(),
                "source": source,
                "result": "dismissed",
                "queue_cleared_count": len(cleared),
            },
        )
    if had_paused and self._session is not None:
        self._session.request_resume()
        await self._record_steering_event(
            "steer_resumed",
            message="Cowork chat resumed.",
            extra={"result": "applied", "source": source, "resume_reason": "clear"},
        )
    if not had_pending and not had_paused:
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info("No steering state to clear.")
    self._sync_chat_stop_control()
    self._refresh_hint_panel()
    return True


async def confirm_redirect_with_mutating_tool(self, tool_name: str) -> bool:
    clean_tool = str(tool_name or "").strip() or "mutating tool"
    waiter: asyncio.Future[bool] = asyncio.Future()
    screen = ProcessRunCloseScreen(
        run_label="cowork chat",
        running=True,
        prompt_override=(
            "[bold #e0af68]Redirect while mutating tool is in flight?[/]"
        ),
        detail_override=(
            f"Tool [bold]{self._escape_markup(clean_tool)}[/bold] is mutating files. "
            "Redirect will interrupt the current turn before rebasing the objective."
        ),
        confirm_label="Redirect",
        cancel_label="Keep Running",
        confirm_variant="warning",
    )

    def handle_result(confirmed: bool) -> None:
        if not waiter.done():
            waiter.set_result(bool(confirmed))

    self.push_screen(screen, callback=handle_result)
    try:
        timeout = self._tui_run_close_modal_timeout_seconds()
        return await asyncio.wait_for(waiter, timeout=timeout)
    except TimeoutError:
        try:
            screen.dismiss(False)
        except Exception:
            pass
        return False
    except asyncio.CancelledError:
        try:
            screen.dismiss(False)
        except Exception:
            pass
        raise


async def request_chat_redirect(
    self,
    text: str,
    *,
    source: str,
) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return False
    if self._session is None:
        await self._record_steering_event(
            "steer_failed",
            message="No active cowork chat session for redirect.",
            extra={"result": "failed", "reason": "no_session", "source": source},
        )
        return False

    directive = self._new_steering_directive(
        kind="redirect",
        text=clean,
        source=source,
    )
    pending = self._clear_pending_inject_directives(clear_session=True)
    if pending:
        for item in pending:
            item.status = "dismissed"
        await self._record_steering_event(
            "steer_inject_dismissed",
            message=(
                f"Cleared {len(pending)} queued inject instruction(s) "
                "before redirect."
            ),
            extra={
                "result": "dismissed",
                "source": source,
                "dismiss_reason": "redirect",
                "queue_cleared_count": len(pending),
            },
        )
    self._active_redirect_directive = directive
    await self._record_steering_event(
        "steer_redirect_requested",
        message="Redirect requested. Interrupting current cowork execution.",
        directive=directive,
        extra={"result": "requested", "apply_mode": "immediate"},
    )
    log_latency_event(
        logger,
        event="steer_redirect_requested",
        duration_seconds=0.0,
        fields={
            "session_id": self._active_session_id(),
            "turn_number": int(getattr(self._session, "_turn_counter", 0) or 0),
            "directive_id": directive.id,
            "directive_kind": directive.kind,
            "source": directive.source,
            "result": "requested",
        },
    )

    mutating_tool = self._cowork_inflight_mutating_tool_name()
    if mutating_tool:
        await self._record_steering_event(
            "steer_redirect_confirm_required",
            message=(
                "Redirect requires confirmation because a mutating tool "
                "is currently in flight."
            ),
            directive=directive,
            extra={"tool_name": mutating_tool, "result": "confirm_required"},
        )
        confirmed = await self._confirm_redirect_with_mutating_tool(mutating_tool)
        if not confirmed:
            self._active_redirect_directive = None
            await self._record_steering_event(
                "steer_failed",
                message="Redirect canceled. Continuing current cowork execution.",
                directive=directive,
                extra={
                    "result": "cancelled",
                    "reason": "redirect_confirm_denied",
                    "tool_name": mutating_tool,
                },
            )
            return False
        await self._record_steering_event(
            "steer_redirect_confirmed",
            message="Redirect confirmed.",
            directive=directive,
            extra={"tool_name": mutating_tool, "result": "confirmed"},
        )

    if self._is_cowork_stop_visible():
        await self._request_chat_stop()
        if self._is_cowork_stop_visible():
            self._active_redirect_directive = None
            self._steer_last_error = "redirect_stop_timeout"
            await self._record_steering_event(
                "steer_failed",
                message="Redirect failed because cowork execution did not stop.",
                directive=directive,
                extra={"result": "failed", "reason": "stop_timeout"},
            )
            return False

    # Redirect should release any pause gate and immediately drive a new turn.
    self._session.request_resume()
    directive.status = "applied"
    self._last_applied_directive_id = directive.id
    await self._record_steering_event(
        "steer_redirect_applied",
        message="Redirect applied. Rebasing cowork objective now.",
        directive=directive,
        extra={"result": "applied", "apply_mode": "immediate"},
    )
    log_latency_event(
        logger,
        event="steer_redirect_applied",
        duration_seconds=0.0,
        fields={
            "session_id": self._active_session_id(),
            "turn_number": int(getattr(self._session, "_turn_counter", 0) or 0),
            "directive_id": directive.id,
            "directive_kind": directive.kind,
            "source": directive.source,
            "result": "applied",
            "interrupt_path": self._chat_stop_last_path or "cooperative",
        },
    )
    self._active_redirect_directive = None
    self._chat_turn_worker = self._run_turn(clean)
    self._sync_activity_indicator()
    return True


def reset_cowork_steering_state(self, *, clear_session: bool = True) -> None:
    """Reset ephemeral cowork steering state."""
    self._cowork_inflight_tool_counts = {}
    self._active_redirect_directive = None
    self._pending_inject_directives = []
    self._last_applied_directive_id = ""
    self._steer_last_error = ""
    if clear_session and self._session is not None:
        request_resume = getattr(self._session, "request_resume", None)
        if callable(request_resume):
            request_resume()
        clear_pending = getattr(
            self._session,
            "clear_pending_inject_instruction",
            None,
        )
        if callable(clear_pending):
            clear_pending()
    self._sync_chat_stop_control()
    self._refresh_hint_panel()


def has_unfinalized_delegate_streams(self) -> bool:
    """Return True when a delegate progress stream is still in progress."""
    for stream in self._active_delegate_streams.values():
        if not isinstance(stream, dict):
            continue
        if not bool(stream.get("finalized", False)):
            return True
    return False


def is_background_work_active(self) -> bool:
    """Return True while chat, process runs, or delegate streams are active."""
    if self._chat_busy:
        return True
    if self._has_active_process_runs():
        return True
    return self._has_unfinalized_delegate_streams()


def is_cowork_stop_visible(self) -> bool:
    """Return True when cowork chat stop affordance should be visible."""
    if self._chat_busy:
        return True
    return self._has_unfinalized_delegate_streams()


def has_input_text(self) -> bool:
    """Return True when main input contains non-whitespace text."""
    try:
        value = self.query_one("#user-input", Input).value
    except Exception:
        return False
    return bool(str(value or "").strip())


def has_pending_inject(self) -> bool:
    return bool(self._pending_inject_directives)


def pending_inject_age_seconds(self) -> float:
    if not self._pending_inject_directives:
        return 0.0
    directive = self._pending_inject_directives[0]
    return max(0.0, time.monotonic() - float(directive.created_at))


def pending_inject_count(self) -> int:
    return len(self._pending_inject_directives)


def steer_queue_signature(self) -> tuple[tuple[str, str, str], ...]:
    """Stable signature for queued steering rows to avoid remount flicker."""
    signature: list[tuple[str, str, str]] = []
    for directive in self._pending_inject_directives:
        signature.append(
            (
                str(getattr(directive, "id", "") or ""),
                str(getattr(directive, "text", "") or ""),
                str(getattr(directive, "status", "") or ""),
            ),
        )
    return tuple(signature)


def current_steering_hint_text(self) -> str:
    current = ""
    try:
        selector = "#landing-input" if self._startup_landing_active else "#user-input"
        current = self.query_one(selector, Input).value
    except Exception:
        current = ""
    return self._render_slash_hint(current)


def should_show_steer_queue_popup(self) -> bool:
    return self._is_cowork_stop_visible() and self._has_pending_inject()


def render_steer_queue_popup(self) -> str:
    if not self._pending_inject_directives:
        return ""
    count = self._pending_inject_count()
    age_seconds = int(self._pending_inject_age_seconds())
    return f"[bold #7dcfff]Inject Queue ({count})[/] [dim]{age_seconds}s[/]"


def render_steer_queue_rows(self) -> None:
    """Render queued inject rows with per-item action buttons."""
    try:
        queue_list = self.query_one("#steer-queue-list", Vertical)
    except Exception:
        return
    signature = self._steer_queue_signature()
    for child in list(queue_list.children):
        child.remove()
    for idx, directive in enumerate(self._pending_inject_directives, start=1):
        preview = textwrap.shorten(
            " ".join(str(directive.text or "").split()),
            width=104,
            placeholder="…",
        )
        edit_btn = Button(
            "✎",
            id=f"steer-queue-edit-{directive.id}",
            classes="steer-queue-item-btn edit",
        )
        edit_btn.tooltip = "Edit queued instruction"
        redirect_btn = Button(
            "↪",
            id=f"steer-queue-redirect-{directive.id}",
            classes="steer-queue-item-btn redirect",
        )
        redirect_btn.tooltip = "Redirect now with this queued instruction"
        dismiss_btn = Button(
            "✕",
            id=f"steer-queue-dismiss-{directive.id}",
            classes="steer-queue-item-btn dismiss",
        )
        dismiss_btn.tooltip = "Dismiss this queued instruction"
        queue_list.mount(
            Horizontal(
                Static(
                    f"[#73daca]{idx}. {self._escape_markup(preview)}[/]",
                    classes="steer-queue-item-text",
                ),
                edit_btn,
                redirect_btn,
                dismiss_btn,
                classes="steer-queue-item-row",
            ),
        )
    self._last_rendered_steer_queue_signature = signature


def refresh_hint_panel(self) -> None:
    """Refresh slash/steering hint UI based on latest input and queue state."""
    try:
        self._set_slash_hint(self._current_steering_hint_text())
    except Exception:
        pass


def sync_chat_stop_control(self) -> None:
    """Keep input-row steering controls visibility/state in sync."""
    visible = self._is_cowork_stop_visible()
    show_steer_buttons = visible and self._has_input_text()
    try:
        stop_btn = self.query_one("#chat-stop-btn", Button)
        stop_btn.display = visible
        stop_btn.disabled = bool(
            visible
            and (self._chat_stop_requested or self._chat_redirect_inflight),
        )
        stop_btn.label = "■"
    except Exception:
        pass
    try:
        inject_btn = self.query_one("#chat-inject-btn", Button)
        inject_btn.display = show_steer_buttons
        inject_btn.disabled = bool(self._chat_redirect_inflight)
    except Exception:
        pass
    try:
        redirect_btn = self.query_one("#chat-redirect-btn", Button)
        redirect_btn.display = show_steer_buttons
        redirect_btn.disabled = bool(self._chat_redirect_inflight)
    except Exception:
        pass
    try:
        input_widget = self.query_one("#user-input", Input)
        add_class = getattr(input_widget, "add_class", None)
        remove_class = getattr(input_widget, "remove_class", None)
        if visible:
            if callable(remove_class):
                remove_class("no-chat-controls")
        elif callable(add_class):
            add_class("no-chat-controls")
    except Exception:
        pass


def chat_stop_cooperative_wait_seconds() -> float:
    return _DEFAULT_TUI_CHAT_STOP_COOPERATIVE_WAIT_SECONDS


def chat_stop_settle_timeout_seconds() -> float:
    return _DEFAULT_TUI_CHAT_STOP_SETTLE_TIMEOUT_SECONDS


async def wait_for_chat_turn_settle(self, *, timeout_seconds: float) -> bool:
    """Wait for current cowork turn to leave busy state."""
    timeout = max(0.05, float(timeout_seconds))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not self._chat_busy:
            return True
        await asyncio.sleep(0.05)
    return not self._chat_busy


async def finalize_unsettled_delegate_streams_for_stop(self) -> None:
    """Mark any in-progress cowork delegate streams as failed on interruption."""
    pending_ids: list[str] = []
    for key, stream in self._active_delegate_streams.items():
        if not isinstance(stream, dict):
            continue
        if bool(stream.get("finalized", False)):
            continue
        pending_ids.append(str(key))
    for tool_call_id in pending_ids:
        stream = self._active_delegate_streams.get(tool_call_id, {})
        elapsed_ms = 0
        if isinstance(stream, dict):
            started_at = float(stream.get("started_at", 0.0) or 0.0)
            if started_at > 0:
                elapsed_ms = max(1, int((time.monotonic() - started_at) * 1000))
        await self._finalize_delegate_progress_stream(
            tool_call_id=tool_call_id,
            success=False,
            elapsed_ms=elapsed_ms,
            persist=True,
        )


async def handle_interrupted_chat_turn(
    self,
    *,
    path: str,
    reason: str = "",
    stage: str = "",
) -> None:
    """Record and render one interrupted cowork turn."""
    await self._finalize_unsettled_delegate_streams_for_stop()
    clean_path = str(path or "").strip() or "unknown"
    clean_reason = str(reason or "").strip()
    clean_stage = str(stage or "").strip()
    self._chat_stop_last_path = clean_path
    self._chat_stop_last_error = clean_reason
    detail: list[str] = [f"path={clean_path}"]
    if clean_reason:
        detail.append(f"reason={clean_reason}")
    if clean_stage:
        detail.append(f"stage={clean_stage}")
    detail_text = ", ".join(detail)
    message = f"Stopped current chat execution. [dim]({self._escape_markup(detail_text)})[/dim]"
    chat = self.query_one("#chat-log", ChatLog)
    chat.add_info(message)
    await self._append_chat_replay_event(
        "turn_interrupted",
        {
            "message": message,
            "markup": True,
            "path": clean_path,
            "reason": clean_reason,
            "stage": clean_stage,
        },
        render=False,
    )
    try:
        events_panel = self.query_one("#events-panel", EventPanel)
        events_panel.add_event(_now_str(), "chat_stop", f"Interrupted ({clean_path})")
    except Exception:
        pass


async def request_chat_stop(self) -> None:
    """Request stop for active cowork chat turn using hybrid cancellation."""
    chat = self.query_one("#chat-log", ChatLog)
    status = self.query_one("#status-bar", StatusBar)

    def _clear_stop_request_state() -> None:
        self._chat_stop_requested = False
        if self._session is not None:
            self._session.clear_stop_request()
        status.state = "Thinking..." if self._chat_busy else "Ready"
        self._sync_activity_indicator()

    if not self._is_cowork_stop_visible():
        chat.add_info("No active cowork chat execution to stop.")
        return
    if self._chat_stop_requested:
        chat.add_info("Stop already requested. Waiting for chat turn to settle.")
        return
    self._chat_stop_requested = True
    self._chat_stop_requested_at = time.monotonic()
    self._chat_stop_last_path = ""
    self._chat_stop_last_error = ""
    status.state = "Stopping..."
    if self._session is not None:
        self._session.request_stop("user_requested")
    self._sync_activity_indicator()
    chat.add_info("Stop requested for active cowork chat turn.")
    log_latency_event(
        logger,
        event="chat_stop_requested",
        duration_seconds=0.0,
        fields={
            "session_id": self._active_session_id(),
            "turn_number": int(getattr(self._session, "_turn_counter", 0) or 0),
            "stop_path": "cooperative",
        },
    )

    if not self._chat_busy and self._has_unfinalized_delegate_streams():
        await self._handle_interrupted_chat_turn(
            path="cooperative",
            reason="user_requested",
            stage="delegate_stream_cleanup",
        )

    cooperative_wait = self._chat_stop_cooperative_wait_seconds()
    settled = await self._wait_for_chat_turn_settle(timeout_seconds=cooperative_wait)
    if settled:
        settled_path = self._chat_stop_last_path or "cooperative"
        log_latency_event(
            logger,
            event="chat_stop_settled",
            duration_seconds=max(0.0, time.monotonic() - self._chat_stop_requested_at),
            fields={
                "session_id": self._active_session_id(),
                "stop_path": settled_path,
                "result": "stopped",
            },
        )
        _clear_stop_request_state()
        return

    fallback_error = ""
    cancel_requested = False
    worker = self._chat_turn_worker
    if worker is not None and hasattr(worker, "cancel"):
        try:
            worker.cancel()
            cancel_requested = True
        except Exception as e:
            fallback_error = str(e)
    else:
        fallback_error = "No active chat worker cancellation path."

    log_latency_event(
        logger,
        event="chat_stop_ack",
        duration_seconds=max(0.0, time.monotonic() - self._chat_stop_requested_at),
        fields={
            "session_id": self._active_session_id(),
            "stop_path": "worker_fallback",
            "cancel_requested": bool(cancel_requested),
            "error": fallback_error,
        },
    )
    if fallback_error:
        chat.add_info(
            "[bold #e0af68]Stop fallback warning:[/] "
            f"{self._escape_markup(fallback_error)}"
        )

    settled = await self._wait_for_chat_turn_settle(
        timeout_seconds=self._chat_stop_settle_timeout_seconds(),
    )
    if settled:
        settled_path = self._chat_stop_last_path or "hybrid"
        log_latency_event(
            logger,
            event="chat_stop_settled",
            duration_seconds=max(0.0, time.monotonic() - self._chat_stop_requested_at),
            fields={
                "session_id": self._active_session_id(),
                "stop_path": settled_path,
                "result": "stopped",
            },
        )
        _clear_stop_request_state()
        return

    log_latency_event(
        logger,
        event="chat_stop_timeout",
        duration_seconds=max(0.0, time.monotonic() - self._chat_stop_requested_at),
        fields={
            "session_id": self._active_session_id(),
            "stop_path": "hybrid",
            "result": "timeout",
        },
    )
    chat.add_info(
        "[bold #e0af68]Stop timed out.[/] "
        "Chat is still running; try /stop again."
    )
    _clear_stop_request_state()


def action_stop_chat(self) -> None:
    """Start non-blocking stop flow for current cowork turn."""
    if self._chat_stop_inflight:
        return
    self._chat_stop_inflight = True

    async def _run_stop() -> None:
        try:
            await self._request_chat_stop()
        except Exception as e:
            log_latency_event(
                logger,
                event="chat_stop_failed",
                duration_seconds=max(0.0, time.monotonic() - self._chat_stop_requested_at),
                fields={
                    "session_id": self._active_session_id(),
                    "stop_path": self._chat_stop_last_path or "hybrid",
                    "result": "failed",
                    "error": str(e),
                },
            )
            logger.exception("chat_stop_failed session=%s", self._active_session_id())
            try:
                chat = self.query_one("#chat-log", ChatLog)
                chat.add_info(
                    "[bold #f7768e]Stop failed:[/] "
                    f"{self._escape_markup(str(e))}"
                )
            except Exception:
                pass
            self._chat_stop_requested = False
            if self._session is not None:
                self._session.clear_stop_request()
            try:
                status = self.query_one("#status-bar", StatusBar)
                status.state = "Thinking..." if self._chat_busy else "Ready"
            except Exception:
                pass
            self._sync_activity_indicator()
        finally:
            self._chat_stop_inflight = False

    self.run_worker(
        _run_stop(),
        group="chat-stop",
        exclusive=False,
    )


def action_inject_chat(self) -> None:
    """Queue an inject steering instruction from the input row."""
    text = self._take_input_text_for_steering()
    if not text:
        return

    async def _run_inject() -> None:
        try:
            await self._queue_chat_inject_instruction(text, source="button")
        except Exception as e:
            logger.exception("chat_inject_failed session=%s", self._active_session_id())
            await self._record_steering_event(
                "steer_failed",
                message=f"[bold #f7768e]Inject failed:[/] {self._escape_markup(str(e))}",
                extra={"result": "failed", "reason": "exception", "source": "button"},
            )

    self.run_worker(
        _run_inject(),
        group="chat-steer",
        exclusive=False,
    )


def action_redirect_chat(self) -> None:
    """Request immediate redirect steering from the input row."""
    text = self._take_input_text_for_steering()
    if not text:
        return
    if self._chat_redirect_inflight:
        return
    self._chat_redirect_inflight = True
    self._sync_chat_stop_control()

    async def _run_redirect() -> None:
        try:
            await self._request_chat_redirect(text, source="button")
        except Exception as e:
            logger.exception("chat_redirect_failed session=%s", self._active_session_id())
            await self._record_steering_event(
                "steer_failed",
                message=f"[bold #f7768e]Redirect failed:[/] {self._escape_markup(str(e))}",
                extra={"result": "failed", "reason": "exception", "source": "button"},
            )
        finally:
            self._chat_redirect_inflight = False
            self._sync_chat_stop_control()

    self.run_worker(
        _run_redirect(),
        group="chat-steer",
        exclusive=False,
    )


def action_steer_queue_edit(self, directive_id: str = "") -> None:
    """Bring queued inject text back to input for editing."""
    index = (
        self._pending_inject_directive_index(directive_id)
        if directive_id
        else 0
    )
    directive = self._remove_pending_inject_directive_at(
        index,
        clear_session=True,
    )
    if directive is None:
        return
    directive.status = "dismissed"
    self._set_user_input_text(directive.text)
    try:
        self.query_one("#user-input", Input).focus()
    except Exception:
        pass

    async def _run_edit() -> None:
        await self._record_steering_event(
            "steer_inject_dismissed",
            message="Moved queued inject instruction back to input for editing.",
            directive=directive,
            extra={"result": "dismissed", "source": "queue_popup", "dismiss_reason": "edit"},
        )

    self.run_worker(
        _run_edit(),
        group="chat-steer",
        exclusive=False,
    )


def action_steer_queue_dismiss(self, directive_id: str = "") -> None:
    """Dismiss queued inject directive from queue popup."""
    index = (
        self._pending_inject_directive_index(directive_id)
        if directive_id
        else 0
    )
    directive = self._remove_pending_inject_directive_at(
        index,
        clear_session=True,
    )
    if directive is None:
        return
    directive.status = "dismissed"

    async def _run_dismiss() -> None:
        await self._record_steering_event(
            "steer_inject_dismissed",
            message="Dismissed queued inject instruction.",
            directive=directive,
            extra={
                "result": "dismissed",
                "source": "queue_popup",
                "dismiss_reason": "dismiss",
            },
        )

    self.run_worker(
        _run_dismiss(),
        group="chat-steer",
        exclusive=False,
    )


def action_steer_queue_redirect(self, directive_id: str = "") -> None:
    """Use queued inject text as immediate redirect objective."""
    index = (
        self._pending_inject_directive_index(directive_id)
        if directive_id
        else 0
    )
    directive = self._remove_pending_inject_directive_at(
        index,
        clear_session=True,
    )
    if directive is None:
        return
    text = str(directive.text or "").strip()
    if not text:
        return
    if self._chat_redirect_inflight:
        return
    self._chat_redirect_inflight = True
    self._sync_chat_stop_control()

    async def _run_redirect() -> None:
        try:
            directive.status = "dismissed"
            await self._record_steering_event(
                "steer_inject_dismissed",
                message="Converted queued inject instruction to immediate redirect.",
                directive=directive,
                extra={
                    "result": "dismissed",
                    "source": "queue_popup",
                    "dismiss_reason": "redirect_now",
                },
            )
            await self._request_chat_redirect(text, source="queue_popup")
        except Exception as e:
            logger.exception(
                "queue_redirect_failed session=%s",
                self._active_session_id(),
            )
            await self._record_steering_event(
                "steer_failed",
                message=f"[bold #f7768e]Redirect failed:[/] {self._escape_markup(str(e))}",
                extra={
                    "result": "failed",
                    "reason": "exception",
                    "source": "queue_popup",
                },
            )
        finally:
            self._chat_redirect_inflight = False
            self._sync_chat_stop_control()

    self.run_worker(
        _run_redirect(),
        group="chat-steer",
        exclusive=False,
    )
