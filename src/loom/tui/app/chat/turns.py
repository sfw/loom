"""Turn-loop related pure helpers."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime

from loom.cowork.session import CoworkStopRequestedError, CoworkTurn, ToolCallEvent
from loom.tui.widgets import ChatLog, EventPanel, StatusBar
from loom.tui.widgets.tool_call import tool_args_preview

from ..models import SteeringDirective


def _now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


def delegate_target_for_tool_call(tool_name: str, args: dict | None) -> str:
    """Return delegated target tool name when this call wraps another tool."""
    name = str(tool_name or "").strip()
    payload = args if isinstance(args, dict) else {}
    if name == "delegate_task":
        return "delegate_task"
    if name != "run_tool":
        return ""
    target = str(payload.get("name", payload.get("tool_name", "")) or "").strip()
    return target


def delegate_progress_title(caller_tool_name: str) -> str:
    caller = str(caller_tool_name or "").strip()
    if caller == "run_tool":
        return "Delegated progress (run_tool)"
    return "Delegated progress"

async def run_turn(self, user_message: str) -> None:
    if self._session is None:
        return

    followup_directive: SteeringDirective | None = None
    followup_message = ""
    self._chat_stop_last_path = ""
    self._chat_stop_last_error = ""
    self._chat_stop_requested = False
    self._cowork_inflight_tool_counts = {}
    self._session.clear_stop_request()
    self._chat_busy = True
    self._sync_activity_indicator()
    chat = self.query_one("#chat-log", ChatLog)
    status = self.query_one("#status-bar", StatusBar)

    chat.add_user_message(user_message)
    await self._append_chat_replay_event(
        "user_message",
        {"text": user_message},
    )
    status.state = "Thinking..."

    try:
        await self._run_interaction(user_message)
    except CoworkStopRequestedError as stop_exc:
        await self._handle_interrupted_chat_turn(
            path=stop_exc.path,
            reason=stop_exc.reason,
            stage=stop_exc.stage,
        )
    except asyncio.CancelledError:
        if self._chat_stop_requested or self._session.stop_requested:
            await self._handle_interrupted_chat_turn(
                path="worker_fallback",
                reason="user_requested",
            )
            return
        raise
    except Exception as e:
        chat.add_model_text(f"[bold #f7768e]Error:[/] {e}", markup=True)
        self.notify(str(e), severity="error", timeout=5)
    finally:
        await self._sync_pending_inject_apply_state()
        self._cowork_inflight_tool_counts = {}
        self._chat_busy = False
        self._chat_stop_requested = False
        self._chat_turn_worker = None
        self._session.clear_stop_request()
        status.state = "Ready"
        self._sync_activity_indicator()
        if not self._chat_stop_last_path and not self._chat_redirect_inflight:
            followup_directive = self._pop_next_queued_followup_directive()
            if followup_directive is not None:
                followup_message = str(followup_directive.text or "").strip()
    if followup_directive is not None and followup_message:
        followup_directive.status = "applied"
        queued_ms = int(max(0.0, time.monotonic() - followup_directive.created_at) * 1000)
        await self._record_steering_event(
            "steer_inject_applied",
            message="Dispatched queued prompt as follow-up turn.",
            directive=followup_directive,
            extra={
                "result": "applied",
                "queued_ms": queued_ms,
                "apply_mode": "queued_turn",
                "queue_size": self._pending_inject_count(),
            },
        )
        self.call_after_refresh(
            lambda message=followup_message: self._start_queued_followup_turn(message),
        )

async def run_interaction(self, message: str) -> None:
    """Execute a turn interaction with the model.

    Shared implementation for both initial turns and follow-ups.
    """
    chat = self.query_one("#chat-log", ChatLog)
    status = self.query_one("#status-bar", StatusBar)
    events_panel = self.query_one("#events-panel", EventPanel)

    streamed_text = False

    await self._sync_pending_inject_apply_state()
    async for event in self._session.send_streaming(message):
        await self._sync_pending_inject_apply_state()
        if isinstance(event, str):
            if not streamed_text:
                streamed_text = True
            chat.add_streaming_text(event)

        elif isinstance(event, ToolCallEvent):
            if event.result is None:
                # Tool starting
                self._mark_cowork_tool_inflight(event.name)
                chat.add_tool_call(
                    event.name,
                    event.args,
                    tool_call_id=event.tool_call_id,
                )
                await self._append_chat_replay_event(
                    "tool_call_started",
                    {
                        "tool_name": event.name,
                        "tool_call_id": event.tool_call_id,
                        "args": dict(event.args or {}),
                    },
                )
                delegated_target = self._delegate_target_for_tool_call(
                    event.name,
                    event.args,
                )
                if delegated_target == "delegate_task":
                    await self._start_delegate_progress_stream(
                        tool_call_id=event.tool_call_id,
                        caller_tool_name=event.name,
                        persist=True,
                    )
                status.state = f"Running {event.name}..."
                events_panel.add_event(
                    _now_str(), "tool_start",
                    (
                        f"{event.name} "
                        f"{tool_args_preview(event.name, event.args)}"
                    ),
                )
            else:
                # Tool completed
                self._clear_cowork_tool_inflight(event.name)
                output = ""
                if event.result.success:
                    output = event.result.output
                error = event.result.error or ""
                chat.add_tool_call(
                    event.name, event.args,
                    tool_call_id=event.tool_call_id,
                    success=event.result.success,
                    elapsed_ms=event.elapsed_ms,
                    output=output,
                    error=error,
                )
                await self._append_chat_replay_event(
                    "tool_call_completed",
                    {
                        "tool_name": event.name,
                        "tool_call_id": event.tool_call_id,
                        "args": dict(event.args or {}),
                        "success": bool(event.result.success),
                        "elapsed_ms": int(event.elapsed_ms or 0),
                        "output": output,
                        "error": error,
                    },
                )
                delegated_target = self._delegate_target_for_tool_call(
                    event.name,
                    event.args,
                )
                if delegated_target == "delegate_task":
                    await self._finalize_delegate_progress_stream(
                        tool_call_id=event.tool_call_id,
                        success=bool(event.result.success),
                        elapsed_ms=int(event.elapsed_ms or 0),
                        persist=True,
                    )

                # Show multimodal content indicators
                if (
                    event.result.content_blocks
                    and event.result.success
                ):
                    chat.add_content_indicator(
                        event.result.content_blocks,
                    )
                    from loom.content import serialize_block

                    await self._append_chat_replay_event(
                        "content_indicator",
                        {
                            "content_blocks": [
                                serialize_block(block)
                                for block in event.result.content_blocks
                            ],
                        },
                    )

                etype = (
                    "tool_ok"
                    if event.result.success
                    else "tool_err"
                )
                events_panel.add_event(
                    _now_str(), etype,
                    f"{event.name} {event.elapsed_ms}ms",
                )
                if event.result.success and self._is_mutating_tool(event.name):
                    self._request_workspace_refresh(f"tool:{event.name}")
                self._ingest_files_panel_from_tool_call_event(event)

                if event.result.data and (
                    event.name in {"task_tracker", "delegate_task"}
                    or delegated_target == "delegate_task"
                ):
                    self._update_sidebar_tasks(event.result.data)

                # Handle ask_user
                if (
                    event.name == "ask_user"
                    and event.result
                    and event.result.success
                ):
                    answer = await self._handle_ask_user(event)
                    if answer:
                        await self._run_followup(answer)

        elif isinstance(event, CoworkTurn):
            if event.text and not streamed_text:
                chat.add_model_text(event.text)
            if event.text:
                await self._append_chat_replay_event(
                    "assistant_text",
                    {
                        "text": event.text,
                        "markup": False,
                    },
                )

            self._total_tokens += event.tokens_used
            status = self.query_one("#status-bar", StatusBar)
            status.total_tokens = self._total_tokens

            chat.add_turn_separator(
                len(event.tool_calls),
                event.tokens_used,
                event.model,
                tokens_per_second=event.tokens_per_second,
                latency_ms=event.latency_ms,
                total_time_ms=event.total_time_ms,
                context_tokens=event.context_tokens,
                context_messages=event.context_messages,
                omitted_messages=event.omitted_messages,
                recall_index_used=event.recall_index_used,
            )
            await self._append_chat_replay_event(
                "turn_separator",
                {
                    "tool_count": len(event.tool_calls),
                    "tokens": int(event.tokens_used),
                    "model": event.model,
                    "tokens_per_second": float(event.tokens_per_second),
                    "latency_ms": int(event.latency_ms),
                    "total_time_ms": int(event.total_time_ms),
                    "context_tokens": int(event.context_tokens),
                    "context_messages": int(event.context_messages),
                    "omitted_messages": int(event.omitted_messages),
                    "recall_index_used": bool(event.recall_index_used),
                },
                journal_through_turn=self._session.persisted_turn_count,
            )
            events_panel.add_event(
                _now_str(), "turn",
                f"{event.tokens_used} tokens",
            )
            events_panel.record_turn_tokens(event.tokens_used)

            # Update files panel
            self._update_files_panel(event)
    await self._sync_pending_inject_apply_state()
