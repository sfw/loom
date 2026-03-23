"""Process-run and cowork ask_user interaction helpers."""

from __future__ import annotations

import asyncio
from typing import Any

from loom.cowork.session import ToolCallEvent
from loom.tools.ask_user import normalize_ask_user_args
from loom.tui.screens import AskUserScreen
from loom.tui.widgets import ChatLog


async def _prompt_process_run_question(
    self,
    *,
    run_id: str,
    question_payload: dict[str, Any],
) -> None:
    run = self._process_runs.get(run_id)
    if run is None or run.closed:
        return
    lock = self._process_run_question_locks.setdefault(run_id, asyncio.Lock())
    async with lock:
        run = self._process_runs.get(run_id)
        if run is None or run.closed:
            return
        payload = dict(question_payload or {})
        question_id = str(payload.get("question_id", "") or "").strip()
        if not question_id:
            return
        normalized = normalize_ask_user_args(payload)

        answer_event = asyncio.Event()
        answer_holder: list[dict[str, Any]] = []

        def handle_answer(answer_payload: object) -> None:
            if isinstance(answer_payload, dict):
                answer_holder.append(dict(answer_payload))
            answer_event.set()

        self.push_screen(
            AskUserScreen(
                str(normalized.get("question", "") or ""),
                options=list(normalized.get("legacy_options", []) or []),
                question_type=str(normalized.get("question_type", "free_text") or "free_text"),
                option_items=list(normalized.get("options", []) or []),
                allow_custom_response=bool(normalized.get("allow_custom_response", True)),
                min_selections=int(normalized.get("min_selections", 0) or 0),
                max_selections=int(normalized.get("max_selections", 0) or 0),
                context_note=str(normalized.get("context_note", "") or ""),
                urgency=str(normalized.get("urgency", "normal") or "normal"),
                default_option_id=str(normalized.get("default_option_id", "") or ""),
                return_payload=True,
                allow_cancel=False,
            ),
            callback=handle_answer,
        )

        self._begin_process_run_user_input_pause(run_id)
        try:
            await answer_event.wait()
        finally:
            self._end_process_run_user_input_pause(run_id)
        if not answer_holder:
            return
        answer_payload = dict(answer_holder[0])
        answer_payload["question_id"] = question_id
        answer_payload.setdefault("source", "tui")

        run = self._process_runs.get(run_id)
        if run is None or run.closed:
            return

        preview = str(answer_payload.get("custom_response", "") or "").strip()
        if not preview:
            labels = answer_payload.get("selected_labels", [])
            if isinstance(labels, list):
                preview = ", ".join(str(item).strip() for item in labels if str(item).strip())
        if not preview:
            option_ids = answer_payload.get("selected_option_ids", [])
            if isinstance(option_ids, list):
                preview = ", ".join(
                    str(item).strip() for item in option_ids if str(item).strip()
                )
        if not preview:
            preview = str(answer_payload.get("response_type", "") or "").strip()
        if preview:
            self._append_process_run_activity(
                run,
                f"Clarification answer submitted: {self._one_line(preview, 140)}",
            )

        response = await self._request_process_run_question_answer(
            run,
            question_id=question_id,
            answer_payload=answer_payload,
        )
        if bool(response.get("requested", False)):
            return

        seen = self._process_run_seen_questions.setdefault(run_id, set())
        seen.discard(question_id)
        error = str(response.get("error", "")).strip() or "Failed to submit answer."
        self._append_process_run_activity(
            run,
            f"Clarification answer failed: {self._one_line(error, 140)}",
        )
        self.notify(error, severity="error", timeout=4)

async def _handle_ask_user(self, event: ToolCallEvent) -> str:
    """Show an ask_user modal and return the answer."""
    normalized = normalize_ask_user_args(event.args if isinstance(event.args, dict) else {})
    question = str(normalized.get("question", "") or "")
    options = list(normalized.get("legacy_options", []) or [])

    answer_event = asyncio.Event()
    answer_holder: list[str] = []

    def handle_answer(answer: str) -> None:
        answer_holder.append(answer)
        answer_event.set()

    self.push_screen(
        AskUserScreen(
            question,
            options=options,
            question_type=str(normalized.get("question_type", "free_text") or "free_text"),
            option_items=list(normalized.get("options", []) or []),
            allow_custom_response=bool(normalized.get("allow_custom_response", True)),
            min_selections=int(normalized.get("min_selections", 0) or 0),
            max_selections=int(normalized.get("max_selections", 0) or 0),
            context_note=str(normalized.get("context_note", "") or ""),
            urgency=str(normalized.get("urgency", "normal") or "normal"),
            default_option_id=str(normalized.get("default_option_id", "") or ""),
        ),
        callback=handle_answer,
    )

    await answer_event.wait()
    answer = answer_holder[0] if answer_holder else ""

    if answer:
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_user_message(answer)
        await self._append_chat_replay_event(
            "user_message",
            {"text": answer},
        )

    return answer
