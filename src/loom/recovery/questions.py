"""Question manager for durable ask_user clarification flow."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from loom.events.bus import Event, EventBus
from loom.events.types import (
    ASK_USER_ANSWERED,
    ASK_USER_CANCELLED,
    ASK_USER_REQUESTED,
    ASK_USER_TIMEOUT,
)
from loom.state.memory import MemoryManager
from loom.tools.ask_user import normalize_ask_user_args


class QuestionStatus(StrEnum):
    PENDING = "pending"
    ANSWERED = "answered"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class QuestionRequest:
    """Normalized question request payload from ask_user tool args."""

    question: str
    question_type: str = "free_text"
    options: list[dict[str, str]] = field(default_factory=list)
    allow_custom_response: bool = True
    min_selections: int = 0
    max_selections: int = 0
    context_note: str = ""
    urgency: str = "normal"
    default_option_id: str = ""
    timeout_policy: str = "block"  # block|timeout_default|fail_closed
    timeout_seconds: int = 0
    timeout_default_response: str | list[str] | None = None
    tool_call_id: str = ""
    retry_attempt: int = 0
    question_id: str = ""

    @classmethod
    def from_ask_user_args(
        cls,
        args: dict[str, Any],
        *,
        timeout_policy: str = "block",
        timeout_seconds: int = 0,
        timeout_default_response: str | list[str] | None = None,
        tool_call_id: str = "",
        retry_attempt: int = 0,
    ) -> QuestionRequest:
        normalized = normalize_ask_user_args(args)
        return cls(
            question=str(normalized.get("question", "") or "").strip(),
            question_type=str(normalized.get("question_type", "free_text") or "free_text"),
            options=[
                {
                    "id": str(opt.get("id", "")).strip(),
                    "label": str(opt.get("label", "")).strip(),
                    "description": str(opt.get("description", "")).strip(),
                }
                for opt in list(normalized.get("options", []) or [])
                if isinstance(opt, dict)
            ],
            allow_custom_response=bool(normalized.get("allow_custom_response", True)),
            min_selections=max(0, int(normalized.get("min_selections", 0) or 0)),
            max_selections=max(0, int(normalized.get("max_selections", 0) or 0)),
            context_note=str(normalized.get("context_note", "") or "").strip(),
            urgency=str(normalized.get("urgency", "normal") or "normal").strip(),
            default_option_id=str(normalized.get("default_option_id", "") or "").strip(),
            timeout_policy=str(timeout_policy or "block").strip().lower(),
            timeout_seconds=max(0, int(timeout_seconds or 0)),
            timeout_default_response=timeout_default_response,
            tool_call_id=str(tool_call_id or "").strip(),
            retry_attempt=max(0, int(retry_attempt or 0)),
        )

    def to_payload(self, *, question_id: str) -> dict[str, Any]:
        timeout_at = ""
        if self.timeout_seconds > 0:
            timeout_at = (datetime.now(UTC) + timedelta(seconds=self.timeout_seconds)).isoformat()
        return {
            "question_id": question_id,
            "question": self.question,
            "question_type": self.question_type,
            "options": [dict(opt) for opt in self.options],
            "allow_custom_response": bool(self.allow_custom_response),
            "min_selections": int(self.min_selections),
            "max_selections": int(self.max_selections),
            "context_note": self.context_note,
            "urgency": self.urgency,
            "default_option_id": self.default_option_id,
            "timeout_policy": self.timeout_policy,
            "timeout_seconds": int(self.timeout_seconds),
            "timeout_default_response": self.timeout_default_response,
            "timeout_at": timeout_at,
            "tool_call_id": self.tool_call_id,
            "retry_attempt": int(self.retry_attempt),
        }


@dataclass
class QuestionAnswer:
    """Resolved answer payload returned to the runner."""

    question_id: str
    status: QuestionStatus
    response_type: str
    selected_option_ids: list[str] = field(default_factory=list)
    selected_labels: list[str] = field(default_factory=list)
    custom_response: str = ""
    answered_at: str = ""
    source: str = ""
    answered_by: str = ""
    client_id: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question_id": self.question_id,
            "response_type": self.response_type,
            "selected_option_ids": list(self.selected_option_ids),
            "selected_labels": list(self.selected_labels),
            "custom_response": self.custom_response,
            "answered_at": self.answered_at,
            "source": self.source,
        }
        if self.answered_by:
            payload["answered_by"] = self.answered_by
        if self.client_id:
            payload["client_id"] = self.client_id
        return payload

    @property
    def text_response(self) -> str:
        custom = str(self.custom_response or "").strip()
        if custom:
            return custom
        if self.selected_labels:
            return ", ".join(self.selected_labels)
        if self.selected_option_ids:
            return ", ".join(self.selected_option_ids)
        return ""

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> QuestionAnswer:
        answer = row.get("answer_payload")
        if not isinstance(answer, dict):
            answer = {}
        status_raw = str(row.get("status", QuestionStatus.PENDING.value) or "").strip().lower()
        if status_raw not in {status.value for status in QuestionStatus}:
            status_raw = QuestionStatus.PENDING.value
        return cls(
            question_id=str(row.get("question_id", "") or "").strip(),
            status=QuestionStatus(status_raw),
            response_type=str(answer.get("response_type", "") or "").strip(),
            selected_option_ids=[
                str(item or "").strip()
                for item in list(answer.get("selected_option_ids", []) or [])
                if str(item or "").strip()
            ],
            selected_labels=[
                str(item or "").strip()
                for item in list(answer.get("selected_labels", []) or [])
                if str(item or "").strip()
            ],
            custom_response=str(answer.get("custom_response", "") or "").strip(),
            answered_at=str(
                answer.get("answered_at", row.get("resolved_at", "")) or "",
            ).strip(),
            source=str(answer.get("source", "") or "").strip(),
            answered_by=str(answer.get("answered_by", "") or "").strip(),
            client_id=str(answer.get("client_id", "") or "").strip(),
        )


class QuestionManager:
    """Manages ask_user question lifecycle for orchestrated task execution."""

    def __init__(
        self,
        event_bus: EventBus,
        memory_manager: MemoryManager,
        *,
        poll_interval_seconds: float = 0.1,
    ) -> None:
        self._events = event_bus
        self._memory = memory_manager
        self._poll_interval_seconds = max(0.05, float(poll_interval_seconds))
        self._pending_events: dict[str, asyncio.Event] = {}

    @staticmethod
    def deterministic_question_id(
        *,
        task_id: str,
        subtask_id: str,
        tool_call_id: str,
        retry_attempt: int = 0,
    ) -> str:
        seed = "|".join([
            str(task_id or "").strip(),
            str(subtask_id or "").strip(),
            str(tool_call_id or "").strip(),
            str(max(0, int(retry_attempt or 0))),
        ])
        digest = hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()[:20]
        return f"q_{digest}"

    @staticmethod
    def deterministic_question_id_for_request(
        *,
        task_id: str,
        subtask_id: str,
        request: QuestionRequest,
    ) -> str:
        """Fallback deterministic ID when tool_call_id is unavailable."""
        canonical = {
            "task_id": str(task_id or "").strip(),
            "subtask_id": str(subtask_id or "").strip(),
            "question": str(request.question or "").strip(),
            "question_type": str(request.question_type or "").strip().lower(),
            "options": [
                {
                    "id": str(item.get("id", "")).strip(),
                    "label": str(item.get("label", "")).strip(),
                    "description": str(item.get("description", "")).strip(),
                }
                for item in list(request.options or [])
                if isinstance(item, dict)
            ],
            "allow_custom_response": bool(request.allow_custom_response),
            "min_selections": int(request.min_selections),
            "max_selections": int(request.max_selections),
            "context_note": str(request.context_note or "").strip(),
            "urgency": str(request.urgency or "").strip().lower(),
            "default_option_id": str(request.default_option_id or "").strip(),
            "timeout_policy": str(request.timeout_policy or "").strip().lower(),
            "timeout_seconds": int(request.timeout_seconds),
            "timeout_default_response": request.timeout_default_response,
            "retry_attempt": int(request.retry_attempt),
        }
        serialized = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        digest = hashlib.sha1(serialized.encode("utf-8", errors="ignore")).hexdigest()[:20]
        return f"q_{digest}"

    async def request_question(
        self,
        *,
        task_id: str,
        subtask_id: str,
        request: QuestionRequest,
        check_task_control: Callable[[], str] | None = None,
    ) -> QuestionAnswer:
        clean_task_id = str(task_id or "").strip()
        clean_subtask_id = str(subtask_id or "").strip()
        if not clean_task_id:
            raise ValueError("task_id is required")
        if not clean_subtask_id:
            raise ValueError("subtask_id is required")
        if not request.question.strip():
            raise ValueError("question is required")

        if request.question_id:
            question_id = str(request.question_id).strip()
        elif request.tool_call_id:
            question_id = self.deterministic_question_id(
                task_id=clean_task_id,
                subtask_id=clean_subtask_id,
                tool_call_id=request.tool_call_id,
                retry_attempt=request.retry_attempt,
            )
        else:
            question_id = self.deterministic_question_id_for_request(
                task_id=clean_task_id,
                subtask_id=clean_subtask_id,
                request=request,
            )
            if not question_id:
                question_id = f"q_{uuid.uuid4().hex[:20]}"

        timeout_policy = str(request.timeout_policy or "block").strip().lower()
        if timeout_policy not in {"block", "timeout_default", "fail_closed"}:
            timeout_policy = "block"

        pending = await self._memory.upsert_pending_task_question(
            question_id=question_id,
            task_id=clean_task_id,
            subtask_id=clean_subtask_id,
            request_payload=request.to_payload(question_id=question_id),
            timeout_at=(
                (datetime.now(UTC) + timedelta(seconds=request.timeout_seconds)).isoformat()
                if request.timeout_seconds > 0 and timeout_policy != "block"
                else ""
            ),
        )
        pending_question_id = str(pending.get("question_id", "") or "").strip()
        if pending_question_id:
            question_id = pending_question_id

        current_status = str(pending.get("status", "") or "").strip().lower()
        if current_status != QuestionStatus.PENDING.value:
            return QuestionAnswer.from_row(pending)

        self._events.emit(Event(
            event_type=ASK_USER_REQUESTED,
            task_id=clean_task_id,
            data={
                "subtask_id": clean_subtask_id,
                "question_id": question_id,
                **(
                    pending.get("request_payload", {})
                    if isinstance(pending.get("request_payload"), dict)
                    else {}
                ),
            },
        ))

        event = self._pending_events.setdefault(question_id, asyncio.Event())
        while True:
            row = await self._memory.get_task_question(clean_task_id, question_id)
            if row is None:
                raise RuntimeError(f"Question disappeared while waiting: {question_id}")

            status = str(row.get("status", "") or "").strip().lower()
            if status != QuestionStatus.PENDING.value:
                return QuestionAnswer.from_row(row)

            if callable(check_task_control):
                task_status = str(check_task_control() or "").strip().lower()
                if task_status == "cancelled":
                    resolved = await self._resolve_cancelled(
                        task_id=clean_task_id,
                        subtask_id=clean_subtask_id,
                        question_id=question_id,
                    )
                    if resolved is not None:
                        return resolved

            if timeout_policy != "block" and request.timeout_seconds > 0:
                created_at = _parse_iso_timestamp(str(row.get("created_at", "") or ""))
                if created_at is not None:
                    elapsed = (datetime.now(UTC) - created_at).total_seconds()
                    if elapsed >= float(request.timeout_seconds):
                        timed_out = await self._resolve_timeout(
                            task_id=clean_task_id,
                            subtask_id=clean_subtask_id,
                            question_id=question_id,
                            request=request,
                            timeout_policy=timeout_policy,
                        )
                        if timed_out is not None:
                            return timed_out

            if event.is_set():
                event.clear()
                continue
            await asyncio.sleep(self._poll_interval_seconds)

    async def answer_question(
        self,
        *,
        task_id: str,
        question_id: str,
        answer_payload: dict[str, Any],
    ) -> dict | None:
        clean_task_id = str(task_id or "").strip()
        clean_question_id = str(question_id or "").strip()
        row = await self._memory.get_task_question(clean_task_id, clean_question_id)
        if row is None:
            return None

        status = str(row.get("status", "") or "").strip().lower()
        if status != QuestionStatus.PENDING.value:
            return row

        request_payload = row.get("request_payload", {})
        if not isinstance(request_payload, dict):
            request_payload = {}
        normalized_answer = _normalize_answer_payload(
            request_payload=request_payload,
            answer_payload=answer_payload,
            fallback_source="api",
            response_type_hint="answered",
        )
        resolved = await self._memory.resolve_task_question(
            task_id=clean_task_id,
            question_id=clean_question_id,
            status=QuestionStatus.ANSWERED.value,
            answer_payload=normalized_answer,
            resolved_at=str(normalized_answer.get("answered_at", "") or ""),
        )
        if resolved is None:
            return None

        self._events.emit(Event(
            event_type=ASK_USER_ANSWERED,
            task_id=clean_task_id,
            data={
                "subtask_id": str(resolved.get("subtask_id", "") or "").strip(),
                "question_id": clean_question_id,
                "answer": dict(normalized_answer),
            },
        ))
        self._signal_question(clean_question_id)
        return resolved

    async def list_pending_questions(self, task_id: str) -> list[dict]:
        clean_task_id = str(task_id or "").strip()
        if not clean_task_id:
            return []
        return await self._memory.list_pending_task_questions(clean_task_id)

    async def get_question(self, task_id: str, question_id: str) -> dict | None:
        clean_task_id = str(task_id or "").strip()
        clean_question_id = str(question_id or "").strip()
        if not clean_task_id or not clean_question_id:
            return None
        return await self._memory.get_task_question(clean_task_id, clean_question_id)

    async def _resolve_timeout(
        self,
        *,
        task_id: str,
        subtask_id: str,
        question_id: str,
        request: QuestionRequest,
        timeout_policy: str,
    ) -> QuestionAnswer | None:
        answer_payload: dict[str, Any] | None = None
        if timeout_policy == "timeout_default":
            answer_payload = _build_timeout_default_payload(request)
            if answer_payload is None:
                timeout_policy = "fail_closed"
        if timeout_policy == "fail_closed":
            answer_payload = {
                "question_id": question_id,
                "response_type": "timeout",
                "selected_option_ids": [],
                "selected_labels": [],
                "custom_response": "",
                "answered_at": datetime.now(UTC).isoformat(),
                "source": "policy_default",
            }

        if answer_payload is None:
            return None
        answer_payload["question_id"] = question_id

        resolved = await self._memory.resolve_task_question(
            task_id=task_id,
            question_id=question_id,
            status=QuestionStatus.TIMEOUT.value,
            answer_payload=answer_payload,
            resolved_at=str(answer_payload.get("answered_at", "") or ""),
        )
        if resolved is None:
            return None
        self._events.emit(Event(
            event_type=ASK_USER_TIMEOUT,
            task_id=task_id,
            data={
                "subtask_id": subtask_id,
                "question_id": question_id,
                "answer": dict(answer_payload),
            },
        ))
        self._signal_question(question_id)
        return QuestionAnswer.from_row(resolved)

    async def _resolve_cancelled(
        self,
        *,
        task_id: str,
        subtask_id: str,
        question_id: str,
    ) -> QuestionAnswer | None:
        payload = {
            "question_id": question_id,
            "response_type": "cancelled",
            "selected_option_ids": [],
            "selected_labels": [],
            "custom_response": "",
            "answered_at": datetime.now(UTC).isoformat(),
            "source": "policy_default",
        }
        resolved = await self._memory.resolve_task_question(
            task_id=task_id,
            question_id=question_id,
            status=QuestionStatus.CANCELLED.value,
            answer_payload=payload,
            resolved_at=str(payload.get("answered_at", "") or ""),
        )
        if resolved is None:
            return None
        self._events.emit(Event(
            event_type=ASK_USER_CANCELLED,
            task_id=task_id,
            data={
                "subtask_id": subtask_id,
                "question_id": question_id,
                "answer": dict(payload),
            },
        ))
        self._signal_question(question_id)
        return QuestionAnswer.from_row(resolved)

    def _signal_question(self, question_id: str) -> None:
        event = self._pending_events.pop(question_id, None)
        if event is not None:
            event.set()


def _parse_iso_timestamp(raw: str) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _normalize_answer_payload(
    *,
    request_payload: dict[str, Any],
    answer_payload: dict[str, Any],
    fallback_source: str,
    response_type_hint: str,
) -> dict[str, Any]:
    payload = dict(answer_payload or {})
    question_id = str(
        payload.get("question_id", request_payload.get("question_id", "")) or "",
    ).strip()
    question_type = str(
        request_payload.get("question_type", "free_text") or "free_text",
    ).strip().lower()
    allow_custom = bool(request_payload.get("allow_custom_response", True))
    options = request_payload.get("options", [])
    if not isinstance(options, list):
        options = []
    options_by_id = {
        str(item.get("id", "")).strip(): str(item.get("label", "")).strip()
        for item in options
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    }
    labels_to_ids = {
        str(item.get("label", "")).strip().lower(): str(item.get("id", "")).strip()
        for item in options
        if (
            isinstance(item, dict)
            and str(item.get("id", "")).strip()
            and str(item.get("label", "")).strip()
        )
    }

    selected_option_ids = [
        str(item or "").strip()
        for item in list(payload.get("selected_option_ids", []) or [])
        if str(item or "").strip()
    ]
    if not selected_option_ids:
        selected_labels = [
            str(item or "").strip()
            for item in list(payload.get("selected_labels", []) or [])
            if str(item or "").strip()
        ]
        for label in selected_labels:
            mapped = labels_to_ids.get(label.lower())
            if mapped:
                selected_option_ids.append(mapped)

    custom_response = str(
        payload.get("custom_response", payload.get("text", payload.get("answer", ""))) or "",
    ).strip()
    source = str(payload.get("source", fallback_source) or fallback_source).strip()
    answered_by = str(payload.get("answered_by", "") or "").strip()
    client_id = str(payload.get("client_id", "") or "").strip()
    response_type = str(payload.get("response_type", "") or "").strip().lower()
    if not response_type:
        if len(selected_option_ids) > 1:
            response_type = "multi_choice"
        elif len(selected_option_ids) == 1:
            response_type = "single_choice"
        elif custom_response:
            response_type = "text"
        else:
            response_type = str(response_type_hint or "text")

    # Validate selected IDs against declared options.
    invalid_ids = [item for item in selected_option_ids if item not in options_by_id]
    if invalid_ids:
        raise ValueError(f"Invalid option id(s): {', '.join(invalid_ids)}")

    if response_type in {"cancelled", "timeout"}:
        selected_labels = [
            options_by_id[item]
            for item in selected_option_ids
            if item in options_by_id
        ]
        answered_at = (
            str(payload.get("answered_at", "") or "").strip()
            or datetime.now(UTC).isoformat()
        )
        return {
            "question_id": question_id,
            "response_type": response_type,
            "selected_option_ids": selected_option_ids,
            "selected_labels": selected_labels,
            "custom_response": custom_response,
            "answered_at": answered_at,
            "source": source,
            "answered_by": answered_by,
            "client_id": client_id,
        }

    if question_type == "single_choice":
        if len(selected_option_ids) > 1:
            raise ValueError("single_choice questions allow exactly one selection")
        if not selected_option_ids and not (allow_custom and custom_response):
            raise ValueError("single_choice question requires one option or custom response")
    elif question_type == "multi_choice":
        min_sel = max(0, int(request_payload.get("min_selections", 0) or 0))
        max_sel = max(0, int(request_payload.get("max_selections", 0) or 0))
        if selected_option_ids:
            if len(selected_option_ids) < min_sel:
                raise ValueError(
                    f"At least {min_sel} selection(s) required",
                )
            if max_sel > 0 and len(selected_option_ids) > max_sel:
                raise ValueError(
                    f"At most {max_sel} selection(s) allowed",
                )
        elif not (allow_custom and custom_response):
            if min_sel > 0:
                raise ValueError(
                    f"At least {min_sel} selection(s) required",
                )
            raise ValueError("multi_choice question requires selection(s) or custom response")
    else:
        if not custom_response and not selected_option_ids:
            raise ValueError("free_text question requires a response")

    if custom_response and not allow_custom and not selected_option_ids:
        raise ValueError("custom response is not allowed for this question")

    selected_labels = [options_by_id[item] for item in selected_option_ids if item in options_by_id]
    answered_at = str(payload.get("answered_at", "") or "").strip() or datetime.now(UTC).isoformat()
    return {
        "question_id": question_id,
        "response_type": response_type,
        "selected_option_ids": selected_option_ids,
        "selected_labels": selected_labels,
        "custom_response": custom_response,
        "answered_at": answered_at,
        "source": source,
        "answered_by": answered_by,
        "client_id": client_id,
    }


def _build_timeout_default_payload(request: QuestionRequest) -> dict[str, Any] | None:
    options_by_id = {
        str(item.get("id", "")).strip(): str(item.get("label", "")).strip()
        for item in request.options
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    }
    answered_at = datetime.now(UTC).isoformat()
    source = "policy_default"

    if request.question_type == "single_choice":
        candidate = str(request.default_option_id or "").strip()
        if not candidate and isinstance(request.timeout_default_response, str):
            candidate = str(request.timeout_default_response).strip()
        if candidate not in options_by_id:
            return None
        return {
            "question_id": "",
            "response_type": "timeout",
            "selected_option_ids": [candidate],
            "selected_labels": [options_by_id[candidate]],
            "custom_response": "",
            "answered_at": answered_at,
            "source": source,
        }

    if request.question_type == "multi_choice":
        defaults = request.timeout_default_response
        selected: list[str] = []
        if isinstance(defaults, str):
            selected = [
                item.strip()
                for item in defaults.split(",")
                if item.strip()
            ]
        elif isinstance(defaults, list):
            selected = [
                str(item or "").strip()
                for item in defaults
                if str(item or "").strip()
            ]
        if not selected and request.default_option_id:
            selected = [request.default_option_id]
        if not selected or any(item not in options_by_id for item in selected):
            return None
        min_sel = max(0, int(request.min_selections or 0))
        max_sel = max(0, int(request.max_selections or 0))
        if len(selected) < min_sel:
            return None
        if max_sel > 0 and len(selected) > max_sel:
            return None
        return {
            "question_id": "",
            "response_type": "timeout",
            "selected_option_ids": selected,
            "selected_labels": [options_by_id[item] for item in selected],
            "custom_response": "",
            "answered_at": answered_at,
            "source": source,
        }

    text_default = ""
    if isinstance(request.timeout_default_response, str):
        text_default = request.timeout_default_response.strip()
    if not text_default:
        return None
    return {
        "question_id": "",
        "response_type": "timeout",
        "selected_option_ids": [],
        "selected_labels": [],
        "custom_response": text_default,
        "answered_at": answered_at,
        "source": source,
    }
