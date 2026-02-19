"""Approval manager for human-in-the-loop gating.

Determines when to proceed automatically, when to pause for review,
and when to hard-stop for approval based on confidence scores.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum

from loom.events.bus import Event, EventBus
from loom.events.types import APPROVAL_RECEIVED, APPROVAL_REQUESTED


class ApprovalDecision(StrEnum):
    """What the approval manager decides."""

    PROCEED = "proceed"
    WAIT = "wait"
    WAIT_WITH_TIMEOUT = "wait_with_timeout"
    ABORT = "abort"


@dataclass
class ApprovalRequest:
    """A request for human approval."""

    task_id: str
    subtask_id: str
    reason: str
    proposed_action: str
    risk_level: str  # "low", "medium", "high", "critical"
    details: dict = field(default_factory=dict)
    auto_approve_timeout: int | None = None  # seconds, None = no auto


# Operations that always require approval regardless of confidence
ALWAYS_GATE_PATTERNS = [
    "rm -rf",
    "drop table",
    "drop database",
    "truncate table",
    "delete from",
    "chmod 777",
    "sudo ",
]

ALWAYS_GATE_FILES = [
    ".env",
    ".env.local",
    ".env.production",
    "credentials",
    "secrets",
    ".ssh/",
    ".gnupg/",
]


class ApprovalManager:
    """Manages approval gates for subtask execution.

    Modes:
    - auto: proceed at high confidence, gate destructive/low confidence
    - manual: gate every subtask
    - confidence_threshold: auto-proceed above configured threshold
    """

    def __init__(self, event_bus: EventBus, default_timeout: int = 10):
        self._event_bus = event_bus
        self._default_timeout = default_timeout
        self._pending_approvals: dict[str, asyncio.Event] = {}
        self._approval_results: dict[str, bool] = {}

    def check_approval(
        self,
        approval_mode: str,
        confidence: float,
        result: object,
        confidence_threshold: float = 0.8,
    ) -> ApprovalDecision:
        """Determine whether to proceed, wait, or abort.

        Args:
            approval_mode: "auto", "manual", or "confidence_threshold"
            confidence: computed confidence score (0.0-1.0)
            result: SubtaskResult with tool_calls
            confidence_threshold: threshold for confidence_threshold mode
        """
        if approval_mode == "manual":
            return ApprovalDecision.WAIT

        if approval_mode in {"disabled", "none", "off"}:
            if self._is_always_gated(result):
                return ApprovalDecision.WAIT
            return ApprovalDecision.PROCEED

        if approval_mode == "auto":
            if self._is_always_gated(result):
                return ApprovalDecision.WAIT
            if confidence >= 0.8:
                return ApprovalDecision.PROCEED
            if confidence >= 0.5:
                return ApprovalDecision.WAIT_WITH_TIMEOUT
            if confidence >= 0.2:
                return ApprovalDecision.WAIT
            return ApprovalDecision.ABORT

        if approval_mode == "confidence_threshold":
            if confidence >= confidence_threshold and not self._is_always_gated(result):
                return ApprovalDecision.PROCEED
            return ApprovalDecision.WAIT

        # Unknown mode — default to auto behavior
        return ApprovalDecision.PROCEED if confidence >= 0.8 else ApprovalDecision.WAIT

    async def request_approval(
        self,
        request: ApprovalRequest,
    ) -> bool:
        """Emit approval request and wait for response.

        Returns True if approved, False if rejected or timed out.
        """
        key = f"{request.task_id}:{request.subtask_id}"
        event = asyncio.Event()
        self._pending_approvals[key] = event

        # Emit event for API/TUI
        self._event_bus.emit(Event(
            event_type=APPROVAL_REQUESTED,
            task_id=request.task_id,
            data={
                "subtask_id": request.subtask_id,
                "reason": request.reason,
                "proposed_action": request.proposed_action,
                "risk_level": request.risk_level,
                "details": request.details,
                "auto_approve_timeout": request.auto_approve_timeout,
            },
        ))

        timeout = request.auto_approve_timeout
        try:
            if timeout is not None:
                try:
                    await asyncio.wait_for(event.wait(), timeout=timeout)
                except TimeoutError:
                    # Deny on timeout — never auto-approve unattended
                    self._approval_results[key] = False
            else:
                await event.wait()

            return self._approval_results.get(key, False)
        finally:
            self._pending_approvals.pop(key, None)
            self._approval_results.pop(key, None)

    def resolve_approval(self, task_id: str, subtask_id: str, approved: bool) -> bool:
        """Called by API endpoint when human responds.

        Returns True if there was a pending approval to resolve.
        """
        key = f"{task_id}:{subtask_id}"
        event = self._pending_approvals.get(key)
        if event is None:
            return False

        self._approval_results[key] = approved
        event.set()

        self._event_bus.emit(Event(
            event_type=APPROVAL_RECEIVED,
            task_id=task_id,
            data={
                "subtask_id": subtask_id,
                "approved": approved,
            },
        ))
        return True

    @staticmethod
    def _is_always_gated(result: object) -> bool:
        """Check if the result contains operations that always need approval."""
        tool_calls = getattr(result, "tool_calls", [])
        for tc in tool_calls:
            # Check shell commands for destructive patterns
            if tc.tool == "shell_execute":
                cmd = tc.args.get("command", "").lower()
                for pattern in ALWAYS_GATE_PATTERNS:
                    if pattern in cmd:
                        return True

            # Check file operations on sensitive paths
            if tc.tool in ("write_file", "edit_file"):
                path = tc.args.get("path", "").lower()
                for sensitive in ALWAYS_GATE_FILES:
                    if sensitive in path:
                        return True

        return False
