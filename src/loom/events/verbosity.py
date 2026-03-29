"""Runtime telemetry mode normalization and sink filtering policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from loom.events.types import (
    ACTIVE_EVENT_TYPES,
    APPROVAL_REJECTED,
    APPROVAL_REQUESTED,
    APPROVAL_TIMED_OUT,
    ASK_USER_REQUESTED,
    DEPRECATED_EVENT_TYPES,
    INTERNAL_ONLY_EVENT_TYPES,
    TASK_CANCEL_ACK,
    TASK_CANCEL_REQUESTED,
    TASK_CANCEL_TIMEOUT,
    TASK_CANCELLED,
    TASK_COMPLETED,
    TASK_CREATED,
    TASK_FAILED,
    TASK_INJECTED,
    TASK_PAUSED,
    TASK_RESTARTED,
    TASK_RESUMED,
    TASK_RUN_ACQUIRED,
    TASK_RUN_HEARTBEAT,
    TASK_RUN_RECOVERED,
    TELEMETRY_DIAGNOSTIC,
    TelemetryMode,
)

DEFAULT_TELEMETRY_MODE: TelemetryMode = "active"

_TELEMETRY_MODE_ALIASES: dict[str, TelemetryMode] = {
    "internal_only": "all_typed",
}

_KNOWN_TELEMETRY_MODES: frozenset[TelemetryMode] = frozenset(
    {"off", "active", "all_typed", "debug"},
)

OPERATOR_PASSTHROUGH_EVENT_TYPES = frozenset(
    {
        TASK_CREATED,
        TASK_RUN_ACQUIRED,
        TASK_RUN_HEARTBEAT,
        TASK_RUN_RECOVERED,
        TASK_CANCEL_REQUESTED,
        TASK_CANCEL_ACK,
        TASK_CANCEL_TIMEOUT,
        TASK_PAUSED,
        TASK_RESUMED,
        TASK_INJECTED,
        APPROVAL_REQUESTED,
        APPROVAL_REJECTED,
        APPROVAL_TIMED_OUT,
        ASK_USER_REQUESTED,
        TASK_COMPLETED,
        TASK_FAILED,
        TASK_CANCELLED,
        TASK_RESTARTED,
    },
)

DEBUG_DIAGNOSTIC_EVENT_TYPES = frozenset({TELEMETRY_DIAGNOSTIC})


@dataclass(frozen=True)
class TelemetryModeResolution:
    """Result of parsing a potentially non-canonical telemetry mode value."""

    mode: TelemetryMode
    input_value: str
    warning_code: str = ""

    @property
    def normalized(self) -> bool:
        return bool(self.warning_code)


def normalize_telemetry_mode(
    value: object,
    *,
    default: TelemetryMode = DEFAULT_TELEMETRY_MODE,
) -> TelemetryModeResolution:
    """Normalize telemetry mode with alias and invalid-input handling."""
    default_mode: TelemetryMode = (
        default if default in _KNOWN_TELEMETRY_MODES else DEFAULT_TELEMETRY_MODE
    )
    raw = str(value or "").strip().lower()
    if not raw:
        return TelemetryModeResolution(
            mode=default_mode,
            input_value=raw,
            warning_code="telemetry_mode_empty_defaulted",
        )
    if raw in _KNOWN_TELEMETRY_MODES:
        return TelemetryModeResolution(mode=cast(TelemetryMode, raw), input_value=raw)
    aliased = _TELEMETRY_MODE_ALIASES.get(raw)
    if aliased is not None:
        return TelemetryModeResolution(
            mode=aliased,
            input_value=raw,
            warning_code="telemetry_mode_alias_normalized",
        )
    return TelemetryModeResolution(
        mode=default_mode,
        input_value=raw,
        warning_code="telemetry_mode_invalid_defaulted",
    )


def should_deliver_operator(event_type: str, mode: TelemetryMode | str) -> bool:
    """Return whether operator-facing sinks should receive this event."""
    normalized_event = str(event_type or "").strip()
    if not normalized_event:
        return False

    resolved_mode = normalize_telemetry_mode(mode, default=DEFAULT_TELEMETRY_MODE).mode
    if normalized_event in OPERATOR_PASSTHROUGH_EVENT_TYPES:
        return True
    if normalized_event in DEBUG_DIAGNOSTIC_EVENT_TYPES:
        return resolved_mode == "debug"
    if resolved_mode == "off":
        return False
    if resolved_mode == "active":
        return normalized_event in ACTIVE_EVENT_TYPES
    if resolved_mode in {"all_typed", "debug"}:
        return (
            normalized_event in ACTIVE_EVENT_TYPES
            or normalized_event in INTERNAL_ONLY_EVENT_TYPES
        )
    return normalized_event in ACTIVE_EVENT_TYPES


def should_persist_compliance(event_type: str) -> bool:
    """Return whether compliance sinks should persist this event."""
    normalized_event = str(event_type or "").strip()
    if not normalized_event:
        return False
    if (
        normalized_event in ACTIVE_EVENT_TYPES
        or normalized_event in INTERNAL_ONLY_EVENT_TYPES
        or normalized_event in DEPRECATED_EVENT_TYPES
        or normalized_event in OPERATOR_PASSTHROUGH_EVENT_TYPES
    ):
        return True
    # Unknown events are retained for forensic troubleshooting.
    return True
