"""Shared runner result and compaction plan types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from loom.tools.registry import ToolResult


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation during subtask execution."""

    tool: str
    args: dict
    result: ToolResult
    call_id: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class SubtaskResultStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class SubtaskResult:
    """Result of a subtask execution."""

    status: SubtaskResultStatus = SubtaskResultStatus.SUCCESS
    summary: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    duration_seconds: float = 0.0
    tokens_used: int = 0
    model_used: str = ""
    evidence_records: list[dict] = field(default_factory=list)
    telemetry_counters: dict[str, int] = field(default_factory=dict)


class CompactionClass(StrEnum):
    CRITICAL = "critical"
    TOOL_TRACE = "tool_trace"
    HISTORICAL_CONTEXT = "historical_context"
    BACKGROUND_EXTRACTION = "background_extraction"


class CompactionPressureTier(StrEnum):
    NORMAL = "normal"
    PRESSURE = "pressure"
    CRITICAL = "critical"


@dataclass(frozen=True)
class _CompactionPlan:
    critical_indices: tuple[int, ...]
    stage1_tool_args: tuple[int, ...]
    stage2_tool_output: tuple[int, ...]
    stage3_historical: tuple[int, ...]
    stage4_merge: tuple[int, ...]
