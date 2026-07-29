"""Typed contracts for the correction state machine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum


class Repairability(StrEnum):
    AUTOMATIC = "automatic"
    CONDITIONAL = "conditional"
    HUMAN_REQUIRED = "human_required"
    TERMINAL = "terminal"


class CorrectionState(StrEnum):
    DETECTED = "detected"
    CLASSIFIED = "classified"
    CHECKPOINTED = "checkpointed"
    PLANNED = "planned"
    PREFLIGHT = "preflight"
    EXECUTING = "executing"
    REVERIFYING = "reverifying"
    RETRYING = "retrying"
    REPLANNING = "replanning"
    RESOLVED = "resolved"
    HUMAN_REQUIRED = "human_required"
    TERMINAL = "terminal"


class CorrectionHandler(StrEnum):
    RETRY_EXECUTION = "retry_execution"
    SCHEMA_REPAIR = "schema_repair"
    RETRY_VERIFICATION = "retry_verification"
    CONTEXT_REFRESH = "context_refresh"
    CHECKPOINT_CONTINUE = "checkpoint_continue"
    SOURCE_FALLBACK = "source_fallback"
    CONFIRM_OR_PRUNE = "confirm_or_prune"
    PLACEHOLDER_PREPASS = "placeholder_prepass"
    REPLAN = "replan"
    HUMAN_REVIEW = "human_review"
    NONE = "none"


@dataclass(frozen=True)
class Blocker:
    """A normalized condition preventing the current unit from completing."""

    code: str
    message: str
    blocking: bool
    repairability: Repairability
    source: str = "verification"
    targets: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["repairability"] = self.repairability.value
        payload["targets"] = list(self.targets)
        return payload

    @property
    def fingerprint(self) -> str:
        payload = {
            "code": self.code,
            "source": self.source,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ProgressVector:
    """Structured signals used to decide whether correction is converging."""

    blocker_count: int
    failed_check_count: int
    missing_target_count: int
    contradicted_claim_count: int
    supported_claim_count: int
    deliverable_count: int
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> ProgressVector | None:
        if not isinstance(payload, dict):
            return None
        try:
            return cls(
                blocker_count=int(payload.get("blocker_count", 0) or 0),
                failed_check_count=int(payload.get("failed_check_count", 0) or 0),
                missing_target_count=int(payload.get("missing_target_count", 0) or 0),
                contradicted_claim_count=int(
                    payload.get("contradicted_claim_count", 0) or 0
                ),
                supported_claim_count=int(payload.get("supported_claim_count", 0) or 0),
                deliverable_count=int(payload.get("deliverable_count", 0) or 0),
                confidence=float(payload.get("confidence", 0.0) or 0.0),
            )
        except (TypeError, ValueError):
            return None

    def improved_from(self, previous: ProgressVector | None) -> bool:
        if previous is None:
            return True
        lower_is_better = (
            self.blocker_count,
            self.failed_check_count,
            self.missing_target_count,
            self.contradicted_claim_count,
        )
        previous_lower = (
            previous.blocker_count,
            previous.failed_check_count,
            previous.missing_target_count,
            previous.contradicted_claim_count,
        )
        if lower_is_better < previous_lower:
            return True
        higher_is_better = (
            self.supported_claim_count,
            self.deliverable_count,
            round(self.confidence, 4),
        )
        previous_higher = (
            previous.supported_claim_count,
            previous.deliverable_count,
            round(previous.confidence, 4),
        )
        return lower_is_better == previous_lower and higher_is_better > previous_higher


@dataclass(frozen=True)
class RepairAction:
    action_type: str
    handler: CorrectionHandler
    arguments: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "action_type": self.action_type,
            "handler": self.handler.value,
            "arguments": self.arguments,
        }


@dataclass(frozen=True)
class CorrectionDecision:
    cycle_id: str
    blockers: tuple[Blocker, ...]
    repairability: Repairability
    handler: CorrectionHandler
    state: CorrectionState
    actions: tuple[RepairAction, ...]
    progress: ProgressVector
    progress_made: bool
    no_progress_count: int
    stop_for_no_progress: bool

    @property
    def blocker_fingerprint(self) -> str:
        joined = "|".join(sorted(blocker.fingerprint for blocker in self.blockers))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:20]
