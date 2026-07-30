"""Durable correction controller coordinating existing recovery mechanisms."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from loom.engine.correction.policy import build_actions, classify_blockers, select_handler
from loom.engine.correction.types import (
    CorrectionDecision,
    CorrectionState,
    ProgressVector,
    Repairability,
)
from loom.events.types import (
    CORRECTION_DETECTED,
    CORRECTION_PLANNED,
    CORRECTION_PROGRESS,
    CORRECTION_RESOLVED,
    CORRECTION_TERMINAL,
)


class CorrectionController:
    """Classify, persist, route, and observe correction cycles."""

    def __init__(
        self,
        memory,
        emit: Callable[[str, str, dict], None],
        *,
        max_no_progress_attempts: int = 2,
        max_total_attempts_per_subtask: int = 6,
    ) -> None:
        self._memory = memory
        self._emit = emit
        self._max_no_progress_attempts = max(1, int(max_no_progress_attempts))
        self._max_total_attempts_per_subtask = max(
            1,
            int(max_total_attempts_per_subtask),
        )

    async def record_failure(
        self,
        *,
        task_id: str,
        run_id: str,
        subtask_id: str,
        result,
        verification,
    ) -> CorrectionDecision:
        blockers = classify_blockers(verification)
        handler = select_handler(blockers)
        actions = build_actions(handler, blockers)
        repairability = max(
            (blocker.repairability for blocker in blockers),
            key=lambda item: {
                Repairability.AUTOMATIC: 0,
                Repairability.CONDITIONAL: 1,
                Repairability.HUMAN_REQUIRED: 2,
                Repairability.TERMINAL: 3,
            }[item],
        )
        joined = "|".join(sorted(blocker.fingerprint for blocker in blockers))
        blocker_fingerprint = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:20]
        cycle_seed = f"{task_id}|{subtask_id}|{blocker_fingerprint}"
        cycle_id = "corr-" + hashlib.sha256(cycle_seed.encode("utf-8")).hexdigest()[:16]
        progress = self._progress_vector(
            result=result,
            verification=verification,
            blockers=blockers,
        )

        previous = await self._memory.get_correction_cycle(cycle_id=cycle_id)
        if not isinstance(previous, dict):
            previous = None
        existing_cycles = await self._memory.list_correction_cycles(
            task_id=task_id,
            subtask_id=subtask_id,
        )
        recent_active_cycle = next(
            (
                item
                for item in reversed(existing_cycles)
                if isinstance(item, dict)
                and str(item.get("id", "") or "") != cycle_id
                and str(item.get("state", "") or "") not in {"resolved"}
            ),
            None,
        )
        # Changing repair handlers as a gap narrows is not itself progress.
        comparison_cycle = previous or recent_active_cycle
        previous_progress = ProgressVector.from_dict(
            comparison_cycle.get("latest_progress") if comparison_cycle else None
        )
        progress_made = progress.improved_from(previous_progress)
        previous_stalled = int(
            (comparison_cycle or {}).get("no_progress_count", 0) or 0
        )
        no_progress_count = 0 if progress_made else previous_stalled + 1
        stop_for_no_progress = no_progress_count >= self._max_no_progress_attempts
        total_prior_attempts = sum(
            int(item.get("attempt_count", 0) or 0)
            for item in existing_cycles
            if isinstance(item, dict)
        )
        total_attempt_count = total_prior_attempts + 1
        stop_for_attempt_budget = (
            total_attempt_count >= self._max_total_attempts_per_subtask
        )
        state = (
            CorrectionState.TERMINAL
            if (
                repairability == Repairability.TERMINAL
                or stop_for_no_progress
                or stop_for_attempt_budget
            )
            else (
                CorrectionState.HUMAN_REQUIRED
                if repairability == Repairability.HUMAN_REQUIRED
                else CorrectionState.PLANNED
            )
        )
        attempt_count = int((previous or {}).get("attempt_count", 0) or 0) + 1
        decision = CorrectionDecision(
            cycle_id=cycle_id,
            blockers=blockers,
            repairability=repairability,
            handler=handler,
            state=state,
            actions=actions,
            progress=progress,
            progress_made=progress_made,
            no_progress_count=no_progress_count,
            stop_for_no_progress=stop_for_no_progress,
            total_attempt_count=total_attempt_count,
            stop_for_attempt_budget=stop_for_attempt_budget,
        )

        await self._memory.upsert_correction_cycle(
            cycle_id=cycle_id,
            task_id=task_id,
            run_id=run_id,
            subtask_id=subtask_id,
            blocker_fingerprint=blocker_fingerprint,
            state=state.value,
            blocking=any(blocker.blocking for blocker in blockers),
            repairability=repairability.value,
            handler=handler.value,
            reason_code=blockers[0].code,
            blocker_snapshot=[blocker.to_dict() for blocker in blockers],
            baseline_progress=(
                previous.get("baseline_progress")
                if previous and previous.get("baseline_progress")
                else progress.to_dict()
            ),
            latest_progress=progress.to_dict(),
            attempt_count=attempt_count,
            no_progress_count=no_progress_count,
            max_attempts=self._max_no_progress_attempts + 1,
            terminal_reason=(
                "no_progress"
                if stop_for_no_progress
                else (
                    "repair_attempt_budget_exhausted"
                    if stop_for_attempt_budget
                    else (
                        "non_repairable_blocker"
                        if state == CorrectionState.TERMINAL
                        else ""
                    )
                )
            ),
        )
        attempt_id = await self._memory.insert_correction_attempt(
            correction_id=cycle_id,
            task_id=task_id,
            run_id=run_id,
            subtask_id=subtask_id,
            attempt=attempt_count,
            state=state.value,
            plan={"actions": [action.to_dict() for action in actions]},
            before_progress=previous_progress.to_dict() if previous_progress else {},
            after_progress=progress.to_dict(),
            progress_made=progress_made,
            outcome="terminal" if state == CorrectionState.TERMINAL else "planned",
            metadata={
                "no_progress_count": no_progress_count,
                "total_attempt_count": total_attempt_count,
                "max_total_attempts_per_subtask": self._max_total_attempts_per_subtask,
            },
        )
        for sequence, action in enumerate(actions, start=1):
            await self._memory.insert_correction_action(
                correction_attempt_id=attempt_id,
                correction_id=cycle_id,
                sequence=sequence,
                action_type=action.action_type,
                handler=action.handler.value,
                arguments=action.arguments,
                idempotency_key=f"{cycle_id}:{attempt_count}:{sequence}",
                state="planned",
            )

        event_data = self._event_data(decision, subtask_id=subtask_id)
        self._emit(CORRECTION_DETECTED, task_id, event_data)
        self._emit(
            CORRECTION_TERMINAL if state == CorrectionState.TERMINAL else CORRECTION_PLANNED,
            task_id,
            event_data,
        )
        if previous is not None:
            self._emit(CORRECTION_PROGRESS, task_id, event_data)
        return decision

    async def mark_routed(
        self,
        *,
        decision: CorrectionDecision,
        state: CorrectionState,
        outcome: str,
    ) -> None:
        await self._memory.update_correction_cycle_state(
            cycle_id=decision.cycle_id,
            state=state.value,
            terminal_reason=(
                "no_progress"
                if decision.stop_for_no_progress
                else (
                    "repair_attempt_budget_exhausted"
                    if decision.stop_for_attempt_budget
                    else ""
                )
            ),
        )

    async def resolve_subtask(self, *, task_id: str, subtask_id: str) -> None:
        resolved = await self._memory.resolve_correction_cycles(
            task_id=task_id,
            subtask_id=subtask_id,
        )
        for cycle_id in resolved:
            self._emit(
                CORRECTION_RESOLVED,
                task_id,
                {"cycle_id": cycle_id, "subtask_id": subtask_id, "state": "resolved"},
            )

    @staticmethod
    def _progress_vector(*, result, verification, blockers) -> ProgressVector:
        metadata = (
            dict(verification.metadata)
            if isinstance(getattr(verification, "metadata", None), dict)
            else {}
        )
        failed_checks = sum(
            1 for check in (getattr(verification, "checks", None) or [])
            if not bool(getattr(check, "passed", False))
        )
        missing_targets = {
            target for blocker in blockers for target in blocker.targets
        }
        tool_calls = getattr(result, "tool_calls", None) or []
        deliverables = sum(
            1 for call in tool_calls
            if bool(getattr(getattr(call, "result", None), "success", False))
            and str(getattr(call, "tool", "") or "") in {
                "write_file", "edit_file", "create_document", "create_spreadsheet"
            }
        )
        return ProgressVector(
            blocker_count=len(blockers),
            failed_check_count=failed_checks,
            missing_target_count=len(missing_targets),
            contradicted_claim_count=int(metadata.get("contradicted_count", 0) or 0),
            supported_claim_count=int(
                metadata.get("supported_count", metadata.get("verified_claim_count", 0)) or 0
            ),
            deliverable_count=deliverables,
            confidence=float(getattr(verification, "confidence", 0.0) or 0.0),
        )

    @staticmethod
    def _event_data(decision: CorrectionDecision, *, subtask_id: str) -> dict:
        return {
            "cycle_id": decision.cycle_id,
            "subtask_id": subtask_id,
            "state": decision.state.value,
            "repairability": decision.repairability.value,
            "handler": decision.handler.value,
            "blockers": [blocker.to_dict() for blocker in decision.blockers],
            "progress": decision.progress.to_dict(),
            "progress_made": decision.progress_made,
            "no_progress_count": decision.no_progress_count,
            "stop_for_no_progress": decision.stop_for_no_progress,
            "total_attempt_count": decision.total_attempt_count,
            "stop_for_attempt_budget": decision.stop_for_attempt_budget,
        }
