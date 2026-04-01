"""Output-coordination policy helpers for orchestrator."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from loom.engine.path_policy import normalize_deliverable_paths, normalize_path_for_policy
from loom.events.types import TASK_CANCELLED, TASK_COMPLETED, TASK_FAILED
from loom.processes.phase_alignment import infer_phase_id_for_subtask
from loom.recovery.retry import AttemptRecord, RetryStrategy
from loom.state.task_state import Subtask, SubtaskStatus, Task, TaskStatus

logger = logging.getLogger(__name__)


def output_intermediate_root(process) -> str:
    """Resolve intermediate artifact root path."""
    if process is None:
        return ".loom/phase-artifacts"
    coordination = getattr(process, "output_coordination", None)
    root = str(getattr(coordination, "intermediate_root", "") or "").strip()
    return root or ".loom/phase-artifacts"


def output_enforce_single_writer(process) -> bool:
    """Resolve single-writer enforcement mode."""
    if process is None:
        return True
    coordination = getattr(process, "output_coordination", None)
    return bool(getattr(coordination, "enforce_single_writer", True))


def output_conflict_policy(process) -> str:
    """Resolve conflict policy with bounded allowed values."""
    if process is None:
        return "defer_fifo"
    coordination = getattr(process, "output_coordination", None)
    policy = str(getattr(coordination, "conflict_policy", "") or "").strip().lower()
    if policy in {"defer_fifo", "fail_fast"}:
        return policy
    return "defer_fifo"


def output_publish_mode(process) -> str:
    """Resolve publish mode with bounded allowed values."""
    if process is None:
        return "transactional"
    coordination = getattr(process, "output_coordination", None)
    mode = str(getattr(coordination, "publish_mode", "") or "").strip().lower()
    if mode in {"transactional", "best_effort"}:
        return mode
    return "transactional"


def phase_finalizer_input_policy(process, phase_id: str) -> str:
    """Resolve per-phase finalizer input policy."""
    if process is None:
        return "require_all_workers"
    resolver = getattr(process, "phase_finalizer_input_policy", None)
    if callable(resolver):
        try:
            resolved = str(resolver(phase_id)).strip().lower()
        except Exception:
            resolved = ""
        if resolved in {"require_all_workers", "allow_partial"}:
            return resolved
    return "require_all_workers"


def normalize_deliverable_paths_for_conflict(
    raw_paths: list[str],
    *,
    workspace: Path | None,
) -> list[str]:
    """Normalize deliverable paths for conflict detection."""
    return normalize_deliverable_paths(
        raw_paths,
        workspace=workspace,
    )


def canonical_deliverable_paths_for_subtask(
    *,
    task,
    subtask,
    output_write_policy_for_subtask: Callable[[object], dict[str, object]],
) -> list[str]:
    """Resolve canonical deliverable paths for a subtask in workspace context."""
    workspace = Path(task.workspace) if str(task.workspace or "").strip() else None
    policy = output_write_policy_for_subtask(subtask)
    expected = [
        str(item).strip()
        for item in list(policy.get("expected_deliverables", []))
        if str(item).strip()
    ]
    return normalize_deliverable_paths_for_conflict(
        expected,
        workspace=workspace,
    )


def prioritize_runnable_for_output_conflicts(
    *,
    runnable: list,
    conflict_tracker: dict[str, dict[str, object]],
    starvation_threshold: int,
) -> list:
    """Prioritize runnable subtasks by conflict-starvation state."""
    threshold = max(1, int(starvation_threshold or 1))
    indexed = list(enumerate(runnable))

    def _sort_key(item: tuple[int, object]) -> tuple[int, int, int]:
        index, subtask = item
        subtask_id = str(getattr(subtask, "id", "") or "")
        state = conflict_tracker.get(subtask_id, {})
        streak = int(state.get("streak", 0) or 0)
        first_seq = int(state.get("first_seq", 10**9) or 10**9)
        if streak >= threshold:
            bucket = 0
        elif streak > 0:
            bucket = 1
        else:
            bucket = 2
            first_seq = 10**9 + index
        return bucket, first_seq, index

    indexed.sort(key=_sort_key)
    return [subtask for _, subtask in indexed]


def select_conflict_safe_batch(
    *,
    runnable: list,
    max_parallel: int,
    conflict_tracker: dict[str, dict[str, object]],
    sequence_counter: int,
    canonical_paths_for_subtask: Callable[[object], list[str]],
    starvation_threshold: int,
    active_pending_ids: set[str],
) -> tuple[list, list[dict[str, object]], int]:
    """Select conflict-safe runnable batch and update deferral tracker."""
    if not runnable or max_parallel <= 0:
        return [], [], sequence_counter

    prioritized = prioritize_runnable_for_output_conflicts(
        runnable=runnable,
        conflict_tracker=conflict_tracker,
        starvation_threshold=starvation_threshold,
    )
    deliverables_by_subtask: dict[str, list[str]] = {}
    for subtask in prioritized:
        subtask_id = str(getattr(subtask, "id", "") or "")
        deliverables_by_subtask[subtask_id] = canonical_paths_for_subtask(subtask)

    selected: list = []
    selected_paths: set[str] = set()
    owner_by_path: dict[str, str] = {}
    deferred: list[dict[str, object]] = []

    for subtask in prioritized:
        if len(selected) >= max_parallel:
            break
        subtask_id = str(getattr(subtask, "id", "") or "")
        paths = set(deliverables_by_subtask.get(subtask_id, []))
        overlap = sorted(path for path in paths if path in selected_paths)
        if overlap:
            conflicting_with = sorted({
                str(owner_by_path.get(path, "")).strip()
                for path in overlap
                if str(owner_by_path.get(path, "")).strip()
            })
            deferred.append({
                "subtask_id": subtask_id,
                "phase_id": str(getattr(subtask, "phase_id", "") or "").strip(),
                "conflicting_paths": overlap,
                "conflicting_with": conflicting_with,
            })
            continue
        selected.append(subtask)
        for path in paths:
            selected_paths.add(path)
            owner_by_path[path] = subtask_id

    for subtask_id in list(conflict_tracker.keys()):
        if subtask_id not in active_pending_ids:
            conflict_tracker.pop(subtask_id, None)

    selected_ids = {str(getattr(subtask, "id", "") or "") for subtask in selected}
    for subtask_id in selected_ids:
        conflict_tracker.pop(subtask_id, None)

    threshold = max(1, int(starvation_threshold or 1))
    for item in deferred:
        subtask_id = str(item.get("subtask_id", "") or "").strip()
        if not subtask_id:
            continue
        state = conflict_tracker.get(subtask_id, {})
        if "first_seq" not in state:
            state["first_seq"] = int(sequence_counter)
            sequence_counter += 1
        streak = int(state.get("streak", 0) or 0) + 1
        total = int(state.get("total", 0) or 0) + 1
        warned = bool(state.get("warned", False))
        starvation_warning = bool(streak >= threshold and not warned)
        state["streak"] = streak
        state["total"] = total
        state["warned"] = bool(warned or starvation_warning)
        conflict_tracker[subtask_id] = state
        item["deferral_streak"] = streak
        item["deferral_count"] = total
        item["starvation_warning"] = starvation_warning
        item["starvation_threshold"] = threshold

    return selected, deferred, sequence_counter


def fan_in_worker_output_prefixes(orchestrator, *, task, subtask) -> list[str]:
    """Return allowed worker intermediate prefixes for fan-in output mode."""
    output_policy = orchestrator._output_write_policy_for_subtask(subtask=subtask)
    output_strategy = str(output_policy.get("output_strategy", "") or "").strip().lower()
    output_role = str(output_policy.get("output_role", "") or "").strip().lower()
    if output_strategy != "fan_in" or output_role != orchestrator._OUTPUT_ROLE_WORKER:
        return []
    phase_id = str(getattr(subtask, "phase_id", "") or "").strip() or subtask.id
    intermediate_root = orchestrator._output_intermediate_root()
    run_id = orchestrator._task_run_id(task)
    candidates = [
        f"{intermediate_root}/{phase_id}/{subtask.id}",
    ]
    if run_id:
        candidates.insert(
            0,
            f"{intermediate_root}/{run_id}/{phase_id}/{subtask.id}",
        )
    normalized = orchestrator._normalize_deliverable_paths_for_conflict(
        [str(item).strip() for item in candidates if str(item).strip()],
        workspace=Path(task.workspace) if task.workspace else None,
    )
    return normalized


def phase_artifact_manifest_path(orchestrator, *, task, phase_id: str) -> Path | None:
    """Resolve manifest path for phase fan-in artifacts."""
    workspace = Path(task.workspace) if task.workspace else None
    normalized_phase_id = str(phase_id or "").strip()
    if workspace is None or not normalized_phase_id:
        return None
    run_id = orchestrator._task_run_id(task)
    if not run_id:
        return None
    return (
        workspace
        / orchestrator._output_intermediate_root()
        / run_id
        / normalized_phase_id
        / "manifest.jsonl"
    )


def phase_worker_subtask_ids(
    orchestrator,
    *,
    task,
    phase_id: str,
    finalizer_id: str,
) -> list[str]:
    """Collect worker subtask ids for a phase excluding the finalizer."""
    normalized_phase_id = str(phase_id or "").strip()
    if not normalized_phase_id:
        return []
    worker_ids: list[str] = []
    for candidate in task.plan.subtasks:
        if str(getattr(candidate, "phase_id", "") or "").strip() != normalized_phase_id:
            continue
        candidate_id = str(getattr(candidate, "id", "") or "").strip()
        if not candidate_id or candidate_id == finalizer_id:
            continue
        role = str(getattr(candidate, "output_role", "") or "").strip().lower()
        if role == orchestrator._OUTPUT_ROLE_PHASE_FINALIZER:
            continue
        if candidate_id not in worker_ids:
            worker_ids.append(candidate_id)
    return sorted(worker_ids)


def phase_worker_artifact_paths(orchestrator, *, task, phase_id: str) -> dict[str, list[str]]:
    """Collect latest artifact paths per worker for the given phase."""
    latest = orchestrator._latest_worker_artifacts_for_phase(task=task, phase_id=phase_id)
    artifact_paths: dict[str, list[str]] = {}
    for worker_id, entries in latest.items():
        if not isinstance(entries, list):
            continue
        paths = sorted({
            str(item.get("artifact_path", "")).strip()
            for item in entries
            if isinstance(item, dict) and str(item.get("artifact_path", "")).strip()
        })
        if paths:
            artifact_paths[str(worker_id).strip()] = paths
    return artifact_paths


def evaluate_finalizer_manifest_requirements(orchestrator, *, task, subtask) -> dict[str, object]:
    """Evaluate worker-manifest requirements for a finalizer subtask."""
    output_policy = orchestrator._output_write_policy_for_subtask(subtask=subtask)
    output_strategy = str(output_policy.get("output_strategy", "") or "").strip().lower()
    output_role = str(output_policy.get("output_role", "") or "").strip().lower()
    phase_id = str(getattr(subtask, "phase_id", "") or "").strip()
    if (
        output_strategy != "fan_in"
        or output_role != orchestrator._OUTPUT_ROLE_PHASE_FINALIZER
    ):
        return {
            "enabled": False,
            "policy": "",
            "worker_ids": [],
            "artifact_paths_by_worker": {},
            "missing_worker_ids": [],
            "allowed_manifest_paths": [],
        }

    finalizer_id = str(getattr(subtask, "id", "") or "").strip()
    worker_ids = orchestrator._phase_worker_subtask_ids(
        task=task,
        phase_id=phase_id,
        finalizer_id=finalizer_id,
    )
    artifact_paths_by_worker = orchestrator._phase_worker_artifact_paths(
        task=task,
        phase_id=phase_id,
    )
    missing_worker_ids = [
        worker_id
        for worker_id in worker_ids
        if worker_id not in artifact_paths_by_worker
    ]
    allowed_manifest_paths = sorted({
        path
        for paths in artifact_paths_by_worker.values()
        for path in paths
        if str(path).strip()
    })
    return {
        "enabled": True,
        "policy": orchestrator._phase_finalizer_input_policy(phase_id),
        "worker_ids": worker_ids,
        "artifact_paths_by_worker": artifact_paths_by_worker,
        "missing_worker_ids": missing_worker_ids,
        "allowed_manifest_paths": allowed_manifest_paths,
    }


def record_fan_in_worker_artifacts(orchestrator, *, task, subtask, result) -> None:
    """Append worker output artifact entries to phase manifest."""
    output_policy = orchestrator._output_write_policy_for_subtask(subtask=subtask)
    output_strategy = str(output_policy.get("output_strategy", "") or "").strip().lower()
    output_role = str(output_policy.get("output_role", "") or "").strip().lower()
    if output_strategy != "fan_in" or output_role != orchestrator._OUTPUT_ROLE_WORKER:
        return

    phase_id = str(getattr(subtask, "phase_id", "") or "").strip()
    manifest_path = orchestrator._phase_artifact_manifest_path(task=task, phase_id=phase_id)
    workspace = Path(task.workspace) if task.workspace else None
    if manifest_path is None or workspace is None:
        return

    allowed_prefixes = orchestrator._fan_in_worker_output_prefixes(
        task=task,
        subtask=subtask,
    )
    if not allowed_prefixes:
        return

    changed_files = orchestrator._normalize_deliverable_paths_for_conflict(
        orchestrator._files_from_tool_calls(result.tool_calls),
        workspace=workspace,
    )
    artifact_paths = [
        path
        for path in changed_files
        if any(path == prefix or path.startswith(prefix + "/") for prefix in allowed_prefixes)
    ]
    if not artifact_paths:
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().isoformat()
    attempt = max(1, int(getattr(subtask, "retry_count", 0) or 0) + 1)
    entries: list[dict[str, object]] = []
    for artifact_path in artifact_paths:
        digest = ""
        try:
            payload = (workspace / artifact_path).read_bytes()
            digest = hashlib.sha1(payload).hexdigest()
        except Exception:
            digest = ""
        entries.append({
            "schema_version": 1,
            "task_id": task.id,
            "run_id": orchestrator._task_run_id(task),
            "phase_id": phase_id,
            "subtask_id": subtask.id,
            "attempt": attempt,
            "generated_at": generated_at,
            "output_role": orchestrator._OUTPUT_ROLE_WORKER,
            "output_strategy": "fan_in",
            "artifact_path": artifact_path,
            "content_hash": digest,
        })
    with manifest_path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False))
            handle.write("\n")


def latest_worker_artifacts_for_phase(orchestrator, *, task, phase_id: str) -> dict[str, list]:
    """Load latest successful worker artifacts from phase manifest."""
    manifest_path = orchestrator._phase_artifact_manifest_path(task=task, phase_id=phase_id)
    if manifest_path is None or not manifest_path.exists():
        return {}
    run_id = orchestrator._task_run_id(task)
    latest: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[dict[str, object]]] = {}
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = str(line or "").strip()
                if not text:
                    continue
                try:
                    entry = json.loads(text)
                except Exception:
                    continue
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("run_id", "")).strip() != run_id:
                    continue
                if str(entry.get("task_id", "")).strip() != task.id:
                    continue
                if str(entry.get("phase_id", "")).strip() != str(phase_id).strip():
                    continue
                role = str(entry.get("output_role", "")).strip().lower()
                if role != orchestrator._OUTPUT_ROLE_WORKER:
                    continue
                worker_id = str(entry.get("subtask_id", "")).strip()
                if not worker_id:
                    continue
                try:
                    attempt = int(entry.get("attempt", 0) or 0)
                except (TypeError, ValueError):
                    attempt = 0
                generated_at = str(entry.get("generated_at", "") or "").strip()
                marker = latest.get(worker_id)
                if not isinstance(marker, dict):
                    latest[worker_id] = {
                        "attempt": attempt,
                        "generated_at": generated_at,
                    }
                    grouped[worker_id] = [entry]
                    continue
                marker_attempt = int(marker.get("attempt", 0) or 0)
                marker_generated = str(marker.get("generated_at", "") or "")
                if (attempt, generated_at) > (marker_attempt, marker_generated):
                    latest[worker_id] = {
                        "attempt": attempt,
                        "generated_at": generated_at,
                    }
                    grouped[worker_id] = [entry]
                elif (attempt, generated_at) == (marker_attempt, marker_generated):
                    grouped.setdefault(worker_id, []).append(entry)
    except Exception:
        return {}
    return grouped


def augment_retry_context_with_phase_artifacts(
    orchestrator,
    *,
    task,
    subtask,
    base_context: str,
) -> str:
    """Append worker-manifest summary context for fan-in finalizers."""
    output_policy = orchestrator._output_write_policy_for_subtask(subtask=subtask)
    output_strategy = str(output_policy.get("output_strategy", "") or "").strip().lower()
    output_role = str(output_policy.get("output_role", "") or "").strip().lower()
    if output_strategy != "fan_in" or output_role != orchestrator._OUTPUT_ROLE_PHASE_FINALIZER:
        return base_context

    phase_id = str(getattr(subtask, "phase_id", "") or "").strip()
    latest = orchestrator._latest_worker_artifacts_for_phase(task=task, phase_id=phase_id)
    if not latest:
        return (
            f"{base_context}\n\n"
            "FAN-IN WORKER ARTIFACT MANIFEST: no worker artifacts were discovered "
            "for this phase in the current run. Verify worker output paths first."
        ).strip()

    lines = ["FAN-IN WORKER ARTIFACT MANIFEST (LATEST SUCCESSFUL BY WORKER):"]
    for worker_id in sorted(latest):
        entries = latest.get(worker_id, [])
        artifact_paths = sorted({
            str(item.get("artifact_path", "")).strip()
            for item in entries
            if str(item.get("artifact_path", "")).strip()
        })
        if artifact_paths:
            lines.append(f"- {worker_id}: {', '.join(artifact_paths)}")
        else:
            lines.append(f"- {worker_id}: (manifest entry without artifact_path)")
    section = "\n".join(lines)
    return f"{base_context}\n\n{section}".strip()


def finalizer_stage_publish_plan(
    orchestrator,
    *,
    task,
    subtask,
    canonical_deliverables: list[str],
    attempt_index: int,
) -> dict[str, object]:
    """Plan transactional stage->publish mapping for fan-in finalizers."""
    output_policy = orchestrator._output_write_policy_for_subtask(subtask=subtask)
    output_strategy = str(output_policy.get("output_strategy", "") or "").strip().lower()
    output_role = str(output_policy.get("output_role", "") or "").strip().lower()
    if output_strategy != "fan_in" or output_role != orchestrator._OUTPUT_ROLE_PHASE_FINALIZER:
        return {"enabled": False}
    if orchestrator._output_publish_mode() != "transactional":
        return {"enabled": False}

    workspace = Path(task.workspace) if task.workspace else None
    if workspace is None:
        return {"enabled": False}

    phase_id = str(getattr(subtask, "phase_id", "") or "").strip() or subtask.id
    run_id = orchestrator._task_run_id(task)
    if not run_id:
        return {"enabled": False}
    canonical = orchestrator._normalize_deliverable_paths_for_conflict(
        canonical_deliverables,
        workspace=workspace,
    )
    if not canonical:
        return {"enabled": False}

    intermediate_root = orchestrator._output_intermediate_root()
    stage_root = (
        f"{intermediate_root}/{run_id}/{phase_id}/{subtask.id}/"
        f"publish-stage/attempt-{attempt_index}"
    )
    stage_to_canonical: dict[str, str] = {}
    stage_deliverables: list[str] = []
    for canonical_path in canonical:
        stage_path = normalize_path_for_policy(
            f"{stage_root}/{canonical_path}",
            workspace,
        )
        if not stage_path:
            continue
        stage_to_canonical[stage_path] = canonical_path
        stage_deliverables.append(stage_path)
    if not stage_deliverables:
        return {"enabled": False}
    stage_prefix = normalize_path_for_policy(stage_root, workspace)
    return {
        "enabled": True,
        "stage_deliverables": stage_deliverables,
        "stage_to_canonical": stage_to_canonical,
        "stage_prefixes": [stage_prefix] if stage_prefix else [],
        "canonical_deliverables": canonical,
    }


def merge_unique_paths(*groups: list[str]) -> list[str]:
    """Merge path groups preserving first appearance order."""
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for raw in list(group or []):
            value = str(raw or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            merged.append(value)
    return merged


def augment_retry_context_for_stage_publish(
    *,
    base_context: str,
    stage_plan: dict[str, object],
) -> str:
    """Append stage-publish instructions to retry context."""
    if not bool(stage_plan.get("enabled", False)):
        return base_context
    stage_deliverables = list(stage_plan.get("stage_deliverables", []))
    canonical_deliverables = list(stage_plan.get("canonical_deliverables", []))
    lines = [
        "TRANSACTIONAL FINALIZER PUBLISH MODE:",
        "- Write output files to STAGING paths only in this attempt.",
        "- Do not write canonical deliverable paths directly.",
    ]
    if stage_deliverables:
        lines.append("REQUIRED STAGING OUTPUT FILES (EXACT FILENAMES):")
        for name in stage_deliverables:
            lines.append(f"- {name}")
    if canonical_deliverables:
        lines.append("CANONICAL DELIVERABLES (PUBLISH TARGETS):")
        for name in canonical_deliverables:
            lines.append(f"- {name}")
    section = "\n".join(lines)
    return f"{base_context}\n\n{section}".strip()


def intermediate_phase_prefix(orchestrator, *, task, phase_id: str) -> str:
    """Resolve normalized intermediate phase root prefix."""
    workspace = Path(task.workspace) if task.workspace else None
    run_id = orchestrator._task_run_id(task)
    normalized_phase_id = str(phase_id or "").strip()
    if workspace is None or not run_id or not normalized_phase_id:
        return ""
    return normalize_path_for_policy(
        f"{orchestrator._output_intermediate_root()}/{run_id}/{normalized_phase_id}",
        workspace,
    )


def intermediate_read_paths_from_tool_calls(orchestrator, *, task, tool_calls: list) -> list[str]:
    """Extract normalized intermediate read paths from tool calls."""
    workspace = Path(task.workspace) if task.workspace else None
    if workspace is None:
        return []
    read_paths: list[str] = []
    seen: set[str] = set()
    for call in list(tool_calls or []):
        tool_name = str(getattr(call, "tool", "") or "").strip().lower()
        args = getattr(call, "args", {})
        if not isinstance(args, dict):
            continue
        candidates: list[str] = []
        if tool_name in {"read_file", "document_read"}:
            raw = args.get("path") or args.get("file") or args.get("file_path")
            if raw is not None:
                candidates.append(str(raw))
        elif tool_name == "read_multiple_files":
            raw_paths = args.get("paths", [])
            if isinstance(raw_paths, list):
                candidates.extend(str(item) for item in raw_paths)
        for raw in candidates:
            normalized = normalize_path_for_policy(raw, workspace)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            read_paths.append(normalized)
    return read_paths


def manifest_only_input_violations(
    orchestrator,
    *,
    task,
    subtask,
    tool_calls: list,
    allowed_manifest_paths: list[str],
    allowed_extra_prefixes: list[str],
) -> list[str]:
    """Detect phase-intermediate reads outside allowed manifest-derived paths."""
    phase_id = str(getattr(subtask, "phase_id", "") or "").strip()
    phase_prefix = orchestrator._intermediate_phase_prefix(task=task, phase_id=phase_id)
    if not phase_prefix:
        return []
    allowed_paths = {
        str(item).strip()
        for item in allowed_manifest_paths
        if str(item).strip()
    }
    extra_prefixes = [
        str(item).strip()
        for item in allowed_extra_prefixes
        if str(item).strip()
    ]
    manifest_path = orchestrator._phase_artifact_manifest_path(task=task, phase_id=phase_id)
    workspace = Path(task.workspace) if task.workspace else None
    manifest_rel = ""
    if manifest_path is not None and workspace is not None:
        manifest_rel = normalize_path_for_policy(str(manifest_path), workspace)
    if manifest_rel:
        allowed_paths.add(manifest_rel)

    violations: list[str] = []
    for path in orchestrator._intermediate_read_paths_from_tool_calls(
        task=task,
        tool_calls=tool_calls,
    ):
        if not (path == phase_prefix or path.startswith(phase_prefix + "/")):
            continue
        if path in allowed_paths:
            continue
        if any(path == prefix or path.startswith(prefix + "/") for prefix in extra_prefixes):
            continue
        violations.append(path)
    return sorted(set(violations))


def artifact_seals_snapshot(*, task) -> dict[str, dict[str, object]]:
    """Capture deep-copied artifact seal registry snapshot."""
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    registry = metadata.get("artifact_seals") if isinstance(metadata, dict) else {}
    if not isinstance(registry, dict):
        return {}
    return deepcopy(registry)


def restore_artifact_seals_snapshot(*, task, snapshot: dict[str, dict[str, object]]) -> None:
    """Restore artifact seal registry from snapshot."""
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["artifact_seals"] = deepcopy(snapshot) if isinstance(snapshot, dict) else {}
    task.metadata = metadata


def seal_paths_after_commit(orchestrator, *, task, subtask_id: str, paths: list[str]) -> None:
    """Seal committed canonical paths after transactional publish."""
    workspace = Path(task.workspace) if task.workspace else None
    if workspace is None:
        return
    seals = orchestrator._artifact_seal_registry(task)
    run_id = orchestrator._task_run_id(task)
    sealed_at = datetime.now().isoformat()
    for relpath in paths:
        text = str(relpath or "").strip()
        if not text:
            continue
        resolved = (workspace / text).resolve()
        try:
            resolved.relative_to(workspace.resolve())
        except Exception:
            continue
        if not resolved.exists() or not resolved.is_file():
            continue
        payload = resolved.read_bytes()
        seals[text] = {
            "path": text,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": int(len(payload)),
            "tool": "fan_in_commit",
            "tool_call_id": "",
            "subtask_id": subtask_id,
            "run_id": run_id,
            "sealed_at": sealed_at,
        }
    task.metadata["artifact_seals"] = seals


def commit_finalizer_stage_publish(
    orchestrator,
    *,
    task,
    subtask,
    stage_plan: dict[str, object],
) -> tuple[bool, str]:
    """Commit staged finalizer outputs to canonical deliverables atomically."""
    if not bool(stage_plan.get("enabled", False)):
        return True, ""
    workspace = Path(task.workspace) if task.workspace else None
    if workspace is None:
        return False, "Transactional publish failed: workspace unavailable."
    stage_to_canonical = dict(stage_plan.get("stage_to_canonical", {}))
    if not stage_to_canonical:
        return False, "Transactional publish failed: no staged outputs declared."

    canonical_paths = sorted({
        str(path).strip()
        for path in stage_to_canonical.values()
        if str(path).strip()
    })
    seals_snapshot = orchestrator._artifact_seals_snapshot(task)
    backup_paths: dict[str, str] = {}
    installed_paths: set[str] = set()
    try:
        workspace_resolved = workspace.resolve()
        for stage_rel, canonical_rel in stage_to_canonical.items():
            stage_path = (workspace_resolved / str(stage_rel)).resolve()
            stage_path.relative_to(workspace_resolved)
            if not stage_path.exists() or not stage_path.is_file():
                return False, (
                    "Transactional publish failed: missing staged output "
                    f"'{stage_rel}' for canonical '{canonical_rel}'."
                )

        for stage_rel, canonical_rel in stage_to_canonical.items():
            stage_path = (workspace_resolved / str(stage_rel)).resolve()
            canonical_path = (workspace_resolved / str(canonical_rel)).resolve()
            canonical_path.relative_to(workspace_resolved)
            canonical_path.parent.mkdir(parents=True, exist_ok=True)
            canonical_rel_text = str(canonical_rel).strip()
            if canonical_path.exists():
                backup_name = canonical_path.name + f".loom-backup-{uuid.uuid4().hex[:10]}"
                backup_path = canonical_path.with_name(backup_name)
                canonical_path.replace(backup_path)
                backup_paths[canonical_rel_text] = str(
                    backup_path.relative_to(workspace_resolved).as_posix(),
                )
            stage_path.replace(canonical_path)
            installed_paths.add(canonical_rel_text)

        orchestrator._seal_paths_after_commit(
            task=task,
            subtask_id=str(getattr(subtask, "id", "") or ""),
            paths=canonical_paths,
        )
        for backup_rel in backup_paths.values():
            backup_path = (workspace_resolved / backup_rel).resolve()
            backup_path.unlink(missing_ok=True)
        return True, ""
    except Exception as e:
        logger.debug(
            "Transactional stage+commit publish failed for %s/%s",
            task.id,
            subtask.id,
            exc_info=True,
        )
        try:
            workspace_resolved = workspace.resolve()
            for canonical_rel in installed_paths:
                path = (workspace_resolved / canonical_rel).resolve()
                path.unlink(missing_ok=True)
            for canonical_rel, backup_rel in backup_paths.items():
                canonical_path = (workspace_resolved / canonical_rel).resolve()
                backup_path = (workspace_resolved / backup_rel).resolve()
                if backup_path.exists():
                    backup_path.replace(canonical_path)
        except Exception:
            logger.debug(
                "Failed rollback during transactional publish failure for %s/%s",
                task.id,
                subtask.id,
                exc_info=True,
            )
        orchestrator._restore_artifact_seals_snapshot(task=task, snapshot=seals_snapshot)
        return False, f"Transactional stage+commit publish failed: {e}"


# Extracted output coordination + retry-context helpers

def _expected_deliverables_for_subtask(self, subtask: Subtask) -> list[str]:
    if self._process is None:
        return []
    deliverables = self._process.get_deliverables()
    if not deliverables:
        return []
    phase_hint = str(getattr(subtask, "phase_id", "") or "").strip()
    if phase_hint in deliverables:
        return [
            str(item).strip()
            for item in deliverables[phase_hint]
            if str(item).strip()
        ]
    if subtask.id in deliverables:
        return [
            str(item).strip()
            for item in deliverables[subtask.id]
            if str(item).strip()
        ]
    if len(deliverables) == 1:
        return [
            str(item).strip()
            for item in next(iter(deliverables.values()))
            if str(item).strip()
        ]
    phase_descriptions: dict[str, str] = {}
    for phase in getattr(self._process, "phases", []):
        phase_id = str(getattr(phase, "id", "")).strip()
        if not phase_id:
            continue
        phase_descriptions[phase_id] = str(
            getattr(phase, "description", ""),
        ).strip()
    phase_id = infer_phase_id_for_subtask(
        subtask_id=subtask.id,
        text=" ".join([
            str(getattr(subtask, "description", "")).strip(),
            str(getattr(subtask, "acceptance_criteria", "")).strip(),
        ]).strip(),
        phase_ids=list(deliverables.keys()),
        phase_descriptions=phase_descriptions,
        phase_deliverables=deliverables,
    )
    if phase_id in deliverables:
        return [
            str(item).strip()
            for item in deliverables[phase_id]
            if str(item).strip()
        ]
    return []

def _output_write_policy_for_subtask(
    self,
    *,
    subtask: Subtask,
) -> dict[str, object]:
    canonical_deliverables = self._expected_deliverables_for_subtask(subtask)
    phase_id = str(getattr(subtask, "phase_id", "") or "").strip() or subtask.id
    output_strategy = self._phase_output_strategy(phase_id)
    output_role = self._subtask_output_role(subtask)
    expected_deliverables: list[str] = []
    forbidden_deliverables: list[str] = []
    if output_strategy == "fan_in" and canonical_deliverables:
        if output_role == self._OUTPUT_ROLE_PHASE_FINALIZER:
            expected_deliverables = list(canonical_deliverables)
        else:
            forbidden_deliverables = list(canonical_deliverables)
    else:
        expected_deliverables = list(canonical_deliverables)
        output_role = self._OUTPUT_ROLE_WORKER
    return {
        "output_role": output_role,
        "output_strategy": output_strategy,
        "expected_deliverables": expected_deliverables,
        "forbidden_deliverables": forbidden_deliverables,
    }

def _files_from_attempts(attempts: list[AttemptRecord], *, max_items: int = 24) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for attempt in attempts:
        raw_calls = getattr(attempt, "successful_tool_calls", [])
        if not isinstance(raw_calls, list):
            continue
        for call in raw_calls:
            result = getattr(call, "result", None)
            changed = getattr(result, "files_changed", [])
            if not isinstance(changed, list):
                continue
            for item in changed:
                text = str(item or "").strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                files.append(text)
                if len(files) >= max_items:
                    return files
    return files

def _files_from_tool_calls(tool_calls: list, *, max_items: int = 24) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    if not isinstance(tool_calls, list):
        return files
    for call in tool_calls:
        result = getattr(call, "result", None)
        if not getattr(result, "success", False):
            continue
        changed = getattr(result, "files_changed", [])
        if not isinstance(changed, list):
            continue
        for item in changed:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            files.append(text)
            if len(files) >= max_items:
                return files
    return files

def _augment_retry_context_for_outputs(
    self,
    *,
    subtask: Subtask,
    attempts: list[AttemptRecord],
    strategy: RetryStrategy,
    expected_deliverables: list[str],
    forbidden_deliverables: list[str],
    base_context: str,
) -> str:
    lines: list[str] = []
    existing_files = self._files_from_attempts(attempts)
    if existing_files:
        lines.append("EDIT-IN-PLACE FILES (do not fork or rename):")
        for path in existing_files:
            lines.append(f"- {path}")
        lines.append(
            "Do not create alternate copies such as *-v2.*, *_v2.*, *-copy.*, "
            "or similarly suffixed variants."
        )

    output_role = str(getattr(subtask, "output_role", "") or "").strip().lower()
    output_strategy = str(getattr(subtask, "output_strategy", "") or "").strip().lower()
    phase_id = str(getattr(subtask, "phase_id", "") or "").strip() or subtask.id
    intermediate_root = self._output_intermediate_root()
    if output_strategy == "fan_in":
        if output_role == self._OUTPUT_ROLE_PHASE_FINALIZER:
            lines.append("OUTPUT COORDINATION MODE: FAN-IN PHASE FINALIZER")
            lines.append(
                "Consolidate worker intermediate artifacts into canonical deliverables."
            )
            lines.append(
                "Read worker artifacts from phase intermediate root before publishing:"
            )
            lines.append(
                f"- {intermediate_root}/<run-id>/{phase_id}/",
            )
        else:
            lines.append("OUTPUT COORDINATION MODE: FAN-IN WORKER")
            lines.append(
                "This worker must not modify canonical deliverables for the phase."
            )
            lines.append("Write intermediate artifacts under:")
            lines.append(f"- {intermediate_root}/<run-id>/{phase_id}/{subtask.id}/")
    if forbidden_deliverables:
        lines.append("CANONICAL DELIVERABLE FILES FORBIDDEN IN THIS SUBTASK:")
        for name in forbidden_deliverables:
            lines.append(f"- {name}")
        lines.append(
            "Do not write these canonical filenames in this step."
        )
    if expected_deliverables:
        lines.append("CANONICAL DELIVERABLE FILES FOR THIS SUBTASK:")
        for name in expected_deliverables:
            lines.append(f"- {name}")
        lines.append(
            "Write/update only these deliverable filenames for this phase. "
            "If fixing verification issues, patch these files in place."
        )
        if output_strategy != "fan_in":
            lines.append(
                "Direct-mode deliverable rule: gather all required evidence before "
                "the first canonical write. Each listed deliverable may be written "
                "at most once in a subtask attempt; after all are written, stop "
                "calling tools and return your completion response."
            )
    if (
        strategy in {
            RetryStrategy.RATE_LIMIT,
            RetryStrategy.EVIDENCE_GAP,
            RetryStrategy.UNCONFIRMED_DATA,
        }
        and expected_deliverables
    ):
        lines.append(
            "Remediation scope: keep validated content and make only minimal edits "
            "needed to satisfy failed checks."
        )
    if not lines:
        return base_context
    block = "\n".join(lines)
    if base_context.strip():
        return f"{base_context}\n\n{block}"
    return block


# Extracted task finalization orchestration

def _apply_finalization_outcome(self, task: Task) -> None:
    """Apply final task status, telemetry, and event emissions before persistence."""
    completed, total = task.progress
    run_validity_summary = self._refresh_run_validity_scorecard(task)
    blocking_remediation_failures: list[str] = []
    blocked_subtasks: list[dict[str, object]] = []
    raw_blocked_subtasks = task.metadata.get("blocked_subtasks")
    if isinstance(raw_blocked_subtasks, list):
        for item in raw_blocked_subtasks:
            if not isinstance(item, dict):
                continue
            subtask_id = str(item.get("subtask_id", "")).strip()
            raw_reasons = item.get("reasons", [])
            reasons: list[str] = []
            if isinstance(raw_reasons, list):
                for reason in raw_reasons:
                    text = str(reason).strip()
                    if text:
                        reasons.append(text)
            else:
                text = str(raw_reasons).strip()
                if text:
                    reasons.append(text)
            if not subtask_id:
                continue
            blocked_subtasks.append({
                "subtask_id": subtask_id,
                "reasons": reasons,
            })
    queue = task.metadata.get("remediation_queue")
    if isinstance(queue, list):
        for item in queue:
            if not isinstance(item, dict):
                continue
            if not bool(item.get("blocking", False)):
                continue
            state = str(item.get("state", "queued")).strip().lower()
            if state != "resolved":
                subtask_id = str(item.get("subtask_id", "")).strip()
                terminal_reason = str(item.get("terminal_reason", "")).strip()
                last_error = str(item.get("last_error", "")).strip()
                label = subtask_id or str(item.get("id", "")).strip() or "unknown"
                if terminal_reason:
                    label = f"{label} ({terminal_reason})"
                elif last_error:
                    label = f"{label} ({last_error})"
                blocking_remediation_failures.append(
                    label,
                )

    all_done = (
        completed == total
        and total > 0
        and not blocking_remediation_failures
    )

    if task.status == TaskStatus.CANCELLED:
        for s in task.plan.subtasks:
            if s.status == SubtaskStatus.PENDING:
                s.status = SubtaskStatus.SKIPPED
        cancel_reason = ""
        if isinstance(task.metadata, dict):
            cancel_reason = str(task.metadata.get("cancel_reason", "") or "").strip()
        self._emit(TASK_CANCELLED, task.id, {
            "completed": completed,
            "total": total,
            "reason": cancel_reason or "cancel_requested",
            "outcome": "cancelled",
        })
    elif all_done:
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now().isoformat()
        self._emit(TASK_COMPLETED, task.id, {
            "completed": completed,
            "total": total,
            "reason": "all_subtasks_completed",
            "outcome": "completed",
            "validity_summary": run_validity_summary,
        })
    else:
        task.status = TaskStatus.FAILED
        failed = [s for s in task.plan.subtasks if s.status == SubtaskStatus.FAILED]
        failure_reason = "subtask_failure"
        if blocking_remediation_failures:
            task.add_error(
                "remediation",
                "Blocking remediation unresolved for: "
                + ", ".join(blocking_remediation_failures),
            )
            failure_reason = "blocking_remediation_unresolved"
        if blocked_subtasks:
            labels = ", ".join(
                entry["subtask_id"] for entry in blocked_subtasks
                if isinstance(entry, dict) and entry.get("subtask_id")
            )
            task.add_error(
                "scheduler",
                "Execution stalled with blocked pending subtasks: "
                + (labels or "unknown"),
            )
            failure_reason = "blocked_pending_subtasks"
        self._emit(TASK_FAILED, task.id, {
            "completed": completed,
            "total": total,
            "failed_subtasks": [s.id for s in failed],
            "reason": failure_reason,
            "outcome": "failed",
            "blocking_remediation_failures": blocking_remediation_failures,
            "blocked_subtasks": blocked_subtasks,
            "validity_summary": run_validity_summary,
        })

    self._emit_run_validity_scorecard(task)
    self._emit_telemetry_run_summary(task)
    self._export_validity_scorecard_json(task)


def _finalize_task(self, task: Task) -> Task:
    """Finalize task: set status, emit events, and persist synchronously."""
    _apply_finalization_outcome(self, task)
    self._state.save(task)
    return task


async def _finalize_task_async(self, task: Task) -> Task:
    """Finalize task: set status, emit events, and persist without blocking the loop."""
    _apply_finalization_outcome(self, task)
    await self._save_task_state(task)
    return task
