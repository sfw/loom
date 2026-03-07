"""Policy helpers for deliverable and sealed-artifact mutation rules."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from loom.engine.path_policy import (
    looks_like_deliverable_variant as shared_looks_like_deliverable_variant,
)
from loom.engine.path_policy import (
    normalize_deliverable_paths as shared_normalize_deliverable_paths,
)
from loom.engine.path_policy import normalize_path_for_policy as shared_normalize_path_for_policy
from loom.state.task_state import Task
from loom.tools.registry import ToolResult

from .types import ToolCallRecord


def normalize_path_for_policy(path_text: str, workspace: Path | None) -> str:
    return shared_normalize_path_for_policy(path_text, workspace)


def normalize_deliverable_paths(
    expected_deliverables: list[str],
    *,
    workspace: Path | None,
) -> list[str]:
    return shared_normalize_deliverable_paths(
        expected_deliverables,
        workspace=workspace,
    )


def is_mutating_file_tool(
    *,
    tool_name: str,
    tool_args: dict,
    write_mutating_tools: set[str] | frozenset[str],
    spreadsheet_write_operations: set[str] | frozenset[str],
) -> bool:
    name = str(tool_name or "").strip().lower()
    if name not in write_mutating_tools:
        return False
    if name != "spreadsheet":
        return True
    operation = str(tool_args.get("operation", "")).strip().lower()
    return operation in spreadsheet_write_operations


def target_paths_for_policy(
    *,
    tool_name: str,
    tool_args: dict,
    workspace: Path | None,
    is_mutating_file_tool_fn: Any,
) -> list[str]:
    if not is_mutating_file_tool_fn(tool_name, tool_args):
        return []
    keys = ("path", "destination", "source", "file", "file_path", "filepath")
    result: list[str] = []
    seen: set[str] = set()
    for key in keys:
        raw = tool_args.get(key)
        if raw is None:
            continue
        value = normalize_path_for_policy(str(raw), workspace)
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def looks_like_deliverable_variant(
    *,
    candidate: str,
    canonical: str,
    variant_suffix_markers: tuple[str, ...],
) -> bool:
    return shared_looks_like_deliverable_variant(
        candidate=candidate,
        canonical=canonical,
        variant_suffix_markers=variant_suffix_markers,
    )


def mutation_target_from_args(arguments: dict[str, Any]) -> str:
    for key in ("path", "dest_path", "destination", "target", "url", "uri"):
        value = arguments.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text.lower()
    return ""


def mutation_idempotency_key(
    *,
    task: Task,
    subtask_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    normalize_run_id: Any,
    stable_json_digest: Any,
) -> tuple[str, str]:
    args_hash = stable_json_digest(arguments)
    seed = "|".join([
        normalize_run_id(task) or task.id,
        subtask_id,
        tool_name,
        mutation_target_from_args(arguments),
        args_hash,
    ])
    idem_key = hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()
    return idem_key, args_hash


def validate_deliverable_write_policy(
    *,
    tool_name: str,
    tool_args: dict,
    workspace: Path | None,
    expected_deliverables: list[str],
    forbidden_deliverables: list[str],
    allowed_output_prefixes: list[str],
    enforce_deliverable_paths: bool,
    edit_existing_only: bool,
    normalize_deliverable_paths: Any,
    target_paths_for_policy: Any,
    looks_like_deliverable_variant: Any,
) -> str | None:
    canonical = list(expected_deliverables)
    forbidden = list(forbidden_deliverables)
    prefixes = normalize_deliverable_paths(
        list(allowed_output_prefixes),
        workspace=None,
    )
    if not canonical and not forbidden and not prefixes:
        return None
    paths = target_paths_for_policy(
        tool_name=tool_name,
        tool_args=tool_args,
        workspace=workspace,
    )
    if not paths:
        return None
    canonical_set = set(canonical)
    forbidden_set = set(forbidden)
    variant_candidates = list(dict.fromkeys([*canonical, *forbidden]))
    for path in paths:
        if path in forbidden_set:
            blocked = ", ".join(forbidden)
            return (
                "reason_code=forbidden_output_path; "
                "Canonical deliverable ownership violation: "
                f"'{path}' is reserved for a phase finalizer subtask. "
                f"Forbidden canonical path(s): {blocked}."
            )
        if path in canonical_set:
            continue
        if any(
            looks_like_deliverable_variant(candidate=path, canonical=item)
            for item in variant_candidates
        ):
            allowed = ", ".join(variant_candidates)
            return (
                "reason_code=forbidden_output_path; "
                "Canonical deliverable policy violation: "
                f"'{path}' looks like a versioned copy of a required file. "
                f"Update canonical file(s) instead: {allowed}."
            )
        if prefixes:
            if any(path == prefix or path.startswith(prefix + "/") for prefix in prefixes):
                continue
            allowed_prefixes = ", ".join(prefixes)
            return (
                "reason_code=forbidden_output_path; "
                "Fan-in worker output path violation: "
                f"'{path}' must be under intermediate artifact prefix(es): "
                f"{allowed_prefixes}."
            )
    if enforce_deliverable_paths and canonical:
        extras = [path for path in paths if path not in canonical_set]
        if extras:
            allowed = ", ".join(canonical)
            return (
                "reason_code=forbidden_output_path; "
                "Canonical deliverable policy violation: "
                "retry/remediation writes must stay in canonical deliverables. "
                f"Unexpected target(s): {', '.join(extras)}. "
                f"Allowed: {allowed}."
            )
    if (
        edit_existing_only
        and str(tool_name or "").strip().lower() in {"move_file", "delete_file"}
    ):
        return (
            "reason_code=forbidden_output_path; "
            "Remediation requires in-place edits to canonical deliverables. "
            "Do not rename or delete files during retry."
        )
    return None


def validate_sealed_artifact_mutation_policy(
    *,
    task: Task,
    tool_name: str,
    tool_args: dict,
    workspace: Path | None,
    prior_successful_tool_calls: list[ToolCallRecord],
    current_tool_calls: list[ToolCallRecord],
    target_paths_for_policy: Any,
    artifact_seal_registry: Any,
    seal_origin_is_verified: Any,
    latest_seal_timestamp: Any,
    has_post_seal_confirmation_evidence: Any,
) -> str | None:
    paths = target_paths_for_policy(
        tool_name=tool_name,
        tool_args=tool_args,
        workspace=workspace,
    )
    if not paths:
        return None

    seals = artifact_seal_registry(task)
    protected_paths: list[tuple[str, dict[str, object]]] = []
    for path in paths:
        seal = seals.get(path)
        if not isinstance(seal, dict):
            continue
        if not str(seal.get("sha256", "") or "").strip():
            continue
        if not seal_origin_is_verified(task=task, seal=seal):
            continue
        protected_paths.append((path, seal))
    if not protected_paths:
        return None

    latest_seal = latest_seal_timestamp(protected_paths)
    has_confirmation = has_post_seal_confirmation_evidence(
        baseline_timestamp=latest_seal,
        prior_successful_tool_calls=prior_successful_tool_calls,
        current_tool_calls=current_tool_calls,
    )
    if has_confirmation:
        return None

    unique_paths = []
    seen: set[str] = set()
    for path, _seal in protected_paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)
    path_preview = ", ".join(unique_paths[:3])
    if len(unique_paths) > 3:
        path_preview = f"{path_preview}, ..."
    if latest_seal:
        return (
            "Sealed artifact edit blocked: target is sealed and verified "
            f"({path_preview}) but no confirmation evidence was recorded "
            f"after seal time {latest_seal}. "
            "Gather evidence first (for example: read_file, web_search/web_fetch, "
            "or fact_checker), then retry the edit."
        )
    return (
        "Sealed artifact edit blocked: target is sealed and verified "
        f"({path_preview}) but no confirmation evidence is recorded in this subtask. "
        "Gather evidence first (for example: read_file, web_search/web_fetch, "
        "or fact_checker), then retry the edit."
    )


def reseal_tracked_artifacts_after_mutation(
    *,
    task: Task,
    workspace: Path | None,
    tool_name: str,
    tool_args: dict,
    tool_result: ToolResult,
    subtask_id: str,
    tool_call_id: str,
    artifact_seal_registry: Any,
    mutation_paths_for_reseal: Any,
    normalize_path_for_policy: Any,
    seal_origin_is_verified: Any,
) -> int:
    if workspace is None or tool_result is None or not tool_result.success:
        return 0

    seals = artifact_seal_registry(task)
    if not seals:
        return 0

    affected_paths = mutation_paths_for_reseal(
        tool_name=tool_name,
        tool_args=tool_args,
        workspace=workspace,
        tool_result=tool_result,
    )
    if not affected_paths:
        return 0

    normalized_tool = str(tool_name or "").strip().lower()
    move_source = normalize_path_for_policy(str(tool_args.get("source", "")), workspace)
    move_destination = normalize_path_for_policy(
        str(tool_args.get("destination", "")),
        workspace,
    )
    source_seal = seals.get(move_source)
    source_was_verified = (
        normalized_tool == "move_file"
        and isinstance(source_seal, dict)
        and seal_origin_is_verified(task=task, seal=source_seal)
    )

    tracked_paths: list[str] = []
    seen: set[str] = set()
    for path in affected_paths:
        if path in seen:
            continue
        if path in seals:
            tracked_paths.append(path)
            seen.add(path)
            continue
        if (
            normalized_tool == "move_file"
            and path == move_destination
            and move_source in seals
        ):
            tracked_paths.append(path)
            seen.add(path)

    if not tracked_paths:
        return 0

    try:
        workspace_resolved = workspace.resolve()
    except Exception:
        return 0

    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    run_id = str(metadata.get("run_id", "") or "").strip() if isinstance(metadata, dict) else ""
    resealed_at = datetime.now().isoformat()
    updated = 0

    for relpath in tracked_paths:
        previous = seals.get(relpath)
        previous_dict = dict(previous) if isinstance(previous, dict) else {}
        verified_origin = seal_origin_is_verified(task=task, seal=previous_dict)
        if (
            normalized_tool == "move_file"
            and relpath == move_destination
            and source_was_verified
        ):
            verified_origin = True
        try:
            artifact_path = (workspace_resolved / relpath).resolve()
            artifact_path.relative_to(workspace_resolved)
        except Exception:
            continue

        if not artifact_path.exists() or not artifact_path.is_file():
            if relpath in seals:
                seals.pop(relpath, None)
                updated += 1
            continue

        try:
            payload = artifact_path.read_bytes()
        except Exception:
            continue

        observed_sha = hashlib.sha256(payload).hexdigest()
        previous_sha = str(previous_dict.get("sha256", "") or "").strip()

        next_seal = dict(previous_dict)
        next_seal.update({
            "path": relpath,
            "sha256": observed_sha,
            "size_bytes": int(len(payload)),
            "tool": normalized_tool,
            "tool_call_id": str(tool_call_id or ""),
            "subtask_id": str(subtask_id or ""),
            "sealed_at": resealed_at,
            "resealed_after_mutation": True,
            "resealed_reason": "post_edit_evidence_confirmed",
        })
        if run_id:
            next_seal["run_id"] = run_id
        if previous_sha and previous_sha != observed_sha:
            next_seal["previous_sha256"] = previous_sha
        else:
            next_seal.pop("previous_sha256", None)
        if verified_origin:
            next_seal["verified_origin"] = True
        else:
            next_seal.pop("verified_origin", None)

        seals[relpath] = next_seal
        updated += 1

    if updated > 0:
        task.metadata["artifact_seals"] = seals
    return updated
