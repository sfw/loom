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

_DEFAULT_MUTATION_PATH_KEYS = frozenset({
    "path",
    "destination",
    "destination_path",
    "source",
    "file",
    "file_path",
    "filepath",
    "target",
    "target_path",
    "target_file",
    "dest_path",
    "output_path",
    "output_file",
    "output_json_path",
    "searchable_output_path",
    "save_path",
    "export_path",
    "report_path",
})
_OUTPUT_MUTATION_PATH_KEYS = frozenset({
    "output_path",
    "output_file",
    "output_json_path",
    "searchable_output_path",
    "save_path",
    "export_path",
    "report_path",
    "destination",
    "destination_path",
    "dest_path",
    "target",
    "target_path",
    "target_file",
})


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
    is_mutating_tool: bool = False,
    write_mutating_tools: set[str] | frozenset[str],
    spreadsheet_write_operations: set[str] | frozenset[str],
) -> bool:
    name = str(tool_name or "").strip().lower()
    if not bool(is_mutating_tool) and name not in write_mutating_tools:
        return False
    if name != "spreadsheet":
        return True
    operation = str(tool_args.get("operation", "")).strip().lower()
    return operation in spreadsheet_write_operations


def _coerce_mutation_target_arg_keys(
    mutation_target_arg_keys: tuple[str, ...] | list[str] | set[str] | frozenset[str] | None,
) -> tuple[str, ...]:
    if not mutation_target_arg_keys:
        return tuple()
    normalized: list[str] = []
    seen: set[str] = set()
    for item in mutation_target_arg_keys:
        key = str(item or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return tuple(normalized)


def _collect_path_candidates(
    value: Any,
    *,
    candidate_keys: set[str],
    found: list[tuple[str, str]],
) -> None:
    if isinstance(value, dict):
        for raw_key, raw_value in value.items():
            key = str(raw_key or "").strip().lower()
            if key in candidate_keys:
                if isinstance(raw_value, str):
                    text = raw_value.strip()
                    if text:
                        found.append((key, text))
                elif isinstance(raw_value, (list, tuple, set)):
                    for item in raw_value:
                        text = str(item or "").strip()
                        if text:
                            found.append((key, text))
            _collect_path_candidates(
                raw_value,
                candidate_keys=candidate_keys,
                found=found,
            )
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_path_candidates(
                item,
                candidate_keys=candidate_keys,
                found=found,
            )


def target_paths_for_policy(
    *,
    tool_name: str,
    tool_args: dict,
    workspace: Path | None,
    is_mutating_tool: bool = False,
    mutation_target_arg_keys: tuple[str, ...] | list[str] | set[str] | frozenset[str] | None = None,
    is_mutating_file_tool_fn: Any,
) -> list[str]:
    try:
        is_mutating = is_mutating_file_tool_fn(
            tool_name,
            tool_args,
            is_mutating_tool=is_mutating_tool,
        )
    except TypeError:
        is_mutating = is_mutating_file_tool_fn(tool_name, tool_args)
    if not is_mutating:
        return []

    explicit_keys = _coerce_mutation_target_arg_keys(mutation_target_arg_keys)
    candidate_keys = set(explicit_keys) if explicit_keys else set(_DEFAULT_MUTATION_PATH_KEYS)
    found: list[tuple[str, str]] = []
    _collect_path_candidates(
        tool_args,
        candidate_keys=candidate_keys,
        found=found,
    )
    if not found:
        return []

    # When output-oriented keys are present, treat generic `path` values as likely
    # read inputs unless the tool explicitly declared otherwise.
    if any(key in _OUTPUT_MUTATION_PATH_KEYS for key, _ in found):
        found = [(key, value) for key, value in found if key in _OUTPUT_MUTATION_PATH_KEYS]

    result: list[str] = []
    seen: set[str] = set()
    for _key, raw in found:
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
    is_mutating_tool: bool = False,
    mutation_target_arg_keys: tuple[str, ...] | list[str] | set[str] | frozenset[str] | None = None,
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
        is_mutating_tool=is_mutating_tool,
        mutation_target_arg_keys=mutation_target_arg_keys,
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
    is_mutating_tool: bool = False,
    mutation_target_arg_keys: tuple[str, ...] | list[str] | set[str] | frozenset[str] | None = None,
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
        is_mutating_tool=is_mutating_tool,
        mutation_target_arg_keys=mutation_target_arg_keys,
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
            "Sealed artifact mutation blocked: target is sealed and verified "
            f"({path_preview}) but no confirmation evidence was recorded "
            f"after seal time {latest_seal}. "
            "Gather evidence first (for example: read_file, spreadsheet read/summary, "
            "web_search/web_fetch, or fact_checker), then retry."
        )
    return (
        "Sealed artifact mutation blocked: target is sealed and verified "
        f"({path_preview}) but no confirmation evidence is recorded in this subtask. "
        "Gather evidence first (for example: read_file, spreadsheet read/summary, "
        "web_search/web_fetch, or fact_checker), then retry."
    )


def snapshot_tracked_artifact_hashes(
    *,
    task: Task,
    workspace: Path | None,
    artifact_seal_registry: Any,
) -> dict[str, str]:
    if workspace is None:
        return {}
    seals = artifact_seal_registry(task)
    if not isinstance(seals, dict) or not seals:
        return {}
    try:
        workspace_resolved = workspace.resolve()
    except Exception:
        return {}
    snapshot: dict[str, str] = {}
    for relpath, seal in seals.items():
        if not isinstance(seal, dict):
            continue
        if not str(seal.get("sha256", "") or "").strip():
            continue
        try:
            artifact_path = (workspace_resolved / str(relpath)).resolve()
            artifact_path.relative_to(workspace_resolved)
        except Exception:
            continue
        if not artifact_path.exists() or not artifact_path.is_file():
            snapshot[str(relpath)] = ""
            continue
        try:
            payload = artifact_path.read_bytes()
        except Exception:
            continue
        snapshot[str(relpath)] = hashlib.sha256(payload).hexdigest()
    return snapshot


def unexpected_sealed_mutation_paths(
    *,
    task: Task,
    workspace: Path | None,
    tool_name: str,
    tool_args: dict,
    tool_result: ToolResult,
    is_mutating_tool: bool,
    mutation_target_arg_keys: tuple[str, ...] | list[str] | set[str] | frozenset[str] | None,
    pre_call_hashes: dict[str, str],
    artifact_seal_registry: Any,
    mutation_paths_for_reseal: Any,
    normalize_path_for_policy: Any,
) -> list[str]:
    if workspace is None or not is_mutating_tool:
        return []
    if tool_result is None or not tool_result.success:
        return []
    if not pre_call_hashes:
        return []

    seals = artifact_seal_registry(task)
    if not isinstance(seals, dict) or not seals:
        return []
    try:
        workspace_resolved = workspace.resolve()
    except Exception:
        return []

    expected = mutation_paths_for_reseal(
        tool_name=tool_name,
        tool_args=tool_args,
        workspace=workspace,
        tool_result=tool_result,
        is_mutating_tool=is_mutating_tool,
        mutation_target_arg_keys=mutation_target_arg_keys,
    )
    expected_set: set[str] = set()
    for path in expected:
        normalized = normalize_path_for_policy(str(path), workspace)
        if normalized:
            expected_set.add(normalized)

    unexpected: list[str] = []
    seen: set[str] = set()
    for relpath, seal in seals.items():
        rel = str(relpath or "").strip()
        if not rel or rel in seen:
            continue
        if not isinstance(seal, dict):
            continue
        if not str(seal.get("sha256", "") or "").strip():
            continue
        try:
            artifact_path = (workspace_resolved / rel).resolve()
            artifact_path.relative_to(workspace_resolved)
        except Exception:
            continue
        if artifact_path.exists() and artifact_path.is_file():
            try:
                payload = artifact_path.read_bytes()
            except Exception:
                continue
            current_hash = hashlib.sha256(payload).hexdigest()
        else:
            current_hash = ""
        previous_hash = str(pre_call_hashes.get(rel, ""))
        if current_hash == previous_hash:
            continue
        normalized_rel = normalize_path_for_policy(rel, workspace)
        if normalized_rel and normalized_rel not in expected_set:
            seen.add(rel)
            unexpected.append(normalized_rel)
    return unexpected


def reseal_tracked_artifacts_after_mutation(
    *,
    task: Task,
    workspace: Path | None,
    tool_name: str,
    tool_args: dict,
    tool_result: ToolResult,
    is_mutating_tool: bool,
    mutation_target_arg_keys: tuple[str, ...] | list[str] | set[str] | frozenset[str] | None,
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
        is_mutating_tool=is_mutating_tool,
        mutation_target_arg_keys=mutation_target_arg_keys,
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
