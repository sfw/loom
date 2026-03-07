"""Placeholder-contradiction scan and claim-failure helpers."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from loom.processes.phase_alignment import infer_phase_id_for_subtask
from loom.state.task_state import Subtask
from loom.utils.concurrency import run_blocking_io

from .types import VerificationResult


def is_placeholder_claim_failure(gates, result: VerificationResult) -> bool:
    """Return True when a failed result appears to be placeholder-claim related."""
    if result.passed:
        return False
    reason_code = str(result.reason_code or "").strip().lower()
    reason_codes = getattr(gates, "_PLACEHOLDER_CLAIM_REASON_CODES", set())
    if reason_code in reason_codes:
        return True
    issue_text = ""
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    raw_issues = metadata.get("issues", [])
    if isinstance(raw_issues, list):
        issue_text = " ".join(str(item or "") for item in raw_issues)
    haystack = " ".join([
        reason_code,
        str(result.feedback or ""),
        issue_text,
    ])
    pattern = getattr(gates, "_PLACEHOLDER_MARKER_PATTERN")
    return bool(pattern.search(haystack))


def expected_deliverables_for_subtask(gates, subtask: Subtask) -> list[str]:
    """Resolve expected deliverables for a subtask from the configured process."""
    process = gates._process
    if process is None:
        return []
    deliverables = process.get_deliverables()
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
    for phase in getattr(process, "phases", []):
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


def files_changed(tool_calls: list) -> list[str]:
    """Collect stable-order unique file-change paths from tool calls."""
    files: list[str] = []
    seen: set[str] = set()
    for call in tool_calls:
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
    return files


def to_nonempty_int(
    value: object,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """Parse int with default and clamp bounds."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    parsed = max(minimum, parsed)
    parsed = min(maximum, parsed)
    return parsed


def to_bool(raw: object, default: bool) -> bool:
    """Parse permissive bool values used by verification config knobs."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    return bool(default)


def normalized_scan_suffixes(gates, raw: object) -> tuple[str, ...]:
    """Normalize configured suffix allowlist for contradiction scanning."""
    if isinstance(raw, str):
        parts = re.split(r"[,\n;]+", raw)
    elif isinstance(raw, (list, tuple, set)):
        parts = list(raw)
    else:
        parts = []
    normalized: list[str] = []
    for part in parts:
        text = str(part or "").strip().lower()
        if not text:
            continue
        text = text.lstrip("*")
        if not text.startswith("."):
            text = f".{text}"
        if text and text not in normalized:
            normalized.append(text)
    if not normalized:
        return getattr(gates, "_DEFAULT_CONTRADICTION_SCAN_ALLOWED_SUFFIXES")
    return tuple(normalized)


def normalize_candidate_path(
    *,
    workspace: Path | None,
    raw_path: object,
) -> str | None:
    """Normalize candidate paths to workspace-relative POSIX strings."""
    if workspace is None:
        return None
    text = str(raw_path or "").strip()
    if not text:
        return None

    workspace_root = Path(os.path.normpath(str(workspace)))
    path = Path(text)
    rel_text = ""
    if path.is_absolute():
        normalized_abs = Path(os.path.normpath(str(path)))
        try:
            rel_text = normalized_abs.relative_to(workspace_root).as_posix()
        except ValueError:
            return None
    else:
        normalized_rel = Path(os.path.normpath(text))
        if normalized_rel.is_absolute():
            return None
        if any(part == ".." for part in normalized_rel.parts):
            return None
        rel_text = normalized_rel.as_posix()

    rel_text = rel_text.strip()
    if rel_text in {"", "."}:
        return None
    return rel_text


def normalize_candidate_bucket(
    *,
    workspace: Path | None,
    raw_paths: list[str],
) -> list[str]:
    """Normalize + dedupe a bucket of candidate paths."""
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_paths:
        rel_path = normalize_candidate_path(workspace=workspace, raw_path=item)
        if not rel_path or rel_path in seen:
            continue
        seen.add(rel_path)
        normalized.append(rel_path)
    return normalized


def evidence_artifact_paths(evidence_records: list[dict] | None) -> list[str]:
    """Extract possible artifact paths from evidence records."""
    paths: list[str] = []
    seen: set[str] = set()
    for record in evidence_records or []:
        if not isinstance(record, dict):
            continue
        candidates: list[object] = [
            record.get("artifact_workspace_relpath"),
            record.get("artifact_path"),
            record.get("path"),
            record.get("file_path"),
        ]
        facets = record.get("facets")
        if isinstance(facets, dict):
            candidates.extend([
                facets.get("artifact_workspace_relpath"),
                facets.get("artifact_path"),
                facets.get("path"),
                facets.get("file_path"),
            ])
        for item in candidates:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            paths.append(text)
    return paths


def path_has_symlink_component(
    *,
    workspace: Path,
    path: Path,
) -> bool:
    """Return True when a candidate path traverses symlink components."""
    workspace_root = workspace.resolve(strict=False)
    current = path
    try:
        current.relative_to(workspace_root)
    except ValueError:
        return True
    while True:
        try:
            current.relative_to(workspace_root)
        except ValueError:
            return True
        if current == workspace_root:
            return False
        try:
            if current.is_symlink():
                return True
        except OSError:
            return True
        parent = current.parent
        if parent == current:
            return True
        current = parent


def scan_placeholder_markers(
    gates,
    *,
    workspace: Path | None,
    candidate_data: dict[str, object],
) -> dict[str, object]:
    """Scan candidate workspace files for deterministic placeholder markers."""
    if workspace is None:
        return {
            "scan_mode": "targeted_only",
            "scanned_files": [],
            "scanned_file_count": 0,
            "scanned_total_bytes": 0,
            "matched_files": [],
            "matched_file_count": 0,
            "coverage_sufficient": False,
            "coverage_insufficient_reason": "workspace_unavailable",
            "candidate_source_counts": {
                "canonical": 0,
                "current_attempt": 0,
                "prior_attempt": 0,
                "evidence_artifact": 0,
                "fallback": 0,
            },
            "cap_exhausted": False,
            "cap_exhaustion_reason": "",
            "skipped_large_file_count": 0,
            "skipped_binary_file_count": 0,
            "skipped_symlink_count": 0,
            "skipped_suffix_count": 0,
        }
    ordered_candidates = candidate_data.get("ordered_candidates", [])
    if not isinstance(ordered_candidates, list):
        ordered_candidates = []
    canonical_candidates = candidate_data.get("canonical_candidates", set())
    if not isinstance(canonical_candidates, set):
        canonical_candidates = set()
    changed_candidates = candidate_data.get("changed_candidates", set())
    if not isinstance(changed_candidates, set):
        changed_candidates = set()
    candidate_source_counts = candidate_data.get("candidate_source_counts", {})
    if not isinstance(candidate_source_counts, dict):
        candidate_source_counts = {}

    max_files = gates._to_nonempty_int(
        getattr(gates._config, "contradiction_scan_max_files", 80),
        80,
        minimum=1,
        maximum=1000,
    )
    max_total_bytes = gates._to_nonempty_int(
        getattr(gates._config, "contradiction_scan_max_total_bytes", 2_500_000),
        2_500_000,
        minimum=1_024,
        maximum=50_000_000,
    )
    max_file_bytes = gates._to_nonempty_int(
        getattr(gates._config, "contradiction_scan_max_file_bytes", 300_000),
        300_000,
        minimum=1,
        maximum=10_000_000,
    )
    max_file_bytes = min(max_file_bytes, max_total_bytes)
    min_files_for_sufficiency = gates._to_nonempty_int(
        getattr(
            gates._config,
            "contradiction_scan_min_files_for_sufficiency",
            2,
        ),
        2,
        minimum=1,
        maximum=100,
    )
    strict_coverage = gates._to_bool(
        getattr(gates._config, "contradiction_guard_strict_coverage", True),
        True,
    )
    allowed_suffixes = set(
        gates._normalized_scan_suffixes(
            getattr(gates._config, "contradiction_scan_allowed_suffixes", ()),
        ),
    )

    workspace_root = workspace.resolve(strict=False)
    scanned_files: list[str] = []
    matched_files: list[str] = []
    scanned_total_bytes = 0
    scanned_canonical_count = 0
    scanned_changed_count = 0
    scanned_seen: set[str] = set()
    cap_exhausted = False
    cap_exhaustion_reason = ""
    skipped_large_file_count = 0
    skipped_binary_file_count = 0
    skipped_symlink_count = 0
    skipped_suffix_count = 0

    def mark_cap(reason: str) -> None:
        nonlocal cap_exhausted, cap_exhaustion_reason
        if not cap_exhausted:
            cap_exhausted = True
            cap_exhaustion_reason = reason

    def scan_candidate(rel_path: str, source: str) -> bool:
        nonlocal scanned_total_bytes
        nonlocal scanned_canonical_count
        nonlocal scanned_changed_count
        nonlocal skipped_large_file_count
        nonlocal skipped_binary_file_count
        nonlocal skipped_symlink_count
        nonlocal skipped_suffix_count

        if rel_path in scanned_seen:
            return False
        if len(scanned_files) >= max_files:
            mark_cap("max_files_reached")
            return False
        if scanned_total_bytes >= max_total_bytes:
            mark_cap("max_total_bytes_reached")
            return False

        fpath = workspace_root / rel_path
        try:
            resolved = fpath.resolve(strict=False)
        except OSError:
            return False
        try:
            resolved.relative_to(workspace_root)
        except ValueError:
            skipped_symlink_count += 1
            return False
        if gates._path_has_symlink_component(workspace=workspace_root, path=fpath):
            skipped_symlink_count += 1
            return False
        try:
            if not fpath.exists() or not fpath.is_file():
                return False
            suffix = fpath.suffix.lower()
            if suffix not in allowed_suffixes:
                skipped_suffix_count += 1
                return False
            fsize = int(fpath.stat().st_size)
        except OSError:
            return False
        if fsize > max_file_bytes:
            skipped_large_file_count += 1
            return False
        if scanned_total_bytes + fsize > max_total_bytes:
            mark_cap("max_total_bytes_reached")
            return False
        try:
            payload = fpath.read_bytes()
        except OSError:
            return False
        if b"\x00" in payload[:2048]:
            skipped_binary_file_count += 1
            return False

        scanned_seen.add(rel_path)
        scanned_files.append(rel_path)
        scanned_total_bytes += fsize
        if rel_path in canonical_candidates:
            scanned_canonical_count += 1
        if rel_path in changed_candidates:
            scanned_changed_count += 1

        content = payload.decode("utf-8", errors="replace")
        if gates._PLACEHOLDER_MARKER_PATTERN.search(content):
            matched_files.append(rel_path)
            return True
        if source == "fallback":
            candidate_source_counts["fallback"] = (
                int(candidate_source_counts.get("fallback", 0) or 0) + 1
            )
        return False

    for item in ordered_candidates:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        rel_path = str(item[0] or "").strip()
        source = str(item[1] or "").strip().lower() or "canonical"
        if not rel_path:
            continue
        if scan_candidate(rel_path, source):
            break
        if cap_exhausted:
            break

    scan_mode = "targeted_only"
    if not matched_files and not cap_exhausted:
        scan_mode = "targeted_plus_fallback"
        fallback_visit_cap = max(200, max_files * 40)
        visited_entries = 0
        stop_fallback = False
        for root, dirs, files in os.walk(workspace_root, topdown=True, followlinks=False):
            filtered_dirs: list[str] = []
            for name in sorted(dirs):
                if (
                    not name
                    or name.startswith(".")
                    or name in gates._CONTRADICTION_SCAN_EXCLUDED_DIRS
                ):
                    continue
                candidate_dir = Path(root) / name
                try:
                    if candidate_dir.is_symlink():
                        continue
                except OSError:
                    continue
                filtered_dirs.append(name)
            dirs[:] = filtered_dirs
            for filename in sorted(files):
                visited_entries += 1
                if visited_entries > fallback_visit_cap:
                    mark_cap("fallback_entry_cap_reached")
                    stop_fallback = True
                    break
                candidate_file = Path(root) / filename
                try:
                    if candidate_file.is_symlink():
                        skipped_symlink_count += 1
                        continue
                except OSError:
                    continue
                try:
                    rel_path = candidate_file.relative_to(workspace_root).as_posix()
                except ValueError:
                    continue
                if rel_path in scanned_seen:
                    continue
                if scan_candidate(rel_path, "fallback"):
                    stop_fallback = True
                    break
                if cap_exhausted:
                    stop_fallback = True
                    break
            if stop_fallback:
                break

    scanned_file_count = len(scanned_files)
    matched_file_count = len(matched_files)
    coverage_sufficient = False
    coverage_insufficient_reason = ""
    if strict_coverage:
        reasons: list[str] = []
        priority_scanned = scanned_canonical_count + scanned_changed_count
        if scanned_file_count <= 0:
            reasons.append("no_files_scanned")
        if priority_scanned <= 0:
            reasons.append("no_canonical_or_changed_candidate_scanned")
        allow_single_canonical = bool(candidate_data.get("single_canonical_candidate", False))
        if not (allow_single_canonical and scanned_canonical_count == 1):
            if scanned_file_count < min_files_for_sufficiency:
                reasons.append("minimum_file_coverage_not_met")
        if cap_exhausted and priority_scanned <= 0:
            reasons.append("cap_exhausted_before_priority_scan")
        coverage_sufficient = not reasons
        if reasons:
            coverage_insufficient_reason = ";".join(dict.fromkeys(reasons))
    else:
        coverage_sufficient = scanned_file_count > 0
        if not coverage_sufficient:
            coverage_insufficient_reason = "no_files_scanned"

    return {
        "scan_mode": scan_mode,
        "scanned_files": scanned_files,
        "scanned_file_count": scanned_file_count,
        "scanned_total_bytes": scanned_total_bytes,
        "matched_files": matched_files,
        "matched_file_count": matched_file_count,
        "coverage_sufficient": coverage_sufficient,
        "coverage_insufficient_reason": coverage_insufficient_reason,
        "candidate_source_counts": candidate_source_counts,
        "cap_exhausted": cap_exhausted,
        "cap_exhaustion_reason": cap_exhaustion_reason,
        "skipped_large_file_count": skipped_large_file_count,
        "skipped_binary_file_count": skipped_binary_file_count,
        "skipped_symlink_count": skipped_symlink_count,
        "skipped_suffix_count": skipped_suffix_count,
    }


def build_placeholder_scan_candidates(
    gates,
    *,
    subtask: Subtask,
    workspace: Path | None,
    tool_calls: list,
    evidence_tool_calls: list | None,
    evidence_records: list[dict] | None,
    expected_deliverables: list[str],
) -> dict[str, object]:
    """Construct prioritized candidate path buckets for contradiction scanning."""
    del subtask  # retained in signature for parity and future expansion
    canonical_paths = gates._normalize_candidate_bucket(
        workspace=workspace,
        raw_paths=expected_deliverables,
    )
    current_paths = gates._normalize_candidate_bucket(
        workspace=workspace,
        raw_paths=gates._files_changed(tool_calls),
    )
    prior_paths = gates._normalize_candidate_bucket(
        workspace=workspace,
        raw_paths=gates._files_changed(evidence_tool_calls or []),
    )
    evidence_paths = gates._normalize_candidate_bucket(
        workspace=workspace,
        raw_paths=gates._evidence_artifact_paths(evidence_records),
    )
    buckets: list[tuple[str, list[str]]] = [
        ("canonical", canonical_paths),
        ("current_attempt", current_paths),
        ("prior_attempt", prior_paths),
        ("evidence_artifact", evidence_paths),
    ]
    ordered_candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source, paths in buckets:
        for rel_path in paths:
            if rel_path in seen:
                continue
            seen.add(rel_path)
            ordered_candidates.append((rel_path, source))
    return {
        "ordered_candidates": ordered_candidates,
        "canonical_candidates": set(canonical_paths),
        "changed_candidates": set(current_paths) | set(prior_paths),
        "candidate_source_counts": {
            "canonical": len(canonical_paths),
            "current_attempt": len(current_paths),
            "prior_attempt": len(prior_paths),
            "evidence_artifact": len(evidence_paths),
            "fallback": 0,
        },
        "single_canonical_candidate": len(canonical_paths) == 1,
    }


async def apply_placeholder_contradiction_guard(
    gates,
    *,
    subtask: Subtask,
    result: VerificationResult,
    workspace: Path | None,
    tool_calls: list,
    evidence_tool_calls: list | None = None,
    evidence_records: list[dict] | None = None,
) -> VerificationResult:
    """Apply deterministic contradiction guard to placeholder-like failures."""
    if not bool(getattr(gates._config, "contradiction_guard_enabled", True)):
        return result
    if result.passed:
        return result
    if str(result.severity_class or "").strip().lower() == "hard_invariant":
        return result
    if not is_placeholder_claim_failure(gates, result):
        return result

    expected_deliverables = gates._expected_deliverables_for_subtask(subtask)
    candidate_data = gates._build_placeholder_scan_candidates(
        subtask=subtask,
        workspace=workspace,
        tool_calls=tool_calls,
        evidence_tool_calls=evidence_tool_calls,
        evidence_records=evidence_records,
        expected_deliverables=expected_deliverables,
    )
    raw_timeout = getattr(gates._config, "contradiction_scan_timeout_seconds", 8.0)
    scan_timeout_seconds = max(1.0, min(30.0, float(raw_timeout or 8.0)))
    try:
        scan = await asyncio.wait_for(
            run_blocking_io(
                gates._scan_placeholder_markers,
                workspace=workspace,
                candidate_data=candidate_data,
            ),
            timeout=scan_timeout_seconds,
        )
    except TimeoutError:
        scan = {
            "scan_mode": "targeted_plus_fallback",
            "scanned_files": [],
            "scanned_file_count": 0,
            "scanned_total_bytes": 0,
            "matched_files": [],
            "matched_file_count": 0,
            "coverage_sufficient": False,
            "coverage_insufficient_reason": "scan_timeout",
            "candidate_source_counts": {
                "canonical": 0,
                "current_attempt": 0,
                "prior_attempt": 0,
                "evidence_artifact": 0,
                "fallback": 0,
            },
            "cap_exhausted": True,
            "cap_exhaustion_reason": "scan_timeout",
            "skipped_large_file_count": 0,
            "skipped_binary_file_count": 0,
            "skipped_symlink_count": 0,
            "skipped_suffix_count": 0,
        }
    matched_file_count = int(scan.get("matched_file_count", 0) or 0)
    coverage_sufficient = bool(scan.get("coverage_sufficient", False))
    coverage_reason = str(scan.get("coverage_insufficient_reason", "") or "")

    metadata = dict(result.metadata) if isinstance(result.metadata, dict) else {}
    metadata["contradicted_reason_code"] = str(result.reason_code or "")
    metadata["deterministic_placeholder_scan"] = scan
    metadata["coverage_sufficient"] = coverage_sufficient
    metadata["contradiction_downgraded"] = False
    metadata["contradiction_detected_no_downgrade"] = False

    if matched_file_count > 0:
        metadata["contradiction_detected"] = False
        metadata["contradiction_marker_found"] = True
        return VerificationResult(
            tier=result.tier,
            passed=result.passed,
            confidence=float(result.confidence or 0.0),
            checks=list(result.checks or []),
            feedback=result.feedback,
            outcome=result.outcome,
            reason_code=result.reason_code,
            severity_class=result.severity_class,
            metadata=metadata,
        )

    if not coverage_sufficient:
        metadata["contradiction_detected"] = False
        metadata["contradiction_detected_no_downgrade"] = True
        metadata["coverage_insufficient_reason"] = coverage_reason
        return VerificationResult(
            tier=result.tier,
            passed=result.passed,
            confidence=float(result.confidence or 0.0),
            checks=list(result.checks or []),
            feedback=result.feedback,
            outcome=result.outcome,
            reason_code=result.reason_code,
            severity_class=result.severity_class,
            metadata=metadata,
        )

    metadata["contradiction_detected"] = True
    metadata["contradiction_downgraded"] = True
    metadata["contradiction_kind"] = (
        "placeholder_claim_without_deterministic_match"
    )
    feedback_parts = [
        str(result.feedback or "").strip(),
        (
            "Verifier placeholder/TODO claim contradicted by deterministic "
            "artifact scan; marking verification inconclusive for verifier-only retry."
        ),
    ]
    feedback = "\n".join(part for part in feedback_parts if part)
    return VerificationResult(
        tier=result.tier,
        passed=False,
        confidence=min(0.5, float(result.confidence or 0.5)),
        checks=list(result.checks or []),
        feedback=feedback,
        outcome="fail",
        reason_code="parse_inconclusive",
        severity_class="inconclusive",
        metadata=metadata,
    )
