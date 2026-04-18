"""Development-oriented verification helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .types import Check, severity_class_for_reason_code

_DEVELOPMENT_REPORT_SCORE_PATTERN = re.compile(
    r"score:\s*(\d+)\s*/\s*(\d+)\s+tests\s+passed",
    re.IGNORECASE,
)
_DETAIL_REASON_CODE_PATTERN = re.compile(
    r"\breason_code\b\s*[:=]\s*([a-z0-9_]+)",
    re.IGNORECASE,
)
_DETAIL_CAPABILITY_PATTERN = re.compile(
    r"\bcapability\b\s*[:=]\s*([a-z0-9_]+)",
    re.IGNORECASE,
)

_DEVELOPMENT_EXECUTOR_CONSTRAINTS = """
- For development verification, prefer deterministic checks before browser/UI probes.
- Prefer the `verification_helper` tool for structured build/test/report
  verification steps when it fits (`run_build_check`, `run_test_suite`,
  `render_verification_report`).
- For richer localhost browser verification flows with clicks, fills, console
  capture, network capture, or screenshots, prefer a structured
  `verification_helper` browser flow such as `browser_session` when available.
- Use `shell_execute` for general commands or when no registered verification
  helper matches the task.
- If a local service probe is required, use one bounded command with explicit
  timeout and cleanup behavior.
- Avoid shell background-process chains like `python -m http.server & ...`
  unless cleanup is explicit and necessary.
- Write machine-readable validation artifacts first. If you also write markdown
  validation reports, derive them from the structured result.
""".strip()

_DEVELOPMENT_VERIFIER_CONSTRAINTS = """
- Distinguish product failures from verifier/harness failures.
- Treat unavailable browser/service/runtime capabilities as infra unless the
  task explicitly requires that capability.
- When a matching helper is registered, prefer the `verification_helper` tool
  over raw shell/browser snippets so build/test/report/browser outcomes stay structured.
- Prefer behavior-based checks over source-style heuristics when both are available.
- When runtime validation artifacts exist, treat the machine-readable result as
  canonical and evaluate markdown reports against it.
""".strip()

_DEVELOPMENT_HELPER_TOOL_GUIDANCE = (
    "For development-oriented verification, prefer `verification_helper` for "
    "structured build/test/report checks. When available, prefer helpers like "
    "`browser_session` for richer localhost browser verification. Use "
    "`shell_execute` for general commands or flows without a matching helper."
)

_DEVELOPMENT_BUILD_REQUIRED_TOOL_ORDER = (
    "search_files",
    "read_file",
    "write_file",
    "verification_helper",
    "shell_execute",
    "ripgrep_search",
    "document_write",
)

_DEVELOPMENT_BUILD_RECOMMENDED_TOOL_ORDER = (
    "verification_helper",
    "shell_execute",
    "ripgrep_search",
    "web_search",
)


@dataclass(frozen=True)
class ToolFailureDisposition:
    """Normalized handling for one failed tool invocation."""

    advisory: bool
    detail: str
    reason_code: str = ""
    capability: str = ""


def extract_shell_execute_command(tool_args: object) -> str:
    """Return the shell command string from structured tool args."""
    if not isinstance(tool_args, dict):
        return ""
    return str(tool_args.get("command", "") or "").strip()


def extract_claude_code_prompt(tool_args: object) -> str:
    """Return the Claude Code prompt text from structured tool args."""
    if not isinstance(tool_args, dict):
        return ""
    return str(tool_args.get("prompt", "") or "").strip()


def looks_like_development_runtime_probe(command: str) -> bool:
    """Return True when a shell command looks like a local runtime probe."""
    text = str(command or "").strip().lower()
    if not text:
        return False
    runtime_markers = (
        "http.server",
        "localhost:",
        "127.0.0.1:",
        "curl ",
        "wget ",
        "playwright",
        "puppeteer",
        "headless",
        "screenshot",
    )
    return any(marker in text for marker in runtime_markers)


def development_runtime_probe_capability(command: str) -> str:
    """Infer which optional capability a runtime probe depends on."""
    text = str(command or "").strip().lower()
    if not text:
        return ""
    browser_markers = ("playwright", "puppeteer", "headless", "screenshot")
    if any(marker in text for marker in browser_markers):
        return "browser_runtime"
    service_markers = ("http.server", "localhost:", "127.0.0.1:", "curl ", "wget ")
    if any(marker in text for marker in service_markers):
        return "service_runtime"
    return ""


def looks_like_development_browser_probe(prompt: str) -> bool:
    """Return True when a prompt looks like browser-based verification."""
    text = str(prompt or "").strip().lower()
    if not text:
        return False
    browser_markers = (
        "playwright",
        "puppeteer",
        "headless browser",
        "browser tests",
        "capture console logs",
        "network requests",
        "take a screenshot",
    )
    return any(marker in text for marker in browser_markers)


def format_development_failure_detail(
    *,
    reason_code: str = "",
    capability: str = "",
    detail: str = "",
) -> str:
    """Render structured development failure metadata into check detail text."""
    parts: list[str] = []
    if reason_code:
        parts.append(f"reason_code={reason_code}")
    if capability:
        parts.append(f"capability={capability}")
    if detail:
        parts.append(detail)
    return "; ".join(parts)


def development_verifier_reason_code(error: str) -> str:
    """Map verifier harness failures to structured development reason codes."""
    text = str(error or "").strip().lower()
    if any(marker in text for marker in ("timed out", "timeout")):
        return "dev_verifier_timeout"
    capability_markers = (
        "command not found",
        "not installed",
        "no module named",
        "failed to launch",
        "browser not available",
        "address already in use",
        "unable to bind",
        "connection refused",
    )
    if any(marker in text for marker in capability_markers):
        return "dev_verifier_capability_unavailable"
    return ""


def development_product_reason_code(command: str) -> str:
    """Map product-facing build/test/check failures to structured reason codes."""
    text = str(command or "").strip().lower()
    if not text:
        return ""
    test_markers = (
        "pytest",
        "unittest",
        "npm test",
        "pnpm test",
        "yarn test",
        "vitest",
        "jest",
        "go test",
        "cargo test",
        "rspec",
        "node test",
        "test-runtime.js",
    )
    if any(marker in text for marker in test_markers):
        return "dev_test_failed"

    build_markers = (
        "npm run build",
        "pnpm build",
        "yarn build",
        "vite build",
        "webpack",
        "parcel build",
        "sphinx-build",
        "cargo build",
        "make build",
    )
    if any(marker in text for marker in build_markers):
        return "dev_build_failed"

    contract_markers = (
        "ruff",
        "mypy",
        "eslint",
        "stylelint",
        "node --check",
        "tsc",
        "py_compile",
    )
    if any(marker in text for marker in contract_markers):
        return "dev_contract_failed"
    return ""


def development_report_consistency_checks(
    *,
    workspace: Path,
    changed_paths: set[str],
) -> list[Check]:
    """Return advisory checks when structured/runtime validation artifacts disagree."""
    if "runtime-validation-results.json" not in changed_paths:
        return []

    runtime_results_path = workspace / "runtime-validation-results.json"
    if not runtime_results_path.exists():
        return []

    report_candidates = [
        workspace / rel
        for rel in changed_paths
        if rel.endswith(".md") and "validation-report" in rel
    ]
    if not report_candidates:
        report_candidates = sorted(workspace.glob("*validation-report*.md"))
    if not report_candidates:
        return []

    runtime_score = parse_runtime_validation_score(runtime_results_path)
    if runtime_score is None:
        return []
    checks: list[Check] = []
    runtime_passed, runtime_total = runtime_score
    for report_path in report_candidates:
        report_score = parse_validation_report_score(report_path)
        if report_score is None:
            continue
        if report_score != runtime_score:
            checks.append(Check(
                name=f"dev_report_consistency_{report_path.name}_advisory",
                passed=True,
                detail=format_development_failure_detail(
                    reason_code="dev_report_contract_violation",
                    capability="report_rendering",
                    detail=(
                        f"Structured runtime results report {runtime_passed}/{runtime_total} "
                        f"while {report_path.name} reports {report_score[0]}/{report_score[1]}."
                    ),
                ),
            ))
    return checks


def development_validation_artifact_snapshot(
    *,
    workspace: Path,
    changed_paths: set[str],
) -> dict[str, object]:
    """Capture canonical development verification artifacts for downstream consumers."""
    snapshot: dict[str, object] = {
        "canonical_result_path": "",
        "canonical_result": {},
        "reports": [],
        "report_mismatch_count": 0,
        "report_paths": [],
    }
    runtime_results_path = workspace / "runtime-validation-results.json"
    if not runtime_results_path.exists():
        return snapshot

    runtime_score = parse_runtime_validation_score(runtime_results_path)
    if runtime_score is None:
        return snapshot

    runtime_passed, runtime_total = runtime_score
    snapshot["canonical_result_path"] = "runtime-validation-results.json"
    snapshot["canonical_result"] = {
        "passed": runtime_passed,
        "total": runtime_total,
        "failed": max(0, runtime_total - runtime_passed),
    }

    report_candidates = [
        workspace / rel
        for rel in changed_paths
        if rel.endswith(".md") and "validation-report" in rel
    ]
    if not report_candidates:
        report_candidates = sorted(workspace.glob("*validation-report*.md"))

    reports: list[dict[str, object]] = []
    mismatch_count = 0
    for report_path in report_candidates:
        report_score = parse_validation_report_score(report_path)
        relative_path = str(report_path.relative_to(workspace))
        report_payload: dict[str, object] = {
            "path": relative_path,
            "matches_canonical": report_score == runtime_score,
        }
        if report_score is not None:
            report_payload["passed"] = report_score[0]
            report_payload["total"] = report_score[1]
        if report_score != runtime_score:
            mismatch_count += 1
        reports.append(report_payload)

    snapshot["reports"] = reports
    snapshot["report_mismatch_count"] = mismatch_count
    snapshot["report_paths"] = [str(item.get("path", "") or "") for item in reports]
    return snapshot


def parse_runtime_validation_score(path: Path) -> tuple[int, int] | None:
    """Read the structured runtime validation score from JSON results."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    try:
        passed = int(payload.get("passed", 0) or 0)
        failed = int(payload.get("failed", 0) or 0)
    except (TypeError, ValueError):
        return None
    total = passed + failed
    if total <= 0:
        tests = payload.get("tests", [])
        if isinstance(tests, list):
            total = len(tests)
    if total <= 0:
        return None
    return passed, total


def parse_validation_report_score(path: Path) -> tuple[int, int] | None:
    """Read the summary score from a generated validation markdown report."""
    try:
        text = path.read_text()
    except OSError:
        return None
    match = _DEVELOPMENT_REPORT_SCORE_PATTERN.search(text)
    if not match:
        return None
    try:
        return int(match.group(1)), int(match.group(2))
    except (TypeError, ValueError):
        return None


def extract_reason_code(detail: str | None) -> str:
    """Extract a structured reason code token from free-form check detail text."""
    text = str(detail or "").strip()
    if not text:
        return ""
    match = _DETAIL_REASON_CODE_PATTERN.search(text)
    if not match:
        return ""
    return str(match.group(1) or "").strip().lower()


def extract_capability(detail: str | None) -> str:
    """Extract a structured optional capability token from check detail text."""
    text = str(detail or "").strip()
    if not text:
        return ""
    match = _DETAIL_CAPABILITY_PATTERN.search(text)
    if not match:
        return ""
    return str(match.group(1) or "").strip().lower()


def build_development_verification_summary(checks: list[Check]) -> dict[str, object]:
    """Summarize development-oriented verification outcomes for metadata/reporting."""
    summary: dict[str, object] = {
        "advisory_failure_count": 0,
        "blocking_failure_count": 0,
        "product_failure_count": 0,
        "infra_failure_count": 0,
        "inconclusive_failure_count": 0,
        "report_mismatch_warning_count": 0,
        "optional_failure_capabilities": [],
        "capability_counts": {},
        "reason_counts": {},
        "advisory_failures": [],
        "blocking_failures": [],
    }
    reason_counts: dict[str, int] = {}
    capability_counts: dict[str, int] = {}
    advisory_failures: list[dict[str, str]] = []
    blocking_failures: list[dict[str, str]] = []

    for check in checks:
        name = str(getattr(check, "name", "") or "").strip().lower()
        passed = bool(getattr(check, "passed", False))
        detail = str(getattr(check, "detail", "") or "").strip()
        reason_code = extract_reason_code(detail)
        capability = extract_capability(detail)
        severity = severity_class_for_reason_code(reason_code) or "semantic"
        entry: dict[str, str] = {"check_name": name}
        if reason_code:
            entry["reason_code"] = reason_code
        if capability:
            entry["capability"] = capability
        if detail:
            entry["detail"] = detail
        is_advisory = passed and bool(
            (name.startswith("tool_") and name.endswith("_advisory"))
            or name.startswith("dev_report_consistency_")
            or reason_code
            or capability
        )
        if is_advisory:
            summary["advisory_failure_count"] = int(summary["advisory_failure_count"]) + 1
            if severity == "infra":
                summary["infra_failure_count"] = int(summary["infra_failure_count"]) + 1
            elif severity == "inconclusive":
                summary["inconclusive_failure_count"] = (
                    int(summary["inconclusive_failure_count"]) + 1
                )
            else:
                summary["product_failure_count"] = int(summary["product_failure_count"]) + 1
            advisory_failures.append(entry)
        elif not passed:
            summary["blocking_failure_count"] = int(summary["blocking_failure_count"]) + 1
            if severity == "infra":
                summary["infra_failure_count"] = int(summary["infra_failure_count"]) + 1
            elif severity == "inconclusive":
                summary["inconclusive_failure_count"] = (
                    int(summary["inconclusive_failure_count"]) + 1
                )
            else:
                summary["product_failure_count"] = int(summary["product_failure_count"]) + 1
            blocking_failures.append(entry)

        if name.startswith("dev_report_consistency_"):
            summary["report_mismatch_warning_count"] = (
                int(summary["report_mismatch_warning_count"]) + 1
            )

        if reason_code:
            reason_counts[reason_code] = reason_counts.get(reason_code, 0) + 1
        if capability:
            capability_counts[capability] = capability_counts.get(capability, 0) + 1

    summary["reason_counts"] = reason_counts
    summary["capability_counts"] = capability_counts
    summary["advisory_failures"] = advisory_failures
    summary["blocking_failures"] = blocking_failures
    summary["optional_failure_capabilities"] = sorted(capability_counts.keys())
    summary["has_optional_verifier_warnings"] = bool(summary["advisory_failure_count"])
    summary["has_report_mismatch_warning"] = bool(summary["report_mismatch_warning_count"])
    return summary


def merge_development_validation_snapshot(
    summary: dict[str, object],
    *,
    artifact_snapshot: object,
) -> dict[str, object]:
    """Attach canonical artifact metadata to a development verification summary."""
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(artifact_snapshot, dict) or not artifact_snapshot:
        return summary
    merged = dict(summary)
    canonical_result_path = str(
        artifact_snapshot.get("canonical_result_path", "") or "",
    ).strip()
    if canonical_result_path:
        merged["canonical_result_path"] = canonical_result_path
    canonical_result = artifact_snapshot.get("canonical_result", {})
    if isinstance(canonical_result, dict) and canonical_result:
        merged["canonical_result"] = dict(canonical_result)
    reports = artifact_snapshot.get("reports", [])
    if isinstance(reports, list):
        merged["report_artifacts"] = [
            dict(item) for item in reports if isinstance(item, dict)
        ]
    report_paths = artifact_snapshot.get("report_paths", [])
    if isinstance(report_paths, list):
        merged["report_paths"] = [
            str(item or "").strip()
            for item in report_paths
            if str(item or "").strip()
        ]
    try:
        merged["report_mismatch_count"] = int(
            artifact_snapshot.get(
                "report_mismatch_count",
                merged.get("report_mismatch_warning_count", 0),
            ) or 0,
        )
    except (TypeError, ValueError):
        pass
    return merged


def event_safe_development_summary(summary: object) -> dict[str, object]:
    """Return a compact, event-safe development verification summary payload."""
    if not isinstance(summary, dict):
        return {}
    event_summary: dict[str, object] = {}
    for key in (
        "advisory_failure_count",
        "blocking_failure_count",
        "product_failure_count",
        "infra_failure_count",
        "inconclusive_failure_count",
        "report_mismatch_warning_count",
        "report_mismatch_count",
        "has_optional_verifier_warnings",
        "has_report_mismatch_warning",
        "canonical_result_path",
        "report_paths",
        "optional_failure_capabilities",
        "capability_counts",
        "reason_counts",
        "canonical_result",
    ):
        value = summary.get(key)
        if value in (None, "", [], {}):
            continue
        event_summary[key] = value
    return event_summary


def development_executor_constraints() -> str:
    """Return standardized executor constraints for development-balanced flows."""
    return _DEVELOPMENT_EXECUTOR_CONSTRAINTS


def development_verifier_constraints() -> str:
    """Return standardized verifier constraints for development-balanced flows."""
    return _DEVELOPMENT_VERIFIER_CONSTRAINTS


def development_helper_tool_guidance() -> str:
    """Return shared guidance for preferring structured development helpers."""
    return _DEVELOPMENT_HELPER_TOOL_GUIDANCE


def preferred_development_build_tools(available_tools: list[str] | tuple[str, ...]) -> list[str]:
    """Return prioritized build-tool defaults for development workflows."""
    available = {
        str(item or "").strip()
        for item in available_tools
        if str(item or "").strip()
    }
    return [
        name
        for name in _DEVELOPMENT_BUILD_REQUIRED_TOOL_ORDER
        if name in available
    ][:5]


def recommended_development_build_tools(
    available_tools: list[str] | tuple[str, ...],
) -> list[str]:
    """Return missing tool suggestions for development workflows."""
    available = {
        str(item or "").strip()
        for item in available_tools
        if str(item or "").strip()
    }
    return [
        name
        for name in _DEVELOPMENT_BUILD_RECOMMENDED_TOOL_ORDER
        if name not in available
    ]


def optional_failure_capability_for_reason(
    summary: object,
    *,
    reason_code: str = "",
) -> str:
    """Return the optional verifier capability associated with a summary reason."""
    if not isinstance(summary, dict):
        return ""
    normalized_reason = str(reason_code or "").strip().lower()
    advisory_failures = summary.get("advisory_failures", [])
    if isinstance(advisory_failures, list):
        for item in advisory_failures:
            if not isinstance(item, dict):
                continue
            item_reason = str(item.get("reason_code", "") or "").strip().lower()
            if normalized_reason and item_reason != normalized_reason:
                continue
            capability = str(item.get("capability", "") or "").strip().lower()
            if capability:
                return capability
    capabilities = summary.get("optional_failure_capabilities", [])
    if not isinstance(capabilities, list) or not capabilities:
        return ""
    return str(capabilities[0] or "").strip().lower()
