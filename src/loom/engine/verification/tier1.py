"""Tier-1 deterministic verification checks."""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from loom.processes.phase_alignment import infer_phase_id_for_subtask
from loom.state.task_state import Subtask

from .development import (
    ToolFailureDisposition,
    build_development_verification_summary,
    development_product_reason_code,
    development_report_consistency_checks,
    development_runtime_probe_capability,
    development_validation_artifact_snapshot,
    development_verifier_reason_code,
    extract_claude_code_prompt,
    extract_shell_execute_command,
    format_development_failure_detail,
    looks_like_development_browser_probe,
    looks_like_development_runtime_probe,
    merge_development_validation_snapshot,
)
from .types import Check, VerificationResult, severity_class_for_reason_code

if TYPE_CHECKING:
    from loom.processes.schema import ProcessDefinition

logger = logging.getLogger(__name__)

_DEVELOPMENT_HELPER_CAPABILITIES = {
    "run_test_suite": "command_execution",
    "run_build_check": "command_execution",
    "serve_static": "service_runtime",
    "http_assert": "service_runtime",
    "browser_assert": "browser_runtime",
    "render_verification_report": "report_rendering",
}
_DEVELOPMENT_HELPER_REASON_CODES = frozenset({
    "dev_verifier_timeout",
    "dev_verifier_capability_unavailable",
    "dev_test_failed",
    "dev_build_failed",
    "dev_contract_failed",
    "dev_browser_check_failed",
    "dev_report_contract_violation",
})

_HTTP_STATUS_PATTERN = re.compile(
    r"\b(?:http|status(?:\s*code)?)\s*[:=]?\s*([1-5]\d{2})\b",
    re.IGNORECASE,
)


def _extract_http_status(error_text: str) -> int | None:
    """Extract an HTTP status code from free-form error text when present."""
    match = _HTTP_STATUS_PATTERN.search(error_text or "")
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


class DeterministicVerifier:
    """Tier 1: Zero-cost checks that don't require any LLM invocation.

    Checks:
    - Did tool calls succeed?
    - Were expected files created/modified?
    - Are output files non-empty?
    - Is file syntax valid (Python, JSON, YAML)?
    - Process regex rules (if a process definition is loaded)
    - Process deliverables (if phases define expected deliverables)
    """

    _ADVISORY_TOOL_FAILURES = frozenset({"web_fetch", "web_fetch_html", "web_search"})
    _TOOL_SUCCESS_POLICIES = frozenset({
        "all_tools_hard",
        "development_balanced",
        "safety_integrity_only",
    })
    _PLACEHOLDER_RULE_HEURISTIC_MARKERS = (
        "placeholder",
        "todo",
        "tbd",
        "missing",
        "insert",
        "n/a",
        r"\bn/?a\b",
    )

    def __init__(
        self,
        process: ProcessDefinition | None = None,
        *,
        phase_scope_default: str = "current_phase",
        regex_default_advisory: bool = True,
    ):
        self._process = process
        self._tool_success_policy = self._resolve_tool_success_policy(process)
        normalized_scope = str(phase_scope_default or "current_phase").strip().lower()
        self._phase_scope_default = (
            normalized_scope if normalized_scope in {"current_phase", "global"}
            else "current_phase"
        )
        self._regex_default_advisory = bool(regex_default_advisory)

    @classmethod
    def _resolve_tool_success_policy(
        cls,
        process: ProcessDefinition | None,
    ) -> str:
        if process is None:
            return "all_tools_hard"
        resolver = getattr(process, "verifier_tool_success_policy", None)
        if callable(resolver):
            raw = str(
                resolver(include_adhoc_fallback=True) or "",
            ).strip().lower()
            if raw in cls._TOOL_SUCCESS_POLICIES:
                return raw
        return "all_tools_hard"

    async def verify(
        self,
        subtask: Subtask,
        result_summary: str,
        tool_calls: list,
        workspace: Path | None = None,
        evidence_tool_calls: list | None = None,
        evidence_records: list[dict] | None = None,
        retry_writable_deliverables: list[str] | None = None,
    ) -> VerificationResult:
        checks: list[Check] = []
        expected_deliverables: list[str] = []
        recoverable_placeholder_check_names: set[str] = set()
        placeholder_findings: list[dict[str, object]] = []
        expected_deliverable_paths: set[str] = set()
        retry_writable_paths: set[str] = set()
        changed_paths: set[str] = set()
        development_artifact_snapshot: dict[str, object] = {}

        if self._process:
            expected_deliverables = self._expected_deliverables_for_subtask(subtask)
        if workspace and expected_deliverables:
            for expected in expected_deliverables:
                normalized = self._normalize_workspace_relpath(workspace, expected)
                if normalized:
                    expected_deliverable_paths.add(normalized)
        if workspace and retry_writable_deliverables:
            for expected in retry_writable_deliverables:
                normalized = self._normalize_workspace_relpath(workspace, expected)
                if normalized:
                    retry_writable_paths.add(normalized)
        syntax_enforcement_paths = retry_writable_paths or expected_deliverable_paths

        # 1. Did tool calls succeed?
        for tc in tool_calls:
            if not tc.result.success:
                disposition = self._classify_tool_failure(
                    subtask=subtask,
                    tool_name=tc.tool,
                    tool_args=getattr(tc, "args", {}),
                    tool_result_data=getattr(tc.result, "data", None),
                    error=tc.result.error,
                )
                if disposition.advisory:
                    checks.append(Check(
                        name=f"tool_{tc.tool}_advisory",
                        passed=True,
                        detail=(
                            "Advisory tool failure (non-blocking): "
                            f"{disposition.detail or tc.result.error or 'unknown error'}"
                        ),
                    ))
                    continue
                checks.append(Check(
                    name=f"tool_{tc.tool}_success",
                    passed=False,
                    detail=disposition.detail or tc.result.error,
                ))
                continue

            checks.append(Check(
                name=f"tool_{tc.tool}_success",
                passed=True,
            ))

        # 2. Were expected files created/modified? Check non-empty.
        if workspace:
            for tc in tool_calls:
                for f in tc.result.files_changed:
                    normalized = self._normalize_workspace_relpath(workspace, str(f or ""))
                    if normalized:
                        changed_paths.add(normalized)
                    fpath = workspace / f
                    if fpath.exists():
                        fsize = fpath.stat().st_size
                        checks.append(Check(
                            name=f"file_nonempty_{f}",
                            passed=fsize > 0,
                            detail=f"File '{f}' is empty" if fsize == 0 else None,
                        ))

        # 3. Syntax checks for known file types
        if workspace:
            for tc in tool_calls:
                for f in tc.result.files_changed:
                    fpath = workspace / f
                    if fpath.exists():
                        syntax_check = await self._check_syntax(fpath)
                        if syntax_check:
                            normalized = self._normalize_workspace_relpath(
                                workspace,
                                str(f or ""),
                            )
                            in_retry_writable_scope = (
                                not syntax_enforcement_paths
                                or (
                                    bool(normalized)
                                    and normalized in syntax_enforcement_paths
                                )
                            )
                            if not in_retry_writable_scope:
                                if not syntax_check.passed:
                                    checks.append(Check(
                                        name=f"syntax_{fpath.name}_advisory",
                                        passed=True,
                                        detail=(
                                            "Advisory: non-canonical artifact "
                                            f"'{normalized or str(f)}' has syntax issues "
                                            f"({syntax_check.detail or 'syntax error'}). "
                                            "Hard-fail checks are limited to retry-writable "
                                            "deliverables."
                                        ),
                                    ))
                                continue
                            checks.append(syntax_check)

            if self._tool_success_policy == "development_balanced":
                development_artifact_snapshot = development_validation_artifact_snapshot(
                    workspace=workspace,
                    changed_paths=changed_paths,
                )
                checks.extend(
                    development_report_consistency_checks(
                        workspace=workspace,
                        changed_paths=changed_paths,
                    ),
                )

        # 4. Process regex rules
        if self._process:
            for rule in self._process.regex_rules_for_subtask(
                subtask.id,
                phase_scope_default=self._phase_scope_default,
            ):
                target_text = result_summary
                if rule.target == "deliverables" and workspace:
                    # Check against deliverable file contents
                    target_text = self._read_deliverable_text(
                        workspace,
                        tool_calls,
                        expected_deliverables=expected_deliverables,
                    )
                try:
                    found = bool(re.search(rule.check, target_text))
                except re.error:
                    found = False
                # For regex rules, a match means the pattern was found.
                # If severity is "error", finding the pattern = failure
                # (e.g., finding [TBD] is bad).
                # Convention: the check description says what to look for
                # as a violation, so finding it = failure.
                hard_enforced = self._is_regex_rule_hard(rule)
                is_failure = found and hard_enforced
                check_name = f"process_rule_{rule.name}"
                checks.append(Check(
                    name=check_name,
                    passed=not is_failure,
                    detail=(
                        f"Rule '{rule.name}' matched (hard): {rule.description}"
                        if found and hard_enforced
                        else (
                            f"Rule '{rule.name}' matched (advisory): "
                            f"{rule.description}"
                            if found else None
                        )
                    ),
                ))
                if is_failure and self._is_recoverable_placeholder_regex_rule(rule):
                    recoverable_placeholder_check_names.add(check_name)
                    placeholder_findings.extend(
                        self._extract_placeholder_findings_for_regex_rule(
                            rule=rule,
                            result_summary=result_summary,
                            tool_calls=tool_calls,
                            workspace=workspace,
                            expected_deliverables=expected_deliverables,
                        ),
                    )

            # 5. Process deliverables check
            # Enforce only deliverables for the matching phase/subtask id.
            if expected_deliverables and workspace:
                files_created = set()
                for tc in tool_calls:
                    files_created.update(tc.result.files_changed)
                for expected_path in expected_deliverables:
                    found_file = (
                        expected_path in files_created
                        or (workspace / expected_path).exists()
                    )
                    checks.append(Check(
                        name=f"deliverable_{expected_path}",
                        passed=found_file,
                        detail=(
                            f"Expected deliverable '{expected_path}' "
                            f"not found"
                            if not found_file else None
                        ),
                    ))
            checks.extend(
                self._synthesis_input_integrity_checks(
                    subtask=subtask,
                    result_summary=result_summary,
                    tool_calls=tool_calls,
                    workspace=workspace,
                ),
            )

        # NOTE: Tier 1 remains process-agnostic by design. Semantic, domain-
        # specific quality checks are handled by Tier 2/3 LLM verification.
        # This prevents hard-coding process-specific schemas into core static
        # verification and keeps modular process packages decoupled.

        all_failures = [c for c in checks if not c.passed]
        recoverable_placeholder_failures = [
            c for c in all_failures if c.name in recoverable_placeholder_check_names
        ]
        hard_failures = [
            c for c in all_failures if c.name not in recoverable_placeholder_check_names
        ]
        advisory_hits = [
            c for c in checks
            if c.passed and c.detail and (
                c.name.startswith("process_rule_")
                or "_advisory" in c.name
                or str(c.detail).lower().startswith("advisory")
            )
        ]
        placeholder_findings = self._dedupe_placeholder_findings(placeholder_findings)
        if recoverable_placeholder_failures and not placeholder_findings:
            placeholder_findings = self._fallback_placeholder_findings_from_checks(
                recoverable_placeholder_failures,
            )

        all_passed = not all_failures
        outcome = "pass"
        if all_failures:
            outcome = "fail"
        elif advisory_hits:
            outcome = "pass_with_warnings"
        reason_code = ""
        severity_class = "semantic"
        feedback = self._build_advisory_feedback(advisory_hits)
        if hard_failures:
            reason_code = self._hard_failure_reason_code(hard_failures)
            severity_class = (
                "hard_invariant"
                if reason_code == "hard_invariant_failed"
                else "semantic"
            )
            feedback = self._build_feedback(hard_failures)
        elif recoverable_placeholder_failures:
            reason_code = "incomplete_deliverable_placeholder"
            severity_class = "semantic"
            feedback = (
                f"{self._build_feedback(recoverable_placeholder_failures)}\n"
                "Recoverable placeholder findings require follow-up remediation."
            )
        metadata: dict[str, object] = {
            "hard_failure_count": len(hard_failures),
            "recoverable_placeholder_failure_count": len(
                recoverable_placeholder_failures,
            ),
            "advisory_count": len(advisory_hits),
        }
        if self._tool_success_policy == "development_balanced":
            metadata["dev_verification_summary"] = merge_development_validation_snapshot(
                build_development_verification_summary(checks),
                artifact_snapshot=development_artifact_snapshot,
            )
        if hard_failures and reason_code:
            metadata["hard_failure_reason_code"] = reason_code
        if placeholder_findings:
            metadata["placeholder_findings"] = placeholder_findings
            metadata["placeholder_finding_count"] = len(placeholder_findings)
        if recoverable_placeholder_failures:
            metadata["remediation_required"] = True
            metadata["remediation_mode"] = "confirm_or_prune"
            metadata["failure_class"] = "recoverable_placeholder"
            metadata["candidate_fill_sources"] = []
            metadata["missing_targets"] = self._placeholder_missing_targets(
                placeholder_findings,
            )
        return VerificationResult(
            tier=1,
            passed=all_passed,
            checks=checks,
            feedback=feedback,
            outcome=outcome,
            reason_code=reason_code,
            severity_class=severity_class,
            metadata=metadata,
        )

    def _classify_tool_failure(
        self,
        *,
        subtask: Subtask,
        tool_name: str,
        tool_args: object,
        tool_result_data: object = None,
        error: str | None,
    ) -> ToolFailureDisposition:
        detail = str(error or "").strip()
        if self._tool_success_policy == "development_balanced":
            disposition = self._classify_development_tool_failure(
                subtask=subtask,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result_data=tool_result_data,
                error=detail,
            )
            if disposition is not None:
                return disposition
        if self._is_advisory_tool_failure(tool_name, detail):
            return ToolFailureDisposition(advisory=True, detail=detail)
        return ToolFailureDisposition(advisory=False, detail=detail)

    def _classify_development_tool_failure(
        self,
        *,
        subtask: Subtask,
        tool_name: str,
        tool_args: object,
        tool_result_data: object,
        error: str,
    ) -> ToolFailureDisposition | None:
        if not error:
            return None
        if self._is_hard_safety_or_integrity_failure(error):
            return ToolFailureDisposition(advisory=False, detail=error)

        if tool_name == "shell_execute":
            command = extract_shell_execute_command(tool_args)
            if looks_like_development_runtime_probe(command):
                reason_code = development_verifier_reason_code(error)
                capability = development_runtime_probe_capability(command)
                if reason_code:
                    return ToolFailureDisposition(
                        advisory=True,
                        detail=format_development_failure_detail(
                            reason_code=reason_code,
                            capability=capability,
                            detail=error,
                        ),
                        reason_code=reason_code,
                        capability=capability,
                    )
            product_reason = development_product_reason_code(command)
            if product_reason:
                return ToolFailureDisposition(
                    advisory=False,
                    detail=format_development_failure_detail(
                        reason_code=product_reason,
                        detail=error,
                    ),
                    reason_code=product_reason,
                )

        if tool_name == "claude_code":
            prompt = extract_claude_code_prompt(tool_args)
            if looks_like_development_browser_probe(prompt):
                reason_code = development_verifier_reason_code(error)
                if reason_code:
                    return ToolFailureDisposition(
                        advisory=True,
                        detail=format_development_failure_detail(
                            reason_code=reason_code,
                            capability="browser_runtime",
                            detail=error,
                        ),
                        reason_code=reason_code,
                        capability="browser_runtime",
                    )

        if tool_name == "verification_helper" and isinstance(tool_args, dict):
            helper_name = str(tool_args.get("helper", "") or "").strip().lower()
            normalized_error = str(error or "").strip().lower()
            result_data = tool_result_data if isinstance(tool_result_data, dict) else {}
            if not normalized_error:
                normalized_error = str(
                    result_data.get("helper_reason_code", "") or "",
                ).strip().lower()
            capability = str(result_data.get("helper_capability", "") or "").strip()
            if not capability:
                capability = _DEVELOPMENT_HELPER_CAPABILITIES.get(helper_name, "")
            if capability and normalized_error in _DEVELOPMENT_HELPER_REASON_CODES:
                severity_class = severity_class_for_reason_code(normalized_error)
                return ToolFailureDisposition(
                    advisory=severity_class != "semantic",
                    detail=format_development_failure_detail(
                        reason_code=normalized_error,
                        capability=capability,
                        detail=error,
                    ),
                    reason_code=normalized_error,
                    capability=capability,
                )

        return None

    @classmethod
    def _hard_failure_reason_code(cls, hard_failures: list[Check]) -> str:
        for check in hard_failures:
            detail = str(getattr(check, "detail", "") or "").strip()
            if not detail:
                continue
            match = re.search(
                r"\breason_code\b\s*[:=]\s*([a-z0-9_]+)",
                detail,
                flags=re.IGNORECASE,
            )
            if match:
                reason = str(match.group(1) or "").strip().lower()
                if reason:
                    return reason
        return "hard_invariant_failed"

    def _is_regex_rule_hard(self, rule) -> bool:
        enforcement = str(getattr(rule, "enforcement", "") or "").strip().lower()
        if enforcement == "hard":
            return True
        if enforcement == "advisory":
            return False
        if self._regex_default_advisory:
            return False
        return str(getattr(rule, "severity", "warning")).strip().lower() == "error"

    def _is_recoverable_placeholder_regex_rule(self, rule) -> bool:
        failure_class = str(getattr(rule, "failure_class", "") or "").strip().lower()
        if failure_class == "recoverable_placeholder":
            return True
        if failure_class in {"hard_integrity", "semantic"}:
            return False

        remediation_mode = str(
            getattr(rule, "remediation_mode", "") or "",
        ).strip().lower()
        if remediation_mode == "confirm_or_prune":
            return True
        if remediation_mode in {
            "none",
            "targeted_remediation",
            "queue_follow_up",
            "remediate_and_retry",
        }:
            return False

        target = str(getattr(rule, "target", "") or "").strip().lower()
        if target not in {"deliverables", "output"}:
            return False
        heuristic_haystack = " ".join([
            str(getattr(rule, "name", "") or ""),
            str(getattr(rule, "description", "") or ""),
            str(getattr(rule, "check", "") or ""),
        ]).lower()
        return any(
            marker in heuristic_haystack
            for marker in self._PLACEHOLDER_RULE_HEURISTIC_MARKERS
        )

    @staticmethod
    def _normalize_workspace_relpath(
        workspace: Path,
        raw_path: str,
    ) -> str | None:
        text = str(raw_path or "").strip()
        if not text:
            return None
        root = workspace.resolve(strict=False)
        candidate = Path(text)
        if candidate.is_absolute():
            resolved = candidate.resolve(strict=False)
        else:
            resolved = (root / candidate).resolve(strict=False)
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            return None
        rel_text = rel.as_posix().strip()
        if not rel_text or rel_text == ".":
            return None
        return rel_text

    @staticmethod
    def _extract_placeholder_matches_from_text(
        *,
        pattern: re.Pattern[str],
        text: str,
        max_items: int = 80,
    ) -> list[tuple[int, int, str, str]]:
        findings: list[tuple[int, int, str, str]] = []
        if not text:
            return findings
        for match in pattern.finditer(text):
            start = max(0, int(match.start()))
            end = max(start, int(match.end()))
            line = text.count("\n", 0, start) + 1
            line_start = text.rfind("\n", 0, start) + 1
            line_end = text.find("\n", end)
            if line_end < 0:
                line_end = len(text)
            column = max(1, start - line_start + 1)
            context = text[line_start:line_end].strip()
            if len(context) > 280:
                context = f"{context[:277]}..."
            token = str(match.group(0) or "").strip()
            findings.append((line, column, token, context))
            if len(findings) >= max_items:
                break
        return findings

    def _extract_placeholder_findings_for_regex_rule(
        self,
        *,
        rule,
        result_summary: str,
        tool_calls: list,
        workspace: Path | None,
        expected_deliverables: list[str],
        max_findings: int = 120,
    ) -> list[dict[str, object]]:
        pattern_text = str(getattr(rule, "check", "") or "")
        try:
            pattern = re.compile(pattern_text)
        except re.error:
            return []

        findings: list[dict[str, object]] = []
        rule_name = str(getattr(rule, "name", "") or "").strip()
        target = str(getattr(rule, "target", "output") or "").strip().lower() or "output"

        def add_finding(
            *,
            file_path: str,
            source: str,
            line: int,
            column: int,
            token: str,
            context: str,
        ) -> None:
            if len(findings) >= max_findings:
                return
            findings.append({
                "rule_name": rule_name,
                "pattern": pattern_text,
                "source": source,
                "file_path": file_path,
                "line": max(0, int(line)),
                "column": max(0, int(column)),
                "token": str(token or ""),
                "context": str(context or ""),
            })

        if target == "deliverables" and workspace is not None:
            candidates: list[str] = []
            seen_candidates: set[str] = set()
            for rel_path in expected_deliverables:
                normalized = self._normalize_workspace_relpath(workspace, rel_path)
                if not normalized or normalized in seen_candidates:
                    continue
                seen_candidates.add(normalized)
                candidates.append(normalized)
            if not candidates:
                for call in tool_calls:
                    result = getattr(call, "result", None)
                    changed = getattr(result, "files_changed", [])
                    if not isinstance(changed, list):
                        continue
                    for rel_path in changed:
                        normalized = self._normalize_workspace_relpath(
                            workspace,
                            str(rel_path or ""),
                        )
                        if not normalized or normalized in seen_candidates:
                            continue
                        seen_candidates.add(normalized)
                        candidates.append(normalized)

            for rel_path in candidates:
                file_path = workspace / rel_path
                try:
                    if not file_path.exists() or not file_path.is_file():
                        continue
                    if file_path.stat().st_size > 1_500_000:
                        continue
                    text = file_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for line, column, token, context in self._extract_placeholder_matches_from_text(
                    pattern=pattern,
                    text=text,
                    max_items=max_findings,
                ):
                    add_finding(
                        file_path=rel_path,
                        source="deliverable",
                        line=line,
                        column=column,
                        token=token,
                        context=context,
                    )
                    if len(findings) >= max_findings:
                        break
                if len(findings) >= max_findings:
                    break
        else:
            for line, column, token, context in self._extract_placeholder_matches_from_text(
                pattern=pattern,
                text=str(result_summary or ""),
                max_items=max_findings,
            ):
                add_finding(
                    file_path="<result_summary>",
                    source="result_summary",
                    line=line,
                    column=column,
                    token=token,
                    context=context,
                )
                if len(findings) >= max_findings:
                    break
        return findings

    @classmethod
    def _dedupe_placeholder_findings(
        cls,
        findings: list[dict[str, object]],
        *,
        max_items: int = 120,
    ) -> list[dict[str, object]]:
        deduped: list[dict[str, object]] = []
        seen: set[tuple[str, str, int, int, str]] = set()
        for raw in findings:
            if not isinstance(raw, dict):
                continue
            file_path = str(raw.get("file_path", "") or "").strip()
            rule_name = str(raw.get("rule_name", "") or "").strip()
            token = str(raw.get("token", "") or "").strip()
            try:
                line = int(raw.get("line", 0) or 0)
            except (TypeError, ValueError):
                line = 0
            try:
                column = int(raw.get("column", 0) or 0)
            except (TypeError, ValueError):
                column = 0
            key = (
                file_path,
                rule_name,
                max(0, line),
                max(0, column),
                token,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append({
                "rule_name": rule_name,
                "pattern": str(raw.get("pattern", "") or ""),
                "source": str(raw.get("source", "") or ""),
                "file_path": file_path,
                "line": max(0, line),
                "column": max(0, column),
                "token": token,
                "context": str(raw.get("context", "") or ""),
            })
            if len(deduped) >= max_items:
                break
        return deduped

    @staticmethod
    def _fallback_placeholder_findings_from_checks(
        checks: list[Check],
    ) -> list[dict[str, object]]:
        fallback: list[dict[str, object]] = []
        for check in checks:
            rule_name = str(check.name or "").removeprefix("process_rule_").strip()
            fallback.append({
                "rule_name": rule_name,
                "pattern": "",
                "source": "rule_match_fallback",
                "file_path": "",
                "line": 0,
                "column": 0,
                "token": "",
                "context": str(check.detail or ""),
            })
            if len(fallback) >= 24:
                break
        return fallback

    @staticmethod
    def _placeholder_missing_targets(findings: list[dict[str, object]]) -> list[str]:
        targets: list[str] = []
        seen: set[str] = set()
        for item in findings:
            if not isinstance(item, dict):
                continue
            file_path = str(item.get("file_path", "") or "").strip()
            rule_name = str(item.get("rule_name", "") or "").strip()
            try:
                line = int(item.get("line", 0) or 0)
            except (TypeError, ValueError):
                line = 0
            if file_path:
                target = f"{file_path}:{line}" if line > 0 else file_path
            elif rule_name:
                target = f"rule:{rule_name}"
            else:
                continue
            if target in seen:
                continue
            seen.add(target)
            targets.append(target)
            if len(targets) >= 40:
                break
        return targets

    async def _check_syntax(self, path: Path) -> Check | None:
        """Run syntax checks based on file extension."""
        ext = path.suffix

        if ext == ".py":
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
                compile(source, str(path), "exec")
                return Check(name=f"syntax_{path.name}", passed=True)
            except SyntaxError as e:
                return Check(
                    name=f"syntax_{path.name}",
                    passed=False,
                    detail=f"Python syntax error: {e}",
                )

        elif ext == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
                return Check(name=f"syntax_{path.name}", passed=True)
            except json.JSONDecodeError as e:
                return Check(
                    name=f"syntax_{path.name}",
                    passed=False,
                    detail=f"JSON syntax error: {e}",
                )

        elif ext in (".yaml", ".yml"):
            try:
                import yaml
                yaml.safe_load(path.read_text(encoding="utf-8"))
                return Check(name=f"syntax_{path.name}", passed=True)
            except Exception as e:
                return Check(
                    name=f"syntax_{path.name}",
                    passed=False,
                    detail=f"YAML syntax error: {e}",
                )

        elif ext == ".csv":
            try:
                with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                    reader = csv.reader(handle)
                    expected_columns: int | None = None
                    for row_number, row in enumerate(reader, start=1):
                        if not row:
                            continue
                        if expected_columns is None:
                            expected_columns = len(row)
                            continue
                        actual_columns = len(row)
                        if actual_columns != expected_columns:
                            return Check(
                                name=f"syntax_{path.name}",
                                passed=False,
                                detail=(
                                    "reason_code=csv_schema_mismatch; "
                                    f"CSV row {row_number} has {actual_columns} columns "
                                    f"(expected {expected_columns})."
                                ),
                            )
                return Check(name=f"syntax_{path.name}", passed=True)
            except Exception as e:
                return Check(
                    name=f"syntax_{path.name}",
                    passed=False,
                    detail=f"CSV syntax error: {e}",
                )

        return None

    @staticmethod
    def _read_deliverable_text(
        workspace: Path,
        tool_calls: list,
        *,
        expected_deliverables: list[str] | None = None,
    ) -> str:
        """Read text content from all deliverable files."""
        texts = []
        candidate_paths: list[str] = []
        seen: set[str] = set()

        for expected in expected_deliverables or []:
            normalized = DeterministicVerifier._normalize_workspace_relpath(workspace, expected)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            candidate_paths.append(normalized)

        if not candidate_paths:
            for tc in tool_calls:
                for f in tc.result.files_changed:
                    normalized = DeterministicVerifier._normalize_workspace_relpath(
                        workspace,
                        str(f or ""),
                    )
                    if not normalized or normalized in seen:
                        continue
                    seen.add(normalized)
                    candidate_paths.append(normalized)

        for relpath in candidate_paths:
            fpath = workspace / relpath
            if not fpath.exists() or fpath.stat().st_size >= 1024 * 1024:
                continue
            try:
                texts.append(
                    fpath.read_text(encoding="utf-8", errors="replace"),
                )
            except Exception as e:
                logger.warning(
                    "Failed to read deliverable %s: %s", relpath, e,
                )
        return "\n".join(texts)

    def _expected_deliverables_for_subtask(
        self,
        subtask: Subtask,
    ) -> list[str]:
        """Return expected deliverable filenames for the current subtask."""
        if not self._process:
            return []

        deliverables = self._process.get_deliverables()
        if not deliverables:
            return []

        phase_hint = str(getattr(subtask, "phase_id", "") or "").strip()
        if phase_hint in deliverables:
            return deliverables[phase_hint]

        # Strict phase-mode subtasks use phase IDs directly; enforce only the
        # matching phase's deliverables to avoid cross-phase false negatives.
        if subtask.id in deliverables:
            return deliverables[subtask.id]

        # Backward-compatible fallback for single-phase processes where the
        # planner subtask ID may not match the declared phase ID exactly.
        if len(deliverables) == 1:
            return next(iter(deliverables.values()))

        phase_descriptions: dict[str, str] = {}
        process = self._process
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
            return deliverables[phase_id]

        return []

    def _is_advisory_tool_failure(
        self,
        tool_name: str,
        error: str | None,
    ) -> bool:
        """Return True if a failed tool call should be treated as advisory."""
        if not error:
            return False

        if self._tool_success_policy == "safety_integrity_only":
            return not self._is_hard_safety_or_integrity_failure(error)

        if tool_name not in self._ADVISORY_TOOL_FAILURES:
            return False

        text = error.lower()
        if self._is_hard_safety_or_integrity_failure(error):
            return False

        status_code = _extract_http_status(error)
        if status_code is not None and 400 <= status_code < 600:
            return True

        advisory_markers = (
            "timeout",
            "timed out",
            "connection failed",
            "temporarily unavailable",
            "not found",
            "rate limit",
            "rate-limited",
            "response too large",
        )
        return any(marker in text for marker in advisory_markers)

    @staticmethod
    def _is_hard_safety_or_integrity_failure(error: str | None) -> bool:
        text = str(error or "").lower()
        hard_fail_markers = (
            "blocked host:",
            "safety violation:",
            "only http:// and https:// urls are allowed",
            "no url provided",
            "no search query provided",
            "permission denied",
            "operation not permitted",
            "read-only file system",
            "outside writable roots",
            "path escapes",
        )
        return any(marker in text for marker in hard_fail_markers)

    @classmethod
    def _flatten_text_values(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, (int, float, bool)):
            return [str(value)]
        if isinstance(value, dict):
            pieces: list[str] = []
            for key in sorted(value.keys(), key=lambda item: str(item)):
                pieces.extend(cls._flatten_text_values(key))
                pieces.extend(cls._flatten_text_values(value.get(key)))
            return pieces
        if isinstance(value, (list, tuple, set)):
            pieces: list[str] = []
            for item in value:
                pieces.extend(cls._flatten_text_values(item))
            return pieces
        text = str(value).strip()
        return [text] if text else []

    @staticmethod
    def _read_text_file_if_small(path: Path, *, max_bytes: int = 300_000) -> str:
        try:
            if not path.exists() or not path.is_file():
                return ""
            if path.stat().st_size > max_bytes:
                return ""
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    def _synthesis_reference_haystack(
        self,
        *,
        subtask: Subtask,
        result_summary: str,
        tool_calls: list,
        workspace: Path | None,
    ) -> str:
        pieces: list[str] = []
        summary = str(result_summary or "").strip()
        if summary:
            pieces.append(summary)

        for tc in tool_calls:
            tool_name = str(getattr(tc, "tool", "") or "").strip()
            if tool_name:
                pieces.append(tool_name)
            args = getattr(tc, "args", {})
            pieces.extend(self._flatten_text_values(args))
            result = getattr(tc, "result", None)
            changed = getattr(result, "files_changed", [])
            if isinstance(changed, list):
                pieces.extend(
                    str(item).strip()
                    for item in changed
                    if str(item).strip()
                )

        if workspace is not None:
            expected_current = self._expected_deliverables_for_subtask(subtask)
            for rel_path in expected_current:
                text = self._read_text_file_if_small(workspace / rel_path)
                if text:
                    pieces.append(text[:60_000])

        return "\n".join(pieces).lower()

    def _synthesis_input_integrity_checks(
        self,
        *,
        subtask: Subtask,
        result_summary: str,
        tool_calls: list,
        workspace: Path | None,
    ) -> list[Check]:
        """Verify synthesis subtasks integrate declared upstream preparation."""
        if not self._process or not subtask.is_synthesis or workspace is None:
            return []
        dependency_ids = [
            str(dep).strip()
            for dep in subtask.depends_on
            if str(dep).strip()
        ]
        if not dependency_ids:
            return []

        deliverables = self._process.get_deliverables()
        if not deliverables:
            return []

        dependency_inputs: dict[str, list[str]] = {}
        for dep_id in dependency_ids:
            raw_paths = deliverables.get(dep_id, [])
            if not isinstance(raw_paths, list):
                continue
            paths: list[str] = []
            for item in raw_paths:
                text = str(item).strip()
                if text and text not in paths:
                    paths.append(text)
            if paths:
                dependency_inputs[dep_id] = paths
        if not dependency_inputs:
            return []

        checks: list[Check] = []
        available_inputs: dict[str, list[str]] = {}
        for dep_id, rel_paths in dependency_inputs.items():
            existing: list[str] = []
            missing: list[str] = []
            for rel_path in rel_paths:
                path = workspace / rel_path
                try:
                    if path.exists() and path.is_file() and path.stat().st_size > 0:
                        existing.append(rel_path)
                    else:
                        missing.append(rel_path)
                except OSError:
                    missing.append(rel_path)
            if existing:
                available_inputs[dep_id] = existing
                checks.append(Check(
                    name=f"synthesis_input_ready_{dep_id}",
                    passed=True,
                ))
                continue
            checks.append(Check(
                name=f"synthesis_input_ready_{dep_id}",
                passed=False,
                detail=(
                    "Missing non-empty upstream deliverables for "
                    f"'{dep_id}': {', '.join(missing) if missing else 'unknown'}"
                ),
            ))

        if not available_inputs:
            return checks

        haystack = self._synthesis_reference_haystack(
            subtask=subtask,
            result_summary=result_summary,
            tool_calls=tool_calls,
            workspace=workspace,
        )
        for dep_id, rel_paths in available_inputs.items():
            signals: list[str] = [dep_id.lower()]
            for rel_path in rel_paths:
                lowered = rel_path.lower()
                if lowered not in signals:
                    signals.append(lowered)
                basename = Path(rel_path).name.lower()
                if basename and basename not in signals:
                    signals.append(basename)
            integrated = any(signal in haystack for signal in signals if signal)
            checks.append(Check(
                name=f"synthesis_input_integrated_{dep_id}",
                passed=integrated,
                detail=(
                    "Synthesis output did not reference upstream dependency "
                    f"'{dep_id}' ({', '.join(rel_paths)})."
                    if not integrated else None
                ),
            ))

        return checks

    @staticmethod
    def _build_feedback(checks: list[Check]) -> str:
        """Build actionable feedback from failed checks."""
        lines = ["Verification failed:"]
        for c in checks:
            detail = f" — {c.detail}" if c.detail else ""
            lines.append(f"  - {c.name}{detail}")
        return "\n".join(lines)

    @staticmethod
    def _build_advisory_feedback(checks: list[Check]) -> str | None:
        if not checks:
            return None
        lines = ["Verification warnings:"]
        for c in checks:
            if c.detail:
                lines.append(f"  - {c.name} — {c.detail}")
        if len(lines) == 1:
            return None
        return "\n".join(lines)
