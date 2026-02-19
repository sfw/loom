"""Three-tier verification gates for subtask outputs.

Tier 1: Deterministic checks (free, instant, no LLM)
Tier 2: Independent LLM verification (fresh context, different model)
Tier 3: Voting verification (N independent checks, majority agreement)

The model that performed the work never checks its own output.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from loom.config import VerificationConfig
from loom.engine.semantic_compactor import SemanticCompactor
from loom.events.bus import Event, EventBus
from loom.events.types import (
    MODEL_INVOCATION,
    VERIFICATION_DETERMINISTIC_BLOCK_RATE,
    VERIFICATION_FALSE_NEGATIVE_CANDIDATE,
    VERIFICATION_INCONCLUSIVE_RATE,
    VERIFICATION_OUTCOME,
    VERIFICATION_RULE_APPLIED,
    VERIFICATION_RULE_FAILURE_BY_TYPE,
    VERIFICATION_RULE_SKIPPED,
    VERIFICATION_SHADOW_DIFF,
)
from loom.models.request_diagnostics import collect_request_diagnostics
from loom.models.retry import ModelRetryPolicy, call_with_model_retry
from loom.models.router import ModelRouter, ResponseValidator
from loom.prompts.assembler import PromptAssembler
from loom.state.task_state import Subtask
from loom.utils.tokens import estimate_tokens

if TYPE_CHECKING:
    from loom.processes.schema import ProcessDefinition

logger = logging.getLogger(__name__)

_VALID_OUTCOMES = {
    "pass",
    "pass_with_warnings",
    "partial_verified",
    "fail",
}

_VALID_SEVERITY_CLASSES = {
    "hard_invariant",
    "semantic",
    "inconclusive",
    "infra",
}

_REASON_CODE_SEVERITY: dict[str, str] = {
    "hard_invariant_failed": "hard_invariant",
    "parse_inconclusive": "inconclusive",
    "infra_verifier_error": "infra",
}


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


@dataclass
class Check:
    """A single verification check result."""

    name: str
    passed: bool
    detail: str | None = None


@dataclass
class VerificationResult:
    """Result of running verification gates."""

    tier: int
    passed: bool
    confidence: float = 1.0
    checks: list[Check] = field(default_factory=list)
    feedback: str | None = None
    outcome: str = "pass"
    reason_code: str = ""
    severity_class: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = str(self.outcome or "pass").strip().lower()
        self.outcome = normalized if normalized in _VALID_OUTCOMES else "pass"
        if self.outcome == "fail":
            self.passed = False
        severity = str(self.severity_class or "").strip().lower()
        if severity not in _VALID_SEVERITY_CLASSES:
            severity = self._infer_severity_class(
                reason_code=self.reason_code,
                outcome=self.outcome,
                tier=self.tier,
                passed=self.passed,
            )
        self.severity_class = severity

    @staticmethod
    def _infer_severity_class(
        *,
        reason_code: str,
        outcome: str,
        tier: int,
        passed: bool,
    ) -> str:
        normalized_reason = str(reason_code or "").strip().lower()
        if normalized_reason in _REASON_CODE_SEVERITY:
            return _REASON_CODE_SEVERITY[normalized_reason]
        if outcome == "fail" and tier <= 1:
            return "hard_invariant"
        if not passed and outcome == "fail":
            return "semantic"
        return "semantic"


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

    def __init__(
        self,
        process: ProcessDefinition | None = None,
        *,
        phase_scope_default: str = "current_phase",
        regex_default_advisory: bool = True,
    ):
        self._process = process
        normalized_scope = str(phase_scope_default or "current_phase").strip().lower()
        self._phase_scope_default = (
            normalized_scope if normalized_scope in {"current_phase", "global"}
            else "current_phase"
        )
        self._regex_default_advisory = bool(regex_default_advisory)

    async def verify(
        self,
        subtask: Subtask,
        result_summary: str,
        tool_calls: list,
        workspace: Path | None = None,
        evidence_tool_calls: list | None = None,
        evidence_records: list[dict] | None = None,
    ) -> VerificationResult:
        checks: list[Check] = []

        # 1. Did tool calls succeed?
        for tc in tool_calls:
            if not tc.result.success:
                if self._is_advisory_tool_failure(tc.tool, tc.result.error):
                    checks.append(Check(
                        name=f"tool_{tc.tool}_advisory",
                        passed=True,
                        detail=(
                            "Advisory tool failure (non-blocking): "
                            f"{tc.result.error or 'unknown error'}"
                        ),
                    ))
                    continue
                checks.append(Check(
                    name=f"tool_{tc.tool}_success",
                    passed=False,
                    detail=tc.result.error,
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
                            checks.append(syntax_check)

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
                        workspace, tool_calls,
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
                checks.append(Check(
                    name=f"process_rule_{rule.name}",
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

            # 5. Process deliverables check
            # Enforce only deliverables for the matching phase/subtask id.
            expected = self._expected_deliverables_for_subtask(subtask)
            if expected and workspace:
                files_created = set()
                for tc in tool_calls:
                    files_created.update(tc.result.files_changed)
                for expected_path in expected:
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

        # NOTE: Tier 1 remains process-agnostic by design. Semantic, domain-
        # specific quality checks are handled by Tier 2/3 LLM verification.
        # This prevents hard-coding process-specific schemas into core static
        # verification and keeps modular process packages decoupled.

        hard_failures = [c for c in checks if not c.passed]
        advisory_hits = [
            c for c in checks
            if c.passed and c.detail and (
                c.name.startswith("process_rule_")
                or "_advisory" in c.name
                or str(c.detail).lower().startswith("advisory")
            )
        ]
        all_passed = not hard_failures
        outcome = "pass"
        if hard_failures:
            outcome = "fail"
        elif advisory_hits:
            outcome = "pass_with_warnings"
        return VerificationResult(
            tier=1,
            passed=all_passed,
            checks=checks,
            feedback=(
                self._build_feedback(hard_failures)
                if hard_failures
                else self._build_advisory_feedback(advisory_hits)
            ),
            outcome=outcome,
            reason_code="hard_invariant_failed" if hard_failures else "",
            severity_class="hard_invariant" if hard_failures else "semantic",
            metadata={
                "hard_failure_count": len(hard_failures),
                "advisory_count": len(advisory_hits),
            },
        )

    def _is_regex_rule_hard(self, rule) -> bool:
        enforcement = str(getattr(rule, "enforcement", "") or "").strip().lower()
        if enforcement == "hard":
            return True
        if enforcement == "advisory":
            return False
        if self._regex_default_advisory:
            return False
        return str(getattr(rule, "severity", "warning")).strip().lower() == "error"

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

        return None

    @staticmethod
    def _read_deliverable_text(
        workspace: Path, tool_calls: list,
    ) -> str:
        """Read text content from all deliverable files."""
        texts = []
        for tc in tool_calls:
            for f in tc.result.files_changed:
                fpath = workspace / f
                if fpath.exists() and fpath.stat().st_size < 1024 * 1024:
                    try:
                        texts.append(
                            fpath.read_text(encoding="utf-8", errors="replace"),
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to read deliverable %s: %s", f, e,
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

        # Strict phase-mode subtasks use phase IDs directly; enforce only the
        # matching phase's deliverables to avoid cross-phase false negatives.
        if subtask.id in deliverables:
            return deliverables[subtask.id]

        # Backward-compatible fallback for single-phase processes where the
        # planner subtask ID may not match the declared phase ID exactly.
        if len(deliverables) == 1:
            return next(iter(deliverables.values()))

        return []

    def _is_advisory_tool_failure(
        self,
        tool_name: str,
        error: str | None,
    ) -> bool:
        """Return True if a failed tool call should be treated as advisory."""
        if tool_name not in self._ADVISORY_TOOL_FAILURES or not error:
            return False

        text = error.lower()
        hard_fail_markers = (
            "blocked host:",
            "safety violation:",
            "only http:// and https:// urls are allowed",
            "no url provided",
            "no search query provided",
        )
        if any(marker in text for marker in hard_fail_markers):
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


class LLMVerifier:
    """Tier 2: Independent LLM verification.

    Uses a DIFFERENT model instance with NO prior conversation history.
    Assesses whether the subtask output meets acceptance criteria.
    """

    def __init__(
        self,
        model_router: ModelRouter,
        prompt_assembler: PromptAssembler,
        validator: ResponseValidator,
        event_bus: EventBus | None = None,
        verification_config: VerificationConfig | None = None,
    ):
        self._router = model_router
        self._prompts = prompt_assembler
        self._validator = validator
        self._compactor = SemanticCompactor(model_router, role="extractor", tier=1)
        self._event_bus = event_bus
        self._config = verification_config or VerificationConfig()

    _MAX_TOOL_ARGS_CHARS = 240
    _MAX_TOOL_STATUS_CHARS = 220
    _MAX_TOOL_CALLS_TOKENS = 2500
    _MAX_VERIFIER_PROMPT_TOKENS = 8000
    _MAX_RESULT_SUMMARY_CHARS = 5000
    _COMPACT_RESULT_SUMMARY_CHARS = 1800
    _MAX_EVIDENCE_SECTION_CHARS = 2600
    _MAX_EVIDENCE_SECTION_COMPACT_CHARS = 1300
    _ADVISORY_TOOL_FAILURES = frozenset({"web_fetch", "web_fetch_html", "web_search"})
    _SOURCE_ATTRIBUTION_TOOLS = frozenset({"web_fetch", "web_fetch_html"})

    @staticmethod
    def _hard_cap_text(text: str, max_chars: int) -> str:
        """Deterministically bound text length when semantic compaction misses."""
        value = str(text or "")
        if max_chars <= 0:
            return ""
        if len(value) <= max_chars:
            return value
        if max_chars <= 40:
            return value[:max_chars]

        marker = "...[truncated]..."
        remaining = max_chars - len(marker)
        if remaining <= 0:
            return value[:max_chars]

        head = max(16, int(remaining * 0.65))
        tail = max(8, remaining - head)
        if head + tail > remaining:
            tail = max(0, remaining - head)
        compacted = f"{value[:head]}{marker}{value[-tail:] if tail else ''}"
        return compacted[:max_chars]

    async def _compact_text(self, text: str, *, max_chars: int, label: str) -> str:
        compacted = await self._compactor.compact(
            str(text or ""),
            max_chars=max_chars,
            label=label,
        )
        if len(compacted) > max_chars:
            logger.warning(
                "Verifier compaction exceeded budget for %s: got %d chars (limit %d)",
                label,
                len(compacted),
                max_chars,
            )
            return self._hard_cap_text(compacted, max_chars)
        return compacted

    async def _summarize_arg_value(
        self,
        value: object,
        *,
        max_chars: int = 80,
    ) -> object:
        if isinstance(value, str):
            return await self._compact_text(
                value,
                max_chars=max_chars,
                label="tool argument value",
            )
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return await self._compact_text(
                json.dumps(value, ensure_ascii=False, default=str),
                max_chars=max_chars,
                label="tool argument object",
            )
        if isinstance(value, list):
            return await self._compact_text(
                json.dumps(value, ensure_ascii=False, default=str),
                max_chars=max_chars,
                label="tool argument list",
            )
        return await self._compact_text(
            str(value),
            max_chars=max_chars,
            label="tool argument scalar",
        )

    async def _summarize_tool_args(self, args: object, *, max_chars: int) -> str:
        if not isinstance(args, dict):
            return await self._compact_text(
                json.dumps(args, default=str),
                max_chars=max_chars,
                label="tool call args",
            )

        preferred = (
            "path", "file_path", "url", "query", "pattern", "command",
            "operation", "source", "destination", "name", "column_name",
            "row_index", "action", "subtask_id",
        )
        summary: dict[str, object] = {}
        for key in preferred:
            if key in args:
                summary[key] = await self._summarize_arg_value(args.get(key))
        if not summary:
            for key, value in args.items():
                summary[str(key)] = await self._summarize_arg_value(value)

        text = json.dumps(summary, ensure_ascii=False, sort_keys=True)
        if len(text) <= max_chars:
            return text
        return await self._compact_text(
            text,
            max_chars=max_chars,
            label="tool call args summary",
        )

    async def _format_tool_calls_compact(self, tool_calls: list) -> str:
        if not tool_calls:
            return "No tool calls."

        detailed = await self._format_tool_calls_for_prompt(tool_calls)
        return await self._compact_text(
            detailed,
            max_chars=2_400,
            label="verification compact tool-call history",
        )

    async def _format_tool_calls_for_prompt(self, tool_calls: list) -> str:
        if not tool_calls:
            return "No tool calls."

        lines: list[str] = []
        for tc in tool_calls:
            args_text = await self._summarize_tool_args(
                getattr(tc, "args", {}),
                max_chars=self._MAX_TOOL_ARGS_CHARS,
            )
            result = getattr(tc, "result", None)
            if getattr(result, "success", False):
                status = "OK"
            else:
                error_text = str(getattr(result, "error", "") or "unknown error")
                compact_error = await self._compact_text(
                    error_text,
                    max_chars=self._MAX_TOOL_STATUS_CHARS,
                    label="tool status detail",
                )
                if self._is_advisory_tool_failure(
                    str(getattr(tc, "tool", "") or ""),
                    error_text,
                ):
                    status = f"ADVISORY_FAILURE: {compact_error}"
                else:
                    status = f"FAILED: {compact_error}"
            tool_name = str(getattr(tc, "tool", "") or "unknown")
            lines.append(f"- {tool_name}({args_text}) -> {status}")

        formatted = "\n".join(lines) if lines else "No tool calls."
        if estimate_tokens(formatted) > self._MAX_TOOL_CALLS_TOKENS:
            return await self._compact_text(
                formatted,
                max_chars=self._MAX_TOOL_CALLS_TOKENS * 4,
                label="verification tool-call history",
            )
        return formatted

    def _is_advisory_tool_failure(
        self,
        tool_name: str,
        error: str | None,
    ) -> bool:
        if tool_name not in self._ADVISORY_TOOL_FAILURES or not error:
            return False
        text = error.lower()
        hard_fail_markers = (
            "blocked host:",
            "safety violation:",
            "only http:// and https:// urls are allowed",
            "no url provided",
            "no search query provided",
        )
        if any(marker in text for marker in hard_fail_markers):
            return False

        status_code = _extract_http_status(error)
        if status_code is not None and 400 <= status_code < 600:
            return True
        return any(marker in text for marker in (
            "timeout",
            "timed out",
            "connection failed",
            "temporarily unavailable",
            "not found",
            "rate limit",
            "rate-limited",
            "response too large",
        ))

    def _build_prompt(
        self,
        subtask: Subtask,
        result_summary: str,
        tool_calls_formatted: str,
        *,
        llm_rules: list | None = None,
        phase_scope_default: str = "current_phase",
    ) -> str:
        return self._prompts.build_verifier_prompt(
            subtask=subtask,
            result_summary=result_summary,
            tool_calls_formatted=tool_calls_formatted,
            llm_rules=llm_rules,
            phase_scope_default=phase_scope_default,
        )

    def _phase_scope_default(self) -> str:
        value = str(getattr(self._config, "phase_scope_default", "current_phase") or "")
        normalized = value.strip().lower()
        if normalized in {"current_phase", "global"}:
            return normalized
        return "current_phase"

    def _expected_verifier_response_keys(self) -> list[str]:
        process = getattr(self._prompts, "process", None)
        if process is None:
            return ["passed"]
        getter = getattr(process, "verifier_required_response_fields", None)
        if callable(getter):
            keys = getter()
            if isinstance(keys, list):
                normalized = [
                    str(item).strip()
                    for item in keys
                    if str(item).strip()
                ]
                if "passed" not in normalized:
                    normalized.insert(0, "passed")
                return normalized
        return ["passed"]

    def _emit_rule_scope_event(
        self,
        *,
        task_id: str,
        subtask_id: str,
        applied: bool,
        rule,
        reason: str,
    ) -> None:
        if not self._event_bus or not task_id:
            return
        data: dict[str, object] = {
            "subtask_id": subtask_id,
            "rule_id": str(getattr(rule, "name", "") or ""),
            "reason": reason,
            "rule_type": str(getattr(rule, "type", "") or ""),
            "severity": str(getattr(rule, "severity", "") or ""),
            "enforcement": str(getattr(rule, "enforcement", "") or ""),
            "scope": str(getattr(rule, "scope", "") or ""),
            "applies_to_phases": list(getattr(rule, "applies_to_phases", []) or []),
        }
        self._event_bus.emit(Event(
            event_type=(
                VERIFICATION_RULE_APPLIED if applied
                else VERIFICATION_RULE_SKIPPED
            ),
            task_id=task_id,
            data=data,
        ))

    def _select_phase_scoped_rules(
        self,
        subtask: Subtask,
        *,
        task_id: str,
    ) -> list:
        process = getattr(self._prompts, "process", None)
        if process is None or not process.has_verification_rules():
            return []

        all_rules = list(process.llm_rules())
        phase_scope_default = self._phase_scope_default()
        if phase_scope_default == "global":
            for rule in all_rules:
                self._emit_rule_scope_event(
                    task_id=task_id,
                    subtask_id=subtask.id,
                    applied=True,
                    rule=rule,
                    reason="global_mode",
                )
            return all_rules

        scoped_rules = process.llm_rules_for_subtask(
            subtask.id,
            phase_scope_default=phase_scope_default,
        )
        scoped_ids = {id(rule) for rule in scoped_rules}
        for rule in all_rules:
            is_applied = id(rule) in scoped_ids
            self._emit_rule_scope_event(
                task_id=task_id,
                subtask_id=subtask.id,
                applied=is_applied,
                rule=rule,
                reason="phase_match" if is_applied else "phase_mismatch",
            )
        return scoped_rules

    @staticmethod
    def _normalized_text(value: object) -> str:
        return " ".join(str(value or "").strip().split())

    @staticmethod
    def _normalize_url(value: object) -> str:
        text = str(value or "").strip().strip("<>")
        if not text:
            return ""
        return text.rstrip("/")

    @classmethod
    def _domain_from_url(cls, value: object) -> str:
        text = cls._normalize_url(value)
        if not text:
            return ""
        try:
            return (urlparse(text).netloc or "").lower()
        except Exception:
            return ""

    @staticmethod
    def _merge_evidence_records(
        base: list[dict] | None,
        extra: list[dict] | None,
    ) -> list[dict]:
        try:
            from loom.state.evidence import merge_evidence_records
            return merge_evidence_records(base or [], extra or [])
        except Exception:
            merged: list[dict] = []
            seen: set[str] = set()
            for bucket in (base or [], extra or []):
                for item in bucket:
                    if not isinstance(item, dict):
                        continue
                    evidence_id = str(item.get("evidence_id", "")).strip()
                    if evidence_id and evidence_id in seen:
                        continue
                    if evidence_id:
                        seen.add(evidence_id)
                    merged.append(dict(item))
            return merged

    @staticmethod
    def _extract_evidence_from_tool_calls(
        subtask_id: str,
        tool_calls: list | None,
    ) -> list[dict]:
        if not tool_calls:
            return []
        try:
            from loom.state.evidence import extract_evidence_records
        except Exception:
            return []
        try:
            return extract_evidence_records(
                task_id="",
                subtask_id=subtask_id,
                tool_calls=list(tool_calls),
                existing_ids=set(),
            )
        except Exception:
            return []

    @classmethod
    def _build_evidence_ledger_section(
        cls,
        records: list[dict],
        *,
        workspace: Path | None,
        tool_calls: list | None,
        max_chars: int,
    ) -> str:
        if not records:
            return ""

        facet_keys: set[str] = set()
        facet_examples: set[str] = set()
        domains: set[str] = set()
        tools: set[str] = set()

        for item in records:
            if not isinstance(item, dict):
                continue
            tool = cls._normalized_text(item.get("tool")).lower() or "unknown"
            tools.add(tool)

            facets = item.get("facets", {})
            if isinstance(facets, dict):
                for key, value in facets.items():
                    facet_key = cls._normalized_text(key).lower()
                    facet_val = cls._normalized_text(value)
                    if facet_key:
                        facet_keys.add(facet_key)
                    if facet_key and facet_val:
                        facet_examples.add(f"{facet_key}={facet_val}")

            source_url = cls._normalize_url(item.get("source_url"))
            if source_url:
                domain = cls._domain_from_url(source_url) or "unknown-domain"
                domains.add(domain)

        lines = ["EVIDENCE CONTEXT SNAPSHOT (advisory, non-binding):"]
        lines.append(
            "- Treat this as support context for LLM judgment, not a strict "
            "schema validator."
        )
        lines.append(
            "- Structured outputs may vary by process; infer schema from "
            "headers/sections and acceptance criteria."
        )
        if tools:
            lines.append("- observed_tools: " + ", ".join(sorted(tools)[:10]))
        if facet_keys:
            lines.append(
                "- observed_facet_keys: "
                + ", ".join(sorted(facet_keys)[:12])
            )
        if facet_examples:
            lines.append(
                "- observed_facet_examples: "
                + ", ".join(sorted(facet_examples)[:12])
            )
        if domains:
            lines.append(
                "- observed_source_domain_examples: "
                + ", ".join(sorted(domains)[:10])
            )

        csv_lines = cls._csv_schema_lines(
            workspace=workspace,
            tool_calls=tool_calls,
        )
        if csv_lines:
            lines.append("- changed_csv_schema_hints:")
            lines.extend(csv_lines)

        lines.append("- sample_records:")
        for item in records[:10]:
            if not isinstance(item, dict):
                continue
            evidence_id = cls._normalized_text(item.get("evidence_id")) or "EV-UNKNOWN"
            facets = {}
            raw_facets = item.get("facets", {})
            if isinstance(raw_facets, dict):
                facets = {
                    cls._normalized_text(key): cls._normalized_text(value)
                    for key, value in raw_facets.items()
                    if cls._normalized_text(key) and cls._normalized_text(value)
                }
            facet_preview = "none"
            if facets:
                facet_preview = ", ".join(
                    f"{key}={value}" for key, value in sorted(facets.items())
                )
                if len(facet_preview) > 96:
                    facet_preview = facet_preview[:93] + "..."
            source = (
                cls._normalize_url(item.get("source_url"))
                or cls._normalized_text(item.get("query"))
                or "n/a"
            )
            if len(source) > 96:
                source = source[:93] + "..."
            lines.append(
                f"  - {evidence_id} | facets={facet_preview} | source={source}"
            )

        return cls._hard_cap_text("\n".join(lines), max_chars=max_chars)

    @staticmethod
    def _is_attribution_column(header: str) -> bool:
        h = str(header or "").strip().lower().replace(" ", "_")
        if not h:
            return False
        if "url" in h or "citation" in h or "reference" in h:
            return True
        if h in {"source", "source_link", "source_links"}:
            return True
        if h.startswith("source_") and any(
            marker in h for marker in ("url", "link", "ref", "citation")
        ):
            return True
        return False

    @classmethod
    def _csv_schema_lines(
        cls,
        *,
        workspace: Path | None,
        tool_calls: list | None,
    ) -> list[str]:
        if workspace is None or not tool_calls:
            return []

        changed_csv: list[Path] = []
        seen: set[str] = set()
        for tc in tool_calls:
            result = getattr(tc, "result", None)
            files_changed = getattr(result, "files_changed", []) if result else []
            for rel_path in files_changed:
                path = workspace / str(rel_path)
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                if path.suffix.lower() != ".csv" or not path.exists() or not path.is_file():
                    continue
                changed_csv.append(path)

        lines: list[str] = []
        for path in changed_csv[:8]:
            try:
                with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                    reader = csv.DictReader(f)
                    headers = list(reader.fieldnames or [])
                    attrib_cols = [
                        col for col in headers
                        if cls._is_attribution_column(col)
                    ]
                    rows = list(reader)
            except Exception:
                continue

            header_preview = ", ".join(headers) if headers else "(none)"
            if len(header_preview) > 420:
                header_preview = header_preview[:417] + "..."
            lines.append(f"  - {path.name}: headers [{header_preview}]")

            if not attrib_cols or not rows:
                continue
            col_text = ", ".join(attrib_cols)
            lines.append(f"    attribution_columns: [{col_text}]")

        return lines

    def _emit_model_event(
        self,
        *,
        task_id: str,
        subtask_id: str,
        model_name: str,
        phase: str,
        details: dict | None = None,
    ) -> None:
        if not self._event_bus or not task_id:
            return
        data: dict = {
            "subtask_id": subtask_id,
            "model": model_name,
            "phase": phase,
        }
        if isinstance(details, dict) and details:
            data.update(details)
        self._event_bus.emit(Event(
            event_type=MODEL_INVOCATION,
            task_id=task_id,
            data=data,
        ))

    @staticmethod
    def _to_bool(value: object) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if value == 1:
                return True
            if value == 0:
                return False
            return None
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "pass", "passed", "success", "successful"}:
                return True
            if lowered in {"false", "no", "fail", "failed", "failure", "unsuccessful"}:
                return False
        return None

    @staticmethod
    def _to_confidence(value: object) -> float | None:
        if isinstance(value, (int, float)):
            numeric = float(value)
            if numeric > 1.0 and numeric <= 100.0:
                numeric = numeric / 100.0
            return max(0.0, min(1.0, numeric))

        if isinstance(value, str):
            text = value.strip().lower()
            if not text:
                return None
            if text.endswith("%"):
                try:
                    numeric = float(text[:-1]) / 100.0
                    return max(0.0, min(1.0, numeric))
                except ValueError:
                    return None
            try:
                numeric = float(text)
            except ValueError:
                return None
            if numeric > 1.0 and numeric <= 100.0:
                numeric = numeric / 100.0
            return max(0.0, min(1.0, numeric))
        return None

    @staticmethod
    def _to_issues(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text or text.lower() in {"none", "no issues", "n/a"}:
                return []
            pieces = re.split(r"[;\n]+", text)
            return [piece.strip(" -\t") for piece in pieces if piece.strip(" -\t")]
        if isinstance(value, list):
            issues: list[str] = []
            for item in value:
                text = str(item).strip()
                if text:
                    issues.append(text)
            return issues
        return [str(value).strip()] if str(value).strip() else []

    @staticmethod
    def _to_int(value: object) -> int | None:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            return None
        if isinstance(value, str):
            text = value.strip().replace(",", "")
            if not text:
                return None
            try:
                return int(float(text))
            except ValueError:
                return None
        return None

    @staticmethod
    def _normalize_outcome(value: object, *, passed: bool) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in _VALID_OUTCOMES:
            return normalized
        return "pass" if passed else "fail"

    @staticmethod
    def _extract_reason_code_from_text(text: str) -> str:
        match = re.search(
            r"\breason_code\b\s*[:=]\s*([a-z0-9_]+)",
            str(text or ""),
            flags=re.IGNORECASE,
        )
        return str(match.group(1)).strip().lower() if match else ""

    @staticmethod
    def _extract_severity_class_from_text(text: str) -> str:
        match = re.search(
            r"\b(?:severity_class|severity)\b\s*[:=]\s*([a-z_]+)",
            str(text or ""),
            flags=re.IGNORECASE,
        )
        if not match:
            return ""
        severity = str(match.group(1)).strip().lower()
        if severity in _VALID_SEVERITY_CLASSES:
            return severity
        return ""

    @classmethod
    def _extract_outcome_from_text(cls, text: str, *, passed: bool) -> str:
        match = re.search(
            r"\boutcome\b\s*[:=]\s*([a-z_]+)",
            str(text or ""),
            flags=re.IGNORECASE,
        )
        if not match:
            return "pass" if passed else "fail"
        return cls._normalize_outcome(match.group(1), passed=passed)

    @classmethod
    def _coerce_assessment_mapping(cls, payload: dict) -> dict | None:
        normalized: dict[str, object] = {
            str(key).strip().lower(): value
            for key, value in payload.items()
        }
        passed_raw = (
            normalized.get("passed")
            if "passed" in normalized
            else normalized.get("pass", normalized.get("result"))
        )
        passed = cls._to_bool(passed_raw)
        if passed is None and isinstance(passed_raw, str):
            passed = cls._extract_passed_from_text(passed_raw)
        if passed is None:
            return None

        confidence = cls._to_confidence(normalized.get("confidence"))
        if confidence is None:
            confidence = 0.8 if passed else 0.4

        feedback_raw = (
            normalized.get("feedback")
            or normalized.get("suggestion")
            or normalized.get("reason")
            or normalized.get("rationale")
        )
        feedback = str(feedback_raw).strip() if feedback_raw is not None else None

        issues = cls._to_issues(normalized.get("issues"))
        outcome = cls._normalize_outcome(normalized.get("outcome"), passed=passed)
        reason_code = str(
            normalized.get("reason_code")
            or normalized.get("failure_reason")
            or "",
        ).strip().lower()
        severity_class = str(
            normalized.get("severity_class")
            or normalized.get("severity")
            or "",
        ).strip().lower()
        if severity_class not in _VALID_SEVERITY_CLASSES:
            severity_class = ""
        metadata: dict[str, object] = {}
        metadata_raw = normalized.get("metadata", {})
        if isinstance(metadata_raw, dict):
            metadata = {
                str(key).strip(): value
                for key, value in metadata_raw.items()
                if str(key).strip()
            }

        base_keys = {
            "passed",
            "pass",
            "result",
            "confidence",
            "feedback",
            "suggestion",
            "reason",
            "rationale",
            "issues",
            "outcome",
            "reason_code",
            "failure_reason",
            "severity_class",
            "severity",
            "metadata",
        }
        for key, value in normalized.items():
            if key in base_keys:
                continue
            if key in metadata:
                continue
            metadata[key] = value

        return {
            "passed": passed,
            "confidence": confidence,
            "feedback": feedback,
            "issues": issues,
            "outcome": outcome,
            "reason_code": reason_code,
            "severity_class": severity_class,
            "metadata": metadata,
        }

    @staticmethod
    def _strip_outer_fences(text: str) -> str:
        value = text.strip()
        if not value.startswith("```"):
            return value
        lines = value.splitlines()
        if not lines:
            return value
        body = "\n".join(lines[1:])
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3]
        return body.strip()

    @classmethod
    def _parse_yaml_like_assessment(cls, text: str) -> dict | None:
        try:
            import yaml
        except Exception:
            return None

        cleaned = cls._strip_outer_fences(str(text or ""))
        if not cleaned:
            return None

        candidates = [cleaned]
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            candidates.append(cleaned[first_brace : last_brace + 1])

        for candidate in candidates:
            try:
                parsed = yaml.safe_load(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                coerced = cls._coerce_assessment_mapping(parsed)
                if coerced is not None:
                    return coerced
        return None

    @staticmethod
    def _extract_passed_from_text(text: str) -> bool | None:
        lowered = str(text or "").lower()
        checks: list[tuple[str, bool]] = [
            (r"\bpassed?\s*[:=]\s*(true|yes|pass(?:ed)?|success(?:ful)?)\b", True),
            (r"\bpassed?\s*[:=]\s*(false|no|fail(?:ed)?|failure|unsuccessful)\b", False),
            (
                r"\b(verdict|result|outcome|assessment)\s*[:=]\s*"
                r"(pass(?:ed)?|success(?:ful)?)\b",
                True,
            ),
            (
                r"\b(verdict|result|outcome|assessment)\s*[:=]\s*"
                r"(fail(?:ed)?|failure|unsuccessful)\b",
                False,
            ),
            (r"\bsubtask\s+(passed|succeeded)\b", True),
            (r"\bsubtask\s+(failed|did not pass)\b", False),
            (r"\bacceptance criteria\s+(were|was)\s+met\b", True),
            (r"\bacceptance criteria\s+(were|was)\s+not met\b", False),
        ]
        for pattern, passed in checks:
            if re.search(pattern, lowered):
                return passed
        return None

    @classmethod
    def _extract_confidence_from_text(cls, text: str) -> float | None:
        lowered = str(text or "").lower()
        match = re.search(r"\bconfidence\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?%?)", lowered)
        if not match:
            return None
        return cls._to_confidence(match.group(1))

    @staticmethod
    def _extract_feedback_from_text(text: str) -> str | None:
        for label in ("feedback", "suggestion", "reason", "rationale"):
            match = re.search(
                rf"\b{label}\b\s*[:=]\s*(.+)",
                text,
                flags=re.IGNORECASE,
            )
            if match:
                value = match.group(1).strip()
                if value:
                    return value
        return None

    @staticmethod
    def _extract_issues_from_text(text: str) -> list[str]:
        lines = str(text or "").splitlines()
        issues: list[str] = []
        capture = False
        for raw in lines:
            line = raw.strip()
            if not line:
                if capture:
                    break
                continue
            lowered = line.lower()
            if lowered.startswith("issues:"):
                capture = True
                trailing = line.split(":", 1)[1].strip()
                if trailing and trailing.lower() not in {"none", "no issues", "n/a"}:
                    issues.append(trailing)
                continue
            if capture:
                if line.startswith(("-", "*", "•")):
                    issues.append(line.lstrip("-*• ").strip())
                    continue
                if ":" in line and not line.startswith(("-", "*", "•")):
                    break
                issues.append(line.strip())
        return [item for item in issues if item]

    @classmethod
    def _coerce_assessment_from_text(cls, text: str) -> dict | None:
        raw = str(text or "").strip()
        if not raw:
            return None

        structured = cls._parse_yaml_like_assessment(raw)
        if structured is not None:
            return structured

        passed = cls._extract_passed_from_text(raw)
        if passed is None:
            return None

        confidence = cls._extract_confidence_from_text(raw)
        if confidence is None:
            confidence = 0.8 if passed else 0.4

        feedback = cls._extract_feedback_from_text(raw)
        issues = cls._extract_issues_from_text(raw)

        return {
            "passed": passed,
            "confidence": confidence,
            "feedback": feedback,
            "issues": issues,
            "outcome": cls._extract_outcome_from_text(raw, passed=passed),
            "reason_code": cls._extract_reason_code_from_text(raw),
            "severity_class": cls._extract_severity_class_from_text(raw),
            "metadata": {},
        }

    @staticmethod
    def _assessment_to_result(assessment: dict) -> VerificationResult:
        passed = bool(assessment.get("passed", True))
        confidence = float(assessment.get("confidence", 0.5))
        issues = LLMVerifier._to_issues(assessment.get("issues"))
        feedback = assessment.get("feedback") or assessment.get("suggestion")
        feedback_text = str(feedback).strip() if feedback is not None else None
        detail = "; ".join(issues) if issues else None
        outcome = LLMVerifier._normalize_outcome(
            assessment.get("outcome"),
            passed=passed,
        )
        reason_code = str(assessment.get("reason_code") or "").strip().lower()
        severity_class = str(assessment.get("severity_class") or "").strip().lower()
        if severity_class not in _VALID_SEVERITY_CLASSES:
            severity_class = ""
        if passed and outcome == "pass" and issues:
            outcome = "pass_with_warnings"
        metadata = assessment.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        return VerificationResult(
            tier=2,
            passed=passed,
            confidence=confidence,
            checks=[Check(
                name="llm_assessment",
                passed=passed,
                detail=detail,
            )],
            feedback=feedback_text,
            outcome=outcome,
            reason_code=reason_code,
            severity_class=severity_class,
            metadata={"issues": issues, **metadata},
        )

    def _inconclusive_result(self) -> VerificationResult:
        return VerificationResult(
            tier=2,
            passed=False,
            confidence=0.3,
            feedback="Verification inconclusive: could not parse verifier output.",
            outcome="fail",
            reason_code="parse_inconclusive",
            severity_class="inconclusive",
        )

    def _parse_verifier_response(self, response) -> VerificationResult | None:
        validation = self._validator.validate_json_response(
            response,
            expected_keys=self._expected_verifier_response_keys(),
        )
        if validation.valid and validation.parsed is not None:
            assessment = self._coerce_assessment_mapping(validation.parsed)
            if assessment is not None:
                return self._assessment_to_result(assessment)

        fallback = self._coerce_assessment_from_text(response.text or "")
        if fallback is not None:
            return self._assessment_to_result(fallback)
        return None

    async def _repair_assessment_with_model(
        self,
        model,
        raw_text: str,
    ) -> VerificationResult | None:
        expected_keys = self._expected_verifier_response_keys()
        expected_display = ", ".join(f'"{key}"' for key in expected_keys)
        metadata_hint = ""
        process = getattr(self._prompts, "process", None)
        if process is not None:
            metadata_getter = getattr(process, "verifier_metadata_fields", None)
            if callable(metadata_getter):
                metadata_fields = metadata_getter()
                if isinstance(metadata_fields, list) and metadata_fields:
                    names = ", ".join(
                        str(item).strip()
                        for item in metadata_fields
                        if str(item).strip()
                    )
                    if names:
                        metadata_hint = (
                            "\nWhen inferable, include these metadata keys: "
                            f"{names}."
                        )
        repair_prompt = (
            "Repair the following verifier output into a strict JSON object with keys:\n"
            "{"
            + expected_display
            + "}\n"
            "Use only values directly inferable from the text. If unknown, use empty "
            "strings, [] or {}. Respond with JSON only."
            + metadata_hint
            + "\n\n"
            "RAW OUTPUT:\n"
            f"{raw_text}"
        )
        try:
            response = await call_with_model_retry(
                lambda: model.complete([{"role": "user", "content": repair_prompt}]),
                policy=ModelRetryPolicy(),
            )
        except Exception:
            return None
        return self._parse_verifier_response(response)

    async def _repair_assessment_with_alternate_model(
        self,
        raw_text: str,
    ) -> VerificationResult | None:
        try:
            alternate = self._router.select(tier=2, role="verifier")
        except Exception:
            return None
        return await self._repair_assessment_with_model(alternate, raw_text)

    async def _invoke_and_parse(self, model, prompt: str) -> VerificationResult:
        response = await call_with_model_retry(
            lambda: model.complete([{"role": "user", "content": prompt}]),
            policy=ModelRetryPolicy(),
        )
        parsed = self._parse_verifier_response(response)
        if parsed is not None:
            return parsed

        if not bool(getattr(self._config, "strict_output_protocol", True)):
            return self._inconclusive_result()

        repaired = await self._repair_assessment_with_model(model, response.text or "")
        if repaired is not None:
            repaired.metadata = dict(repaired.metadata)
            repaired.metadata["parser_stage"] = "repair_same_model"
            return repaired

        repaired_alt = await self._repair_assessment_with_alternate_model(
            response.text or "",
        )
        if repaired_alt is not None:
            repaired_alt.metadata = dict(repaired_alt.metadata)
            repaired_alt.metadata["parser_stage"] = "repair_alternate_model"
            return repaired_alt

        return self._inconclusive_result()

    @staticmethod
    def _exception_feedback(error: Exception) -> str:
        detail = str(error) or type(error).__name__
        detail = " ".join(detail.split())
        return (
            "Verification inconclusive: verifier raised an exception: "
            f"{detail}"
        )

    async def verify(
        self,
        subtask: Subtask,
        result_summary: str,
        tool_calls: list,
        workspace: Path | None,
        evidence_tool_calls: list | None = None,
        evidence_records: list[dict] | None = None,
        task_id: str = "",
    ) -> VerificationResult:
        try:
            model = self._router.select(tier=1, role="verifier")
        except Exception as e:
            logger.warning("Verifier model not available: %s", e)
            return VerificationResult(
                tier=0, passed=True, confidence=0.5,
                feedback="Verification skipped: verifier model not configured",
                outcome="pass_with_warnings",
                reason_code="infra_verifier_error",
                severity_class="infra",
            )

        summary_for_prompt = await self._compact_text(
            result_summary,
            max_chars=self._MAX_RESULT_SUMMARY_CHARS,
            label="verification result summary",
        )
        tool_calls_formatted = await self._format_tool_calls_for_prompt(tool_calls)
        derived_evidence = self._extract_evidence_from_tool_calls(
            subtask.id,
            evidence_tool_calls,
        )
        effective_evidence = self._merge_evidence_records(
            [item for item in (evidence_records or []) if isinstance(item, dict)],
            [item for item in derived_evidence if isinstance(item, dict)],
        )
        evidence_section = self._build_evidence_ledger_section(
            effective_evidence,
            workspace=workspace,
            tool_calls=tool_calls,
            max_chars=self._MAX_EVIDENCE_SECTION_CHARS,
        )
        phase_scope_default = self._phase_scope_default()
        selected_rules = self._select_phase_scoped_rules(
            subtask,
            task_id=task_id,
        )
        prompt = self._build_prompt(
            subtask,
            summary_for_prompt,
            tool_calls_formatted,
            llm_rules=selected_rules,
            phase_scope_default=phase_scope_default,
        )
        if evidence_section:
            prompt = prompt + "\n\n" + evidence_section
        if estimate_tokens(prompt) > self._MAX_VERIFIER_PROMPT_TOKENS:
            summary_for_prompt = await self._compact_text(
                summary_for_prompt,
                max_chars=self._COMPACT_RESULT_SUMMARY_CHARS,
                label="verification compact summary",
            )
            tool_calls_formatted = await self._format_tool_calls_compact(tool_calls)
            prompt = self._build_prompt(
                subtask,
                summary_for_prompt,
                tool_calls_formatted,
                llm_rules=selected_rules,
                phase_scope_default=phase_scope_default,
            )
            if evidence_section:
                prompt = (
                    prompt
                    + "\n\n"
                    + self._hard_cap_text(
                        evidence_section,
                        self._MAX_EVIDENCE_SECTION_COMPACT_CHARS,
                    )
                )

        request_messages = [{"role": "user", "content": prompt}]
        request_diag = collect_request_diagnostics(
            messages=request_messages,
            origin="verification.tier2.complete",
        )
        self._emit_model_event(
            task_id=task_id,
            subtask_id=subtask.id,
            model_name=model.name,
            phase="start",
            details={
                "operation": "complete",
                "attempt": 1,
                "evidence_record_count": len(effective_evidence),
                **request_diag.to_event_payload(),
            },
        )
        try:
            first_result = await self._invoke_and_parse(model, prompt)
            if not first_result.reason_code and not first_result.passed:
                first_result.reason_code = "llm_semantic_failed"
            self._emit_model_event(
                task_id=task_id,
                subtask_id=subtask.id,
                model_name=model.name,
                phase="done",
                details={
                    "operation": "complete",
                    "attempt": 1,
                    "origin": request_diag.origin,
                    "verifier_passed": first_result.passed,
                    "verifier_confidence": first_result.confidence,
                    "verifier_outcome": first_result.outcome,
                    "reason_code": first_result.reason_code,
                },
            )
            return first_result
        except Exception as e:
            logger.warning("Verifier raised exception: %s", e)
            compact_tool_calls = await self._format_tool_calls_compact(tool_calls)
            compact_prompt = self._build_prompt(
                subtask,
                await self._compact_text(
                    summary_for_prompt,
                    max_chars=self._COMPACT_RESULT_SUMMARY_CHARS,
                    label="verification retry summary",
                ),
                compact_tool_calls,
                llm_rules=selected_rules,
                phase_scope_default=phase_scope_default,
            )
            if evidence_section:
                compact_prompt = (
                    compact_prompt
                    + "\n\n"
                    + self._hard_cap_text(
                        evidence_section,
                        self._MAX_EVIDENCE_SECTION_COMPACT_CHARS,
                    )
                )
            retry_messages = [{"role": "user", "content": compact_prompt}]
            retry_diag = collect_request_diagnostics(
                messages=retry_messages,
                origin="verification.tier2.retry.complete",
            )
            self._emit_model_event(
                task_id=task_id,
                subtask_id=subtask.id,
                model_name=model.name,
                phase="start",
                details={
                    "operation": "complete",
                    "attempt": 2,
                    "evidence_record_count": len(effective_evidence),
                    **retry_diag.to_event_payload(),
                },
            )
            try:
                retry_result = await self._invoke_and_parse(model, compact_prompt)
                if not retry_result.reason_code and not retry_result.passed:
                    retry_result.reason_code = "llm_semantic_failed"
                self._emit_model_event(
                    task_id=task_id,
                    subtask_id=subtask.id,
                    model_name=model.name,
                    phase="done",
                    details={
                        "operation": "complete",
                        "attempt": 2,
                        "origin": retry_diag.origin,
                        "verifier_passed": retry_result.passed,
                        "verifier_confidence": retry_result.confidence,
                        "verifier_outcome": retry_result.outcome,
                        "reason_code": retry_result.reason_code,
                    },
                )
                return retry_result
            except Exception as retry_error:
                logger.warning(
                    "Verifier compact retry raised exception: %s",
                    retry_error,
                )
                self._emit_model_event(
                    task_id=task_id,
                    subtask_id=subtask.id,
                    model_name=model.name,
                    phase="done",
                    details={
                        "operation": "complete",
                        "attempt": 2,
                        "origin": retry_diag.origin,
                        "error": str(retry_error),
                    },
                )
                return VerificationResult(
                    tier=2,
                    passed=False,
                    confidence=0.3,
                    feedback=self._exception_feedback(retry_error),
                    outcome="fail",
                    reason_code="infra_verifier_error",
                    severity_class="infra",
                )


class VotingVerifier:
    """Tier 3: Voting verification.

    Runs N independent Tier 2 verifications. Majority agreement = pass.
    Divergence = flag for human review.
    """

    def __init__(self, llm_verifier: LLMVerifier, vote_count: int = 3):
        self._llm_verifier = llm_verifier
        self._vote_count = vote_count

    async def verify(
        self,
        subtask: Subtask,
        result_summary: str,
        tool_calls: list,
        workspace: Path | None,
        evidence_tool_calls: list | None = None,
        evidence_records: list[dict] | None = None,
        task_id: str = "",
    ) -> VerificationResult:
        tasks = [
            self._llm_verifier.verify(
                subtask,
                result_summary,
                tool_calls,
                workspace,
                evidence_tool_calls=evidence_tool_calls,
                evidence_records=evidence_records,
                task_id=task_id,
            )
            for _ in range(self._vote_count)
        ]
        results = await asyncio.gather(*tasks)

        pass_count = sum(1 for r in results if r.passed)
        majority = pass_count > self._vote_count / 2

        return VerificationResult(
            tier=3,
            passed=majority,
            confidence=pass_count / self._vote_count,
            checks=[Check(
                name="voting",
                passed=majority,
                detail=f"{pass_count}/{self._vote_count} verifiers agreed output is correct",
            )],
            feedback="Divergent verification — flagged for human review" if not majority else None,
            outcome="pass" if majority else "fail",
            reason_code="" if majority else "llm_semantic_failed",
        )


class VerificationGates:
    """Orchestrates the three-tier verification pipeline."""

    def __init__(
        self,
        model_router: ModelRouter,
        prompt_assembler: PromptAssembler,
        config: VerificationConfig,
        process: ProcessDefinition | None = None,
        event_bus: EventBus | None = None,
    ):
        self._config = config
        self._event_bus = event_bus
        validator = ResponseValidator()
        self._tier1 = DeterministicVerifier(
            process=process,
            phase_scope_default=config.phase_scope_default,
            regex_default_advisory=config.regex_default_advisory,
        )
        self._tier2 = LLMVerifier(
            model_router,
            prompt_assembler,
            validator,
            event_bus=event_bus,
            verification_config=config,
        )
        self._tier3 = VotingVerifier(self._tier2, config.tier3_vote_count)

    def _emit_outcome_event(
        self,
        *,
        task_id: str,
        subtask_id: str,
        result: VerificationResult,
    ) -> None:
        if not self._event_bus or not task_id:
            return
        self._event_bus.emit(Event(
            event_type=VERIFICATION_OUTCOME,
            task_id=task_id,
            data={
                "subtask_id": subtask_id,
                "tier": result.tier,
                "passed": result.passed,
                "outcome": result.outcome,
                "reason_code": result.reason_code,
                "severity_class": result.severity_class,
                "confidence": result.confidence,
            },
        ))

    @staticmethod
    def classify_shadow_diff(
        legacy_result: VerificationResult,
        result: VerificationResult,
    ) -> str:
        if not legacy_result.passed and result.passed:
            return "old_fail_new_pass"
        if legacy_result.passed and not result.passed:
            return "old_pass_new_fail"
        if (
            not legacy_result.passed
            and not result.passed
            and legacy_result.reason_code != result.reason_code
        ):
            return "both_fail_reason_diff"
        return "no_diff"

    def _emit_instrumentation_events(
        self,
        *,
        task_id: str,
        subtask_id: str,
        result: VerificationResult,
        legacy_result: VerificationResult | None = None,
    ) -> None:
        if not self._event_bus or not task_id:
            return

        if result.reason_code == "parse_inconclusive":
            self._event_bus.emit(Event(
                event_type=VERIFICATION_INCONCLUSIVE_RATE,
                task_id=task_id,
                data={
                    "subtask_id": subtask_id,
                    "tier": result.tier,
                    "reason_code": result.reason_code,
                },
            ))

        if result.tier == 1 and not result.passed:
            self._event_bus.emit(Event(
                event_type=VERIFICATION_DETERMINISTIC_BLOCK_RATE,
                task_id=task_id,
                data={
                    "subtask_id": subtask_id,
                    "reason_code": result.reason_code,
                    "failed_checks": len([c for c in result.checks if not c.passed]),
                },
            ))

        for check in result.checks:
            if check.passed:
                continue
            if not check.name.startswith("process_rule_"):
                continue
            self._event_bus.emit(Event(
                event_type=VERIFICATION_RULE_FAILURE_BY_TYPE,
                task_id=task_id,
                data={
                    "subtask_id": subtask_id,
                    "check_name": check.name,
                    "tier": result.tier,
                    "reason_code": result.reason_code,
                },
            ))
        if result.tier >= 2 and not result.passed:
            self._event_bus.emit(Event(
                event_type=VERIFICATION_RULE_FAILURE_BY_TYPE,
                task_id=task_id,
                data={
                    "subtask_id": subtask_id,
                    "check_name": "llm_assessment",
                    "tier": result.tier,
                    "reason_code": result.reason_code,
                },
            ))

        if legacy_result is None:
            return

        classification = self.classify_shadow_diff(legacy_result, result)
        if classification == "old_fail_new_pass":
            self._event_bus.emit(Event(
                event_type=VERIFICATION_FALSE_NEGATIVE_CANDIDATE,
                task_id=task_id,
                data={
                    "subtask_id": subtask_id,
                    "old_outcome": legacy_result.outcome,
                    "new_outcome": result.outcome,
                },
            ))

        self._event_bus.emit(Event(
            event_type=VERIFICATION_SHADOW_DIFF,
            task_id=task_id,
            data={
                "subtask_id": subtask_id,
                "classification": classification,
                "old_passed": legacy_result.passed,
                "new_passed": result.passed,
                "old_reason_code": legacy_result.reason_code,
                "new_reason_code": result.reason_code,
                "old_outcome": legacy_result.outcome,
                "new_outcome": result.outcome,
            },
        ))

    @staticmethod
    def _aggregate_non_failing(results: list[VerificationResult]) -> VerificationResult:
        if not results:
            return VerificationResult(
                tier=0,
                passed=True,
                confidence=0.5,
                outcome="pass",
            )

        merged_checks: list[Check] = []
        feedbacks: list[str] = []
        for item in results:
            merged_checks.extend(item.checks or [])
            if item.feedback:
                feedbacks.append(item.feedback)

        outcome = "pass"
        reason_code = ""
        if any(item.outcome == "partial_verified" for item in results):
            outcome = "partial_verified"
            reason_code = next(
                (
                    item.reason_code
                    for item in results
                    if item.outcome == "partial_verified" and item.reason_code
                ),
                "",
            )
        elif any(item.outcome == "pass_with_warnings" for item in results):
            outcome = "pass_with_warnings"

        highest = max(results, key=lambda item: item.tier)
        merged_feedback = None
        if outcome in {"pass_with_warnings", "partial_verified"} and feedbacks:
            merged_feedback = "\n".join(dict.fromkeys(feedbacks))

        merged_metadata: dict[str, object] = {}
        for item in results:
            if item.metadata:
                merged_metadata[f"tier{item.tier}"] = item.metadata

        return VerificationResult(
            tier=highest.tier,
            passed=True,
            confidence=highest.confidence,
            checks=merged_checks,
            feedback=merged_feedback,
            outcome=outcome,
            reason_code=reason_code,
            metadata=merged_metadata,
        )

    @staticmethod
    def _legacy_result_from_tiers(results: list[VerificationResult]) -> VerificationResult:
        if not results:
            return VerificationResult(
                tier=0,
                passed=True,
                confidence=0.5,
                outcome="pass",
            )

        t1 = next((item for item in results if item.tier == 1), None)
        if t1 is not None:
            legacy_regex_hits = [
                check
                for check in (t1.checks or [])
                if check.passed
                and check.name.startswith("process_rule_")
                and "(advisory)" in str(check.detail or "").lower()
            ]
            if legacy_regex_hits:
                return VerificationResult(
                    tier=1,
                    passed=False,
                    confidence=t1.confidence,
                    checks=list(t1.checks or []),
                    feedback=(
                        "Legacy verification would fail on advisory regex rule match."
                    ),
                    outcome="fail",
                    reason_code="legacy_regex_failure",
                    metadata={"legacy_mode": True},
                )

        for item in results:
            if not item.passed:
                return VerificationResult(
                    tier=item.tier,
                    passed=False,
                    confidence=item.confidence,
                    checks=list(item.checks or []),
                    feedback=item.feedback,
                    outcome="fail",
                    reason_code=item.reason_code,
                    metadata={"legacy_mode": True},
                )

        highest = max(results, key=lambda item: item.tier)
        return VerificationResult(
            tier=highest.tier,
            passed=True,
            confidence=highest.confidence,
            checks=list(highest.checks or []),
            outcome="pass",
            metadata={"legacy_mode": True},
        )

    async def verify(
        self,
        subtask: Subtask,
        result_summary: str,
        tool_calls: list,
        workspace: Path | None,
        evidence_tool_calls: list | None = None,
        evidence_records: list[dict] | None = None,
        tier: int = 1,
        task_id: str = "",
    ) -> VerificationResult:
        """Run verification up to the specified tier.

        Each tier must pass before proceeding to the next.
        """
        policy_enabled = bool(getattr(self._config, "policy_engine_enabled", True))
        results: list[VerificationResult] = []

        # Tier 1 always runs when enabled.
        if self._config.tier1_enabled:
            t1 = await self._tier1.verify(
                subtask,
                result_summary,
                tool_calls,
                evidence_tool_calls=evidence_tool_calls,
                evidence_records=evidence_records,
                workspace=workspace,
            )
            results.append(t1)
            if not t1.passed:
                self._emit_outcome_event(
                    task_id=task_id,
                    subtask_id=subtask.id,
                    result=t1,
                )
                return t1

        if tier < 2 or not self._config.tier2_enabled:
            if not policy_enabled:
                result = (
                    results[-1] if results else VerificationResult(
                        tier=1,
                        passed=True,
                        confidence=0.7 if self._config.tier1_enabled else 0.5,
                    )
                )
            else:
                result = self._aggregate_non_failing(results)
            legacy = None
            if policy_enabled and bool(getattr(self._config, "shadow_compare_enabled", False)):
                legacy = self._legacy_result_from_tiers(results)
            self._emit_instrumentation_events(
                task_id=task_id,
                subtask_id=subtask.id,
                result=result,
                legacy_result=legacy,
            )
            self._emit_outcome_event(
                task_id=task_id,
                subtask_id=subtask.id,
                result=result,
            )
            return result

        # Tier 2: independent LLM check.
        t2 = await self._tier2.verify(
            subtask,
            result_summary,
            tool_calls,
            workspace,
            evidence_tool_calls=evidence_tool_calls,
            evidence_records=evidence_records,
            task_id=task_id,
        )
        results.append(t2)
        if not t2.passed:
            self._emit_instrumentation_events(
                task_id=task_id,
                subtask_id=subtask.id,
                result=t2,
            )
            self._emit_outcome_event(
                task_id=task_id,
                subtask_id=subtask.id,
                result=t2,
            )
            return t2

        if tier < 3 or not self._config.tier3_enabled:
            result = t2 if not policy_enabled else self._aggregate_non_failing(results)
            legacy = None
            if policy_enabled and bool(getattr(self._config, "shadow_compare_enabled", False)):
                legacy = self._legacy_result_from_tiers(results)
            self._emit_instrumentation_events(
                task_id=task_id,
                subtask_id=subtask.id,
                result=result,
                legacy_result=legacy,
            )
            self._emit_outcome_event(
                task_id=task_id,
                subtask_id=subtask.id,
                result=result,
            )
            return result

        # Tier 3: voting.
        t3 = await self._tier3.verify(
            subtask,
            result_summary,
            tool_calls,
            workspace,
            evidence_tool_calls=evidence_tool_calls,
            evidence_records=evidence_records,
            task_id=task_id,
        )
        results.append(t3)
        if not t3.passed:
            self._emit_instrumentation_events(
                task_id=task_id,
                subtask_id=subtask.id,
                result=t3,
            )
            self._emit_outcome_event(
                task_id=task_id,
                subtask_id=subtask.id,
                result=t3,
            )
            return t3

        result = t3 if not policy_enabled else self._aggregate_non_failing(results)
        legacy = None
        if policy_enabled and bool(getattr(self._config, "shadow_compare_enabled", False)):
            legacy = self._legacy_result_from_tiers(results)
        self._emit_instrumentation_events(
            task_id=task_id,
            subtask_id=subtask.id,
            result=result,
            legacy_result=legacy,
        )
        self._emit_outcome_event(
            task_id=task_id,
            subtask_id=subtask.id,
            result=result,
        )
        return result
