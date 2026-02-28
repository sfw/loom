"""Three-tier verification gates for subtask outputs.

Tier 1: Deterministic checks (free, instant, no LLM)
Tier 2: Independent LLM verification (fresh context, different model)
Tier 3: Voting verification (N independent checks, majority agreement)

The model that performed the work never checks its own output.
"""

from __future__ import annotations

import asyncio
import contextvars
import csv
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from loom.config import CompactorLimitsConfig, VerificationConfig, VerifierLimitsConfig
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
from loom.models.request_diagnostics import (
    collect_request_diagnostics,
    collect_response_diagnostics,
)
from loom.models.retry import ModelRetryPolicy, call_with_model_retry
from loom.models.router import ModelRouter, ResponseValidator
from loom.prompts.assembler import PromptAssembler
from loom.state.task_state import Subtask
from loom.utils.concurrency import run_blocking_io
from loom.utils.tokens import estimate_tokens

if TYPE_CHECKING:
    from loom.processes.schema import ProcessDefinition

logger = logging.getLogger(__name__)
_COMPACTOR_EVENT_CONTEXT: contextvars.ContextVar[tuple[str, str] | None] = (
    contextvars.ContextVar("verification_compactor_event_context", default=None)
)
_VERIFICATION_CONTRADICTION_EVENT_TYPE = "verification_contradiction_detected"

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
    _TOOL_SUCCESS_POLICIES = frozenset({
        "all_tools_hard",
        "safety_integrity_only",
    })

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
        policy = getattr(process, "verification_policy", None)
        static_checks = getattr(policy, "static_checks", {})
        if not isinstance(static_checks, dict):
            return "all_tools_hard"
        raw = str(static_checks.get("tool_success_policy", "") or "").strip().lower()
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
        limits: VerifierLimitsConfig | None = None,
        compactor_limits: CompactorLimitsConfig | None = None,
        evidence_context_text_max_chars: int = 4000,
    ):
        self._router = model_router
        self._prompts = prompt_assembler
        self._validator = validator
        resolved_limits = limits or VerifierLimitsConfig()
        self._max_tool_args_chars = int(
            getattr(resolved_limits, "max_tool_args_chars", self._MAX_TOOL_ARGS_CHARS),
        )
        self._max_tool_status_chars = int(
            getattr(
                resolved_limits,
                "max_tool_status_chars",
                self._MAX_TOOL_STATUS_CHARS,
            ),
        )
        self._max_tool_calls_tokens = int(
            getattr(
                resolved_limits,
                "max_tool_calls_tokens",
                self._MAX_TOOL_CALLS_TOKENS,
            ),
        )
        self._max_verifier_prompt_tokens = int(
            getattr(
                resolved_limits,
                "max_verifier_prompt_tokens",
                self._MAX_VERIFIER_PROMPT_TOKENS,
            ),
        )
        self._max_result_summary_chars = int(
            getattr(
                resolved_limits,
                "max_result_summary_chars",
                self._MAX_RESULT_SUMMARY_CHARS,
            ),
        )
        self._compact_result_summary_chars = int(
            getattr(
                resolved_limits,
                "compact_result_summary_chars",
                self._COMPACT_RESULT_SUMMARY_CHARS,
            ),
        )
        self._max_evidence_section_chars = int(
            getattr(
                resolved_limits,
                "max_evidence_section_chars",
                self._MAX_EVIDENCE_SECTION_CHARS,
            ),
        )
        self._max_evidence_section_compact_chars = int(
            getattr(
                resolved_limits,
                "max_evidence_section_compact_chars",
                self._MAX_EVIDENCE_SECTION_COMPACT_CHARS,
            ),
        )
        self._max_artifact_section_chars = int(
            getattr(
                resolved_limits,
                "max_artifact_section_chars",
                self._MAX_ARTIFACT_SECTION_CHARS,
            ),
        )
        self._max_artifact_section_compact_chars = int(
            getattr(
                resolved_limits,
                "max_artifact_section_compact_chars",
                self._MAX_ARTIFACT_SECTION_COMPACT_CHARS,
            ),
        )
        self._max_tool_output_excerpt_chars = int(
            getattr(
                resolved_limits,
                "max_tool_output_excerpt_chars",
                self._MAX_TOOL_OUTPUT_EXCERPT_CHARS,
            ),
        )
        self._max_artifact_file_excerpt_chars = int(
            getattr(
                resolved_limits,
                "max_artifact_file_excerpt_chars",
                self._MAX_ARTIFACT_FILE_EXCERPT_CHARS,
            ),
        )
        self._evidence_context_text_max_chars = max(
            120,
            int(evidence_context_text_max_chars or 0),
        )
        comp_limits = compactor_limits or CompactorLimitsConfig()
        self._compactor = SemanticCompactor(
            model_router,
            model_event_hook=self._emit_compactor_model_event,
            role="compactor",
            tier=1,
            allow_role_fallback=True,
            max_chunk_chars=int(
                getattr(comp_limits, "max_chunk_chars", SemanticCompactor._MAX_CHUNK_CHARS),
            ),
            max_chunks_per_round=int(
                getattr(
                    comp_limits,
                    "max_chunks_per_round",
                    SemanticCompactor._MAX_CHUNKS_PER_ROUND,
                ),
            ),
            max_reduction_rounds=int(
                getattr(
                    comp_limits,
                    "max_reduction_rounds",
                    SemanticCompactor._MAX_REDUCTION_ROUNDS,
                ),
            ),
            min_compact_target_chars=int(
                getattr(
                    comp_limits,
                    "min_compact_target_chars",
                    SemanticCompactor._MIN_COMPACT_TARGET_CHARS,
                ),
            ),
            response_tokens_floor=int(
                getattr(
                    comp_limits,
                    "response_tokens_floor",
                    SemanticCompactor._RESPONSE_TOKENS_FLOOR,
                ),
            ),
            response_tokens_ratio=float(
                getattr(
                    comp_limits,
                    "response_tokens_ratio",
                    SemanticCompactor._RESPONSE_TOKENS_RATIO,
                ),
            ),
            response_tokens_buffer=int(
                getattr(
                    comp_limits,
                    "response_tokens_buffer",
                    SemanticCompactor._RESPONSE_TOKENS_BUFFER,
                ),
            ),
            json_headroom_chars_floor=int(
                getattr(
                    comp_limits,
                    "json_headroom_chars_floor",
                    SemanticCompactor._JSON_HEADROOM_CHARS_FLOOR,
                ),
            ),
            json_headroom_chars_ratio=float(
                getattr(
                    comp_limits,
                    "json_headroom_chars_ratio",
                    SemanticCompactor._JSON_HEADROOM_CHARS_RATIO,
                ),
            ),
            json_headroom_chars_cap=int(
                getattr(
                    comp_limits,
                    "json_headroom_chars_cap",
                    SemanticCompactor._JSON_HEADROOM_CHARS_CAP,
                ),
            ),
            chars_per_token_estimate=float(
                getattr(
                    comp_limits,
                    "chars_per_token_estimate",
                    SemanticCompactor._CHARS_PER_TOKEN_ESTIMATE,
                ),
            ),
            token_headroom=int(
                getattr(
                    comp_limits,
                    "token_headroom",
                    SemanticCompactor._TOKEN_HEADROOM,
                ),
            ),
            target_chars_ratio=float(
                getattr(
                    comp_limits,
                    "target_chars_ratio",
                    SemanticCompactor._TARGET_CHARS_RATIO,
                ),
            ),
        )
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
    _MAX_ARTIFACT_SECTION_CHARS = 2400
    _MAX_ARTIFACT_SECTION_COMPACT_CHARS = 1200
    _MAX_TOOL_OUTPUT_EXCERPT_CHARS = 700
    _MAX_ARTIFACT_FILE_EXCERPT_CHARS = 420
    _ADVISORY_TOOL_FAILURES = frozenset({"web_fetch", "web_fetch_html", "web_search"})
    _SOURCE_ATTRIBUTION_TOOLS = frozenset({"web_fetch", "web_fetch_html"})
    _TOOL_OUTPUT_EVIDENCE_TOOLS = frozenset({
        "read_file",
        "spreadsheet",
        "document_write",
    })
    _ARTIFACT_PREVIEW_SUFFIXES = frozenset({
        ".md",
        ".txt",
        ".csv",
        ".json",
        ".yaml",
        ".yml",
        ".xml",
        ".html",
        ".htm",
        ".rst",
    })

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
                max_chars=self._max_tool_args_chars,
            )
            result = getattr(tc, "result", None)
            if getattr(result, "success", False):
                status = "OK"
            else:
                error_text = str(getattr(result, "error", "") or "unknown error")
                compact_error = await self._compact_text(
                    error_text,
                    max_chars=self._max_tool_status_chars,
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
            line = f"- {tool_name}({args_text}) -> {status}"
            if getattr(result, "success", False):
                files_changed = [
                    str(item).strip()
                    for item in (getattr(result, "files_changed", []) or [])
                    if str(item).strip()
                ]
                if files_changed:
                    preview = ", ".join(files_changed[:4])
                    line += f"\n  files_changed: {preview}"
                excerpt = await self._tool_output_excerpt(
                    tool_name=tool_name,
                    args=getattr(tc, "args", {}),
                    result=result,
                )
                if excerpt:
                    indented_excerpt = self._indent_text_block(
                        excerpt,
                        prefix="    ",
                    )
                    line += f"\n  output_excerpt:\n{indented_excerpt}"
            lines.append(line)

        formatted = "\n".join(lines) if lines else "No tool calls."
        if estimate_tokens(formatted) > self._max_tool_calls_tokens:
            return await self._compact_text(
                formatted,
                max_chars=self._max_tool_calls_tokens * 4,
                label="verification tool-call history",
            )
        return formatted

    @staticmethod
    def _indent_text_block(text: str, *, prefix: str = "  ") -> str:
        if not text:
            return ""
        return "\n".join(f"{prefix}{line}" for line in str(text).splitlines())

    async def _tool_output_excerpt(
        self,
        *,
        tool_name: str,
        args: object,
        result: object,
    ) -> str:
        """Return bounded tool output excerpts that help semantic verification."""
        if tool_name not in self._TOOL_OUTPUT_EVIDENCE_TOOLS:
            return ""

        normalized_args = args if isinstance(args, dict) else {}
        if tool_name == "spreadsheet":
            operation = str(normalized_args.get("operation", "")).strip().lower()
            if operation not in {"read", "summary"}:
                return ""
        if tool_name == "document_write":
            parts: list[str] = []
            title = str(normalized_args.get("title", "")).strip()
            if title:
                parts.append(f"Title: {title}")
            raw_content = str(normalized_args.get("content", "")).strip()
            if raw_content:
                parts.append(raw_content)
            sections = normalized_args.get("sections", [])
            if isinstance(sections, list):
                for section in sections[:4]:
                    if not isinstance(section, dict):
                        continue
                    heading = str(section.get("heading", "")).strip()
                    body = str(section.get("body", "")).strip()
                    if heading:
                        parts.append(f"## {heading}")
                    if body:
                        parts.append(body)
            output = "\n\n".join(part for part in parts if part).strip()
            if not output:
                output = str(getattr(result, "output", "") or "").strip()
            if not output:
                return ""
            compacted = await self._compact_text(
                output,
                max_chars=self._max_tool_output_excerpt_chars,
                label="verification tool output excerpt (document_write)",
            )
            return compacted.strip()

        output = str(getattr(result, "output", "") or "").strip()
        if not output:
            return ""
        compacted = await self._compact_text(
            output,
            max_chars=self._max_tool_output_excerpt_chars,
            label=f"verification tool output excerpt ({tool_name})",
        )
        return compacted.strip()

    async def _build_artifact_content_section(
        self,
        *,
        workspace: Path | None,
        tool_calls: list | None,
        max_chars: int,
    ) -> str:
        """Build bounded excerpts from changed artifacts for semantic review."""
        if workspace is None or not tool_calls:
            return ""

        workspace_resolved = workspace.resolve()
        changed_files: list[tuple[str, Path]] = []
        seen: set[str] = set()

        for tc in tool_calls:
            result = getattr(tc, "result", None)
            if result is None or not getattr(result, "success", False):
                continue
            files_changed = getattr(result, "files_changed", []) or []
            for rel_path in files_changed:
                rel = str(rel_path or "").strip()
                if not rel or rel in seen:
                    continue
                seen.add(rel)
                candidate = (workspace / rel).resolve()
                try:
                    candidate.relative_to(workspace_resolved)
                except ValueError:
                    continue
                if not candidate.exists() or not candidate.is_file():
                    continue
                suffix = candidate.suffix.lower()
                if suffix and suffix not in self._ARTIFACT_PREVIEW_SUFFIXES:
                    continue
                changed_files.append((rel, candidate))

        if not changed_files:
            return ""

        lines = [
            "ARTIFACT CONTENT SNAPSHOT (for semantic review):",
            "- Excerpts from files changed in this subtask. Use these to judge output quality.",
        ]
        for rel, path in changed_files[:6]:
            try:
                content = path.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                continue
            if not content:
                continue
            excerpt = await self._compact_text(
                content,
                max_chars=self._max_artifact_file_excerpt_chars,
                label=f"verification artifact excerpt ({path.name})",
            )
            lines.append(f"- {rel}:")
            lines.append(self._indent_text_block(excerpt, prefix="    "))

        if len(lines) <= 2:
            return ""
        return self._hard_cap_text("\n".join(lines), max_chars=max_chars)

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

    def _extract_evidence_from_tool_calls(
        self,
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
                context_text_max_chars=self._evidence_context_text_max_chars,
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

    def _emit_compactor_model_event(self, payload: dict) -> None:
        """Bridge semantic-compactor model events into verifier model_invocation events."""
        context = _COMPACTOR_EVENT_CONTEXT.get()
        if not context:
            return
        task_id, subtask_id = context
        model_name = str(payload.get("model", "")).strip() or "unknown"
        phase = str(payload.get("phase", "")).strip() or "done"
        details = {
            key: value
            for key, value in payload.items()
            if key not in {"model", "phase"}
        }
        self._emit_model_event(
            task_id=task_id,
            subtask_id=subtask_id,
            model_name=model_name,
            phase=phase,
            details=details,
        )

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
    def _to_string_list(value: object) -> list[str]:
        if isinstance(value, list):
            items = [str(item or "").strip() for item in value]
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            items = re.split(r"[,\n;]+", text)
        else:
            return []
        normalized: list[str] = []
        for item in items:
            text = str(item or "").strip()
            if text and text not in normalized:
                normalized.append(text)
        return normalized

    @classmethod
    def _normalize_verifier_metadata(cls, metadata: dict[str, object]) -> dict[str, object]:
        normalized: dict[str, object] = {
            str(key).strip(): value
            for key, value in metadata.items()
            if str(key).strip()
        }
        lookup: dict[str, str] = {
            key.lower(): key for key in normalized
        }

        def value_for(field_name: str) -> object | None:
            key = lookup.get(field_name)
            if key is None:
                return None
            return normalized.get(key)

        remediation_required_raw = value_for("remediation_required")
        if remediation_required_raw is not None:
            remediation_required = cls._to_bool(remediation_required_raw)
            if remediation_required is None:
                remediation_required = bool(remediation_required_raw)
            normalized["remediation_required"] = remediation_required

        remediation_mode_raw = value_for("remediation_mode")
        if remediation_mode_raw is not None:
            normalized["remediation_mode"] = str(
                remediation_mode_raw or "",
            ).strip().lower()

        missing_targets_raw = value_for("missing_targets")
        if missing_targets_raw is not None:
            normalized["missing_targets"] = cls._to_string_list(missing_targets_raw)

        unverified_claim_count_raw = value_for("unverified_claim_count")
        if unverified_claim_count_raw is not None:
            unverified_claim_count = cls._to_int(unverified_claim_count_raw)
            if unverified_claim_count is not None:
                normalized["unverified_claim_count"] = max(0, unverified_claim_count)

        verified_claim_count_raw = value_for("verified_claim_count")
        if verified_claim_count_raw is not None:
            verified_claim_count = cls._to_int(verified_claim_count_raw)
            if verified_claim_count is not None:
                normalized["verified_claim_count"] = max(0, verified_claim_count)

        supporting_ratio_raw = value_for("supporting_ratio")
        if supporting_ratio_raw is not None:
            supporting_ratio = cls._to_confidence(supporting_ratio_raw)
            if supporting_ratio is not None:
                normalized["supporting_ratio"] = supporting_ratio

        return normalized

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
        metadata = LLMVerifier._normalize_verifier_metadata(metadata)
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
            metadata={**metadata, "issues": issues},
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
        *,
        task_id: str = "",
        subtask_id: str = "",
        origin: str = "verification.repair_assessment.complete",
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
        request_messages = [{"role": "user", "content": repair_prompt}]
        policy = ModelRetryPolicy()
        invocation_attempt = 0
        request_diag = collect_request_diagnostics(
            messages=request_messages,
            origin=origin,
        )

        async def _invoke_model():
            nonlocal invocation_attempt
            invocation_attempt += 1
            self._emit_model_event(
                task_id=task_id,
                subtask_id=subtask_id,
                model_name=getattr(model, "name", "unknown"),
                phase="start",
                details={
                    "operation": "complete",
                    "invocation_attempt": invocation_attempt,
                    "invocation_max_attempts": policy.max_attempts,
                    **request_diag.to_event_payload(),
                },
            )
            return await model.complete(request_messages)

        def _on_failure(
            attempt: int,
            max_attempts: int,
            error: BaseException,
            remaining: int,
        ) -> None:
            self._emit_model_event(
                task_id=task_id,
                subtask_id=subtask_id,
                model_name=getattr(model, "name", "unknown"),
                phase="done",
                details={
                    "operation": "complete",
                    "origin": request_diag.origin,
                    "invocation_attempt": attempt,
                    "invocation_max_attempts": max_attempts,
                    "retry_queue_remaining": remaining,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )

        try:
            response = await call_with_model_retry(
                _invoke_model,
                policy=policy,
                on_failure=_on_failure,
            )
        except Exception:
            return None
        self._emit_model_event(
            task_id=task_id,
            subtask_id=subtask_id,
            model_name=getattr(model, "name", "unknown"),
            phase="done",
            details={
                "operation": "complete",
                "origin": request_diag.origin,
                "invocation_attempt": invocation_attempt,
                "invocation_max_attempts": policy.max_attempts,
                **collect_response_diagnostics(response).to_event_payload(),
            },
        )
        return self._parse_verifier_response(response)

    async def _repair_assessment_with_alternate_model(
        self,
        raw_text: str,
        *,
        task_id: str = "",
        subtask_id: str = "",
        origin: str = "verification.repair_assessment.alternate_model.complete",
    ) -> VerificationResult | None:
        try:
            alternate = self._router.select(tier=2, role="verifier")
        except Exception:
            return None
        return await self._repair_assessment_with_model(
            alternate,
            raw_text,
            task_id=task_id,
            subtask_id=subtask_id,
            origin=origin,
        )

    async def _invoke_and_parse(
        self,
        model,
        prompt: str,
        *,
        task_id: str = "",
        subtask_id: str = "",
        origin: str = "verification.invoke_and_parse.complete",
    ) -> VerificationResult:
        request_messages = [{"role": "user", "content": prompt}]
        policy = ModelRetryPolicy()
        invocation_attempt = 0
        request_diag = collect_request_diagnostics(
            messages=request_messages,
            origin=origin,
        )

        async def _invoke_model():
            nonlocal invocation_attempt
            invocation_attempt += 1
            self._emit_model_event(
                task_id=task_id,
                subtask_id=subtask_id,
                model_name=getattr(model, "name", "unknown"),
                phase="start",
                details={
                    "operation": "complete",
                    "invocation_attempt": invocation_attempt,
                    "invocation_max_attempts": policy.max_attempts,
                    **request_diag.to_event_payload(),
                },
            )
            return await model.complete(request_messages)

        def _on_failure(
            attempt: int,
            max_attempts: int,
            error: BaseException,
            remaining: int,
        ) -> None:
            self._emit_model_event(
                task_id=task_id,
                subtask_id=subtask_id,
                model_name=getattr(model, "name", "unknown"),
                phase="done",
                details={
                    "operation": "complete",
                    "origin": request_diag.origin,
                    "invocation_attempt": attempt,
                    "invocation_max_attempts": max_attempts,
                    "retry_queue_remaining": remaining,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )

        response = await call_with_model_retry(
            _invoke_model,
            policy=policy,
            on_failure=_on_failure,
        )
        self._emit_model_event(
            task_id=task_id,
            subtask_id=subtask_id,
            model_name=getattr(model, "name", "unknown"),
            phase="done",
            details={
                "operation": "complete",
                "origin": request_diag.origin,
                "invocation_attempt": invocation_attempt,
                "invocation_max_attempts": policy.max_attempts,
                **collect_response_diagnostics(response).to_event_payload(),
            },
        )
        parsed = self._parse_verifier_response(response)
        if parsed is not None:
            return parsed

        if not bool(getattr(self._config, "strict_output_protocol", True)):
            return self._inconclusive_result()

        repaired = await self._repair_assessment_with_model(
            model,
            response.text or "",
            task_id=task_id,
            subtask_id=subtask_id,
            origin="verification.repair_assessment.same_model.complete",
        )
        if repaired is not None:
            repaired.metadata = dict(repaired.metadata)
            repaired.metadata["parser_stage"] = "repair_same_model"
            return repaired

        repaired_alt = await self._repair_assessment_with_alternate_model(
            response.text or "",
            task_id=task_id,
            subtask_id=subtask_id,
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
        compactor_context_token = _COMPACTOR_EVENT_CONTEXT.set((task_id, subtask.id))
        try:
            model = self._router.select(tier=1, role="verifier")
        except Exception as e:
            logger.warning("Verifier model not available: %s", e)
            _COMPACTOR_EVENT_CONTEXT.reset(compactor_context_token)
            return VerificationResult(
                tier=0, passed=True, confidence=0.5,
                feedback="Verification skipped: verifier model not configured",
                outcome="pass_with_warnings",
                reason_code="infra_verifier_error",
                severity_class="infra",
            )

        summary_for_prompt = await self._compact_text(
            result_summary,
            max_chars=self._max_result_summary_chars,
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
            max_chars=self._max_evidence_section_chars,
        )
        artifact_section = await self._build_artifact_content_section(
            workspace=workspace,
            tool_calls=tool_calls,
            max_chars=self._max_artifact_section_chars,
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
        if artifact_section:
            prompt = prompt + "\n\n" + artifact_section
        if estimate_tokens(prompt) > self._max_verifier_prompt_tokens:
            summary_for_prompt = await self._compact_text(
                summary_for_prompt,
                max_chars=self._compact_result_summary_chars,
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
                        self._max_evidence_section_compact_chars,
                    )
                )
            if artifact_section:
                prompt = (
                    prompt
                    + "\n\n"
                    + self._hard_cap_text(
                        artifact_section,
                        self._max_artifact_section_compact_chars,
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
            first_result = await self._invoke_and_parse(
                model,
                prompt,
                task_id=task_id,
                subtask_id=subtask.id,
                origin="verification.tier2.invoke.complete",
            )
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
            _COMPACTOR_EVENT_CONTEXT.reset(compactor_context_token)
            return first_result
        except Exception as e:
            logger.warning("Verifier raised exception: %s", e)
            compact_tool_calls = await self._format_tool_calls_compact(tool_calls)
            compact_prompt = self._build_prompt(
                subtask,
                await self._compact_text(
                    summary_for_prompt,
                    max_chars=self._compact_result_summary_chars,
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
                        self._max_evidence_section_compact_chars,
                    )
                )
            if artifact_section:
                compact_prompt = (
                    compact_prompt
                    + "\n\n"
                    + self._hard_cap_text(
                        artifact_section,
                        self._max_artifact_section_compact_chars,
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
                retry_result = await self._invoke_and_parse(
                    model,
                    compact_prompt,
                    task_id=task_id,
                    subtask_id=subtask.id,
                    origin="verification.tier2.retry.invoke.complete",
                )
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
                _COMPACTOR_EVENT_CONTEXT.reset(compactor_context_token)
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
                _COMPACTOR_EVENT_CONTEXT.reset(compactor_context_token)
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

    _PLACEHOLDER_CLAIM_REASON_CODES = frozenset({
        "incomplete_deliverable_placeholder",
        "incomplete_deliverable_content",
    })
    _PLACEHOLDER_MARKER_PATTERN = re.compile(
        r"\[(?:TBD|TODO|INSERT|PLACEHOLDER|MISSING)\]|\bTODO\b|\bPLACEHOLDER\b",
        flags=re.IGNORECASE,
    )
    _DEFAULT_CONTRADICTION_SCAN_ALLOWED_SUFFIXES = (
        ".md",
        ".txt",
        ".rst",
        ".csv",
        ".tsv",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".xml",
        ".html",
        ".htm",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".sql",
        ".sh",
    )
    _CONTRADICTION_SCAN_EXCLUDED_DIRS = frozenset({
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        ".tox",
        ".idea",
        ".vscode",
        ".next",
        "dist",
        "build",
        "target",
    })

    def __init__(
        self,
        model_router: ModelRouter,
        prompt_assembler: PromptAssembler,
        config: VerificationConfig,
        limits: VerifierLimitsConfig | None = None,
        compactor_limits: CompactorLimitsConfig | None = None,
        evidence_context_text_max_chars: int = 4000,
        process: ProcessDefinition | None = None,
        event_bus: EventBus | None = None,
    ):
        self._config = config
        self._event_bus = event_bus
        self._process = process
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
            limits=limits,
            compactor_limits=compactor_limits,
            evidence_context_text_max_chars=evidence_context_text_max_chars,
        )
        self._tier3 = VotingVerifier(self._tier2, config.tier3_vote_count)

    def _expected_deliverables_for_subtask(self, subtask: Subtask) -> list[str]:
        process = self._process
        if process is None:
            return []
        deliverables = process.get_deliverables()
        if not deliverables:
            return []
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
        return []

    @staticmethod
    def _files_changed(tool_calls: list) -> list[str]:
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

    @staticmethod
    def _to_nonempty_int(
        value: object,
        default: int,
        *,
        minimum: int,
        maximum: int,
    ) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        parsed = max(minimum, parsed)
        parsed = min(maximum, parsed)
        return parsed

    @staticmethod
    def _to_bool(raw: object, default: bool) -> bool:
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

    @classmethod
    def _normalized_scan_suffixes(cls, raw: object) -> tuple[str, ...]:
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
            return cls._DEFAULT_CONTRADICTION_SCAN_ALLOWED_SUFFIXES
        return tuple(normalized)

    @classmethod
    def _normalize_candidate_path(
        cls,
        *,
        workspace: Path | None,
        raw_path: object,
    ) -> str | None:
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

    @classmethod
    def _normalize_candidate_bucket(
        cls,
        *,
        workspace: Path | None,
        raw_paths: list[str],
    ) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in raw_paths:
            rel_path = cls._normalize_candidate_path(workspace=workspace, raw_path=item)
            if not rel_path or rel_path in seen:
                continue
            seen.add(rel_path)
            normalized.append(rel_path)
        return normalized

    @staticmethod
    def _evidence_artifact_paths(evidence_records: list[dict] | None) -> list[str]:
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

    @classmethod
    def _build_placeholder_scan_candidates(
        cls,
        *,
        subtask: Subtask,
        workspace: Path | None,
        tool_calls: list,
        evidence_tool_calls: list | None,
        evidence_records: list[dict] | None,
        expected_deliverables: list[str],
    ) -> dict[str, object]:
        canonical_paths = cls._normalize_candidate_bucket(
            workspace=workspace,
            raw_paths=expected_deliverables,
        )
        current_paths = cls._normalize_candidate_bucket(
            workspace=workspace,
            raw_paths=cls._files_changed(tool_calls),
        )
        prior_paths = cls._normalize_candidate_bucket(
            workspace=workspace,
            raw_paths=cls._files_changed(evidence_tool_calls or []),
        )
        evidence_paths = cls._normalize_candidate_bucket(
            workspace=workspace,
            raw_paths=cls._evidence_artifact_paths(evidence_records),
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

    @classmethod
    def _path_has_symlink_component(
        cls,
        *,
        workspace: Path,
        path: Path,
    ) -> bool:
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

    @classmethod
    def _is_placeholder_claim_failure(cls, result: VerificationResult) -> bool:
        if result.passed:
            return False
        reason_code = str(result.reason_code or "").strip().lower()
        if reason_code in cls._PLACEHOLDER_CLAIM_REASON_CODES:
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
        return bool(cls._PLACEHOLDER_MARKER_PATTERN.search(haystack))

    def _scan_placeholder_markers(
        self,
        *,
        workspace: Path | None,
        candidate_data: dict[str, object],
    ) -> dict[str, object]:
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

        max_files = self._to_nonempty_int(
            getattr(self._config, "contradiction_scan_max_files", 80),
            80,
            minimum=1,
            maximum=1000,
        )
        max_total_bytes = self._to_nonempty_int(
            getattr(self._config, "contradiction_scan_max_total_bytes", 2_500_000),
            2_500_000,
            minimum=1_024,
            maximum=50_000_000,
        )
        max_file_bytes = self._to_nonempty_int(
            getattr(self._config, "contradiction_scan_max_file_bytes", 300_000),
            300_000,
            minimum=1,
            maximum=10_000_000,
        )
        max_file_bytes = min(max_file_bytes, max_total_bytes)
        min_files_for_sufficiency = self._to_nonempty_int(
            getattr(
                self._config,
                "contradiction_scan_min_files_for_sufficiency",
                2,
            ),
            2,
            minimum=1,
            maximum=100,
        )
        strict_coverage = self._to_bool(
            getattr(self._config, "contradiction_guard_strict_coverage", True),
            True,
        )
        allowed_suffixes = set(
            self._normalized_scan_suffixes(
                getattr(self._config, "contradiction_scan_allowed_suffixes", ()),
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
            if self._path_has_symlink_component(workspace=workspace_root, path=fpath):
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
            if self._PLACEHOLDER_MARKER_PATTERN.search(content):
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
                        or name in self._CONTRADICTION_SCAN_EXCLUDED_DIRS
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

    async def _apply_placeholder_contradiction_guard(
        self,
        *,
        subtask: Subtask,
        result: VerificationResult,
        workspace: Path | None,
        tool_calls: list,
        evidence_tool_calls: list | None = None,
        evidence_records: list[dict] | None = None,
    ) -> VerificationResult:
        if not bool(getattr(self._config, "contradiction_guard_enabled", True)):
            return result
        if result.passed:
            return result
        if str(result.severity_class or "").strip().lower() == "hard_invariant":
            return result
        if not self._is_placeholder_claim_failure(result):
            return result

        expected_deliverables = self._expected_deliverables_for_subtask(subtask)
        candidate_data = self._build_placeholder_scan_candidates(
            subtask=subtask,
            workspace=workspace,
            tool_calls=tool_calls,
            evidence_tool_calls=evidence_tool_calls,
            evidence_records=evidence_records,
            expected_deliverables=expected_deliverables,
        )
        raw_timeout = getattr(self._config, "contradiction_scan_timeout_seconds", 8.0)
        scan_timeout_seconds = max(1.0, min(30.0, float(raw_timeout or 8.0)))
        try:
            scan = await asyncio.wait_for(
                run_blocking_io(
                    self._scan_placeholder_markers,
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

        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        scan = metadata.get("deterministic_placeholder_scan", {})
        if isinstance(scan, dict):
            scan_dict = scan if isinstance(scan, dict) else {}
            candidate_source_counts = scan_dict.get("candidate_source_counts", {})
            if not isinstance(candidate_source_counts, dict):
                candidate_source_counts = {}
            contradiction_downgraded = bool(metadata.get("contradiction_downgraded", False))
            contradiction_detected_no_downgrade = bool(
                metadata.get("contradiction_detected_no_downgrade", False),
            )
            self._event_bus.emit(Event(
                event_type=_VERIFICATION_CONTRADICTION_EVENT_TYPE,
                task_id=task_id,
                data={
                    "subtask_id": subtask_id,
                    "tier": result.tier,
                    "reason_code": result.reason_code,
                    "contradicted_reason_code": str(
                        metadata.get("contradicted_reason_code", ""),
                    ),
                    "scanned_file_count": int(
                        scan_dict.get("scanned_file_count", 0) or 0,
                    ),
                    "scanned_total_bytes": int(
                        scan_dict.get("scanned_total_bytes", 0) or 0,
                    ),
                    "matched_file_count": int(
                        scan_dict.get("matched_file_count", 0) or 0,
                    ),
                    "scan_mode": str(
                        scan_dict.get("scan_mode", "targeted_only") or "targeted_only",
                    ),
                    "coverage_sufficient": bool(
                        scan_dict.get("coverage_sufficient", False),
                    ),
                    "candidate_source_counts": candidate_source_counts,
                    "coverage_insufficient_reason": str(
                        scan_dict.get("coverage_insufficient_reason", "") or "",
                    ),
                    "contradiction_downgrade_count": 1 if contradiction_downgraded else 0,
                    "contradiction_detected_no_downgrade_count": (
                        1 if contradiction_detected_no_downgrade else 0
                    ),
                    "cap_exhaustion_count": (
                        1 if bool(scan_dict.get("cap_exhausted", False)) else 0
                    ),
                },
            ))

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

    @staticmethod
    def _fallback_from_tier1_for_inconclusive_tier2(
        *,
        tier1_result: VerificationResult | None,
        tier2_result: VerificationResult,
    ) -> VerificationResult | None:
        if tier1_result is None or not tier1_result.passed:
            return None
        if (
            isinstance(tier2_result.metadata, dict)
            and bool(tier2_result.metadata.get("contradiction_detected", False))
        ):
            return None
        reason = str(tier2_result.reason_code or "").strip().lower()
        severity = str(tier2_result.severity_class or "").strip().lower()
        if reason != "parse_inconclusive" and severity != "inconclusive":
            return None
        note = (
            "Tier-2 verifier output was inconclusive; accepting Tier-1 checks "
            "with warning."
        )
        merged_feedback = "\n".join(
            part for part in [tier2_result.feedback or "", note] if part
        )
        metadata: dict[str, object] = {
            "fallback": "tier1_due_to_tier2_parse_inconclusive",
            "tier2_reason_code": reason or str(tier2_result.reason_code or ""),
            "tier2_outcome": tier2_result.outcome,
        }
        if isinstance(tier2_result.metadata, dict) and tier2_result.metadata:
            metadata["tier2"] = dict(tier2_result.metadata)
        return VerificationResult(
            tier=2,
            passed=True,
            confidence=max(0.5, float(tier1_result.confidence)),
            checks=list(tier1_result.checks or []),
            feedback=merged_feedback or None,
            outcome="pass_with_warnings",
            reason_code="infra_verifier_error",
            severity_class="infra",
            metadata=metadata,
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
        t2 = await self._apply_placeholder_contradiction_guard(
            subtask=subtask,
            result=t2,
            workspace=workspace,
            tool_calls=tool_calls,
            evidence_tool_calls=evidence_tool_calls,
            evidence_records=evidence_records,
        )
        results.append(t2)
        t1_result = next((item for item in results if item.tier == 1), None)
        inconclusive_fallback = self._fallback_from_tier1_for_inconclusive_tier2(
            tier1_result=t1_result,
            tier2_result=t2,
        )
        if inconclusive_fallback is not None:
            self._emit_instrumentation_events(
                task_id=task_id,
                subtask_id=subtask.id,
                result=inconclusive_fallback,
            )
            self._emit_outcome_event(
                task_id=task_id,
                subtask_id=subtask.id,
                result=inconclusive_fallback,
            )
            return inconclusive_fallback
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
