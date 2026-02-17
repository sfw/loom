"""Three-tier verification gates for subtask outputs.

Tier 1: Deterministic checks (free, instant, no LLM)
Tier 2: Independent LLM verification (fresh context, different model)
Tier 3: Voting verification (N independent checks, majority agreement)

The model that performed the work never checks its own output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loom.config import VerificationConfig
from loom.models.router import ModelRouter, ResponseValidator
from loom.prompts.assembler import PromptAssembler
from loom.state.task_state import Subtask

if TYPE_CHECKING:
    from loom.processes.schema import ProcessDefinition

logger = logging.getLogger(__name__)


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

    _ADVISORY_TOOL_FAILURES = frozenset({"web_fetch", "web_search"})
    _ADVISORY_HTTP_STATUS = frozenset({403, 408, 425, 429, 500, 502, 503, 504})

    def __init__(
        self,
        process: ProcessDefinition | None = None,
    ):
        self._process = process

    async def verify(
        self,
        subtask: Subtask,
        result_summary: str,
        tool_calls: list,
        workspace: Path | None,
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
            for rule in self._process.regex_rules():
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
                passed = not found
                # Warning-severity rules record the match but don't fail
                is_failure = not passed and rule.severity == "error"
                checks.append(Check(
                    name=f"process_rule_{rule.name}",
                    passed=not is_failure,
                    detail=(
                        f"Rule '{rule.name}' matched: {rule.description}"
                        if found else None
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

        all_passed = all(c.passed for c in checks) if checks else True
        return VerificationResult(
            tier=1,
            passed=all_passed,
            checks=checks,
            feedback=self._build_feedback(checks) if not all_passed else None,
        )

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

        advisory_codes = "|".join(
            str(code) for code in sorted(self._ADVISORY_HTTP_STATUS)
        )
        if re.search(rf"\b(?:{advisory_codes})\b", text):
            return True

        advisory_markers = (
            "timeout",
            "timed out",
            "connection failed",
            "temporarily unavailable",
            "rate limit",
            "rate-limited",
            "response too large",
        )
        return any(marker in text for marker in advisory_markers)

    @staticmethod
    def _build_feedback(checks: list[Check]) -> str:
        """Build actionable feedback from failed checks."""
        failed = [c for c in checks if not c.passed]
        lines = ["Verification failed:"]
        for c in failed:
            detail = f" — {c.detail}" if c.detail else ""
            lines.append(f"  - {c.name}{detail}")
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
    ):
        self._router = model_router
        self._prompts = prompt_assembler
        self._validator = validator

    async def verify(
        self,
        subtask: Subtask,
        result_summary: str,
        tool_calls: list,
        workspace: Path | None,
    ) -> VerificationResult:
        try:
            model = self._router.select(tier=1, role="verifier")
        except Exception as e:
            logger.warning("Verifier model not available: %s", e)
            return VerificationResult(
                tier=0, passed=True, confidence=0.5,
                feedback="Verification skipped: verifier model not configured",
            )

        # Format tool calls
        tool_lines = []
        for tc in tool_calls:
            status = "OK" if tc.result.success else f"FAILED: {tc.result.error}"
            tool_lines.append(f"- {tc.tool}({json.dumps(tc.args)}) → {status}")
        tool_calls_formatted = "\n".join(tool_lines) if tool_lines else "No tool calls."

        prompt = self._prompts.build_verifier_prompt(
            subtask=subtask,
            result_summary=result_summary,
            tool_calls_formatted=tool_calls_formatted,
        )

        try:
            response = await model.complete([{"role": "user", "content": prompt}])
            validation = self._validator.validate_json_response(
                response, expected_keys=["passed"]
            )

            if not validation.valid or validation.parsed is None:
                # Can't parse verifier output — fail to be safe
                return VerificationResult(
                    tier=2, passed=False, confidence=0.3,
                    feedback="Verification inconclusive: could not parse verifier output.",
                )

            assessment = validation.parsed
            return VerificationResult(
                tier=2,
                passed=bool(assessment.get("passed", True)),
                confidence=float(assessment.get("confidence", 0.5)),
                checks=[Check(
                    name="llm_assessment",
                    passed=bool(assessment.get("passed", True)),
                    detail="; ".join(assessment.get("issues", [])),
                )],
                feedback=assessment.get("feedback") or assessment.get("suggestion"),
            )
        except Exception as e:
            logger.warning("Verifier raised exception: %s", e)
            return VerificationResult(
                tier=2, passed=False, confidence=0.3,
                feedback="Verification inconclusive: verifier raised an exception.",
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
    ) -> VerificationResult:
        tasks = [
            self._llm_verifier.verify(subtask, result_summary, tool_calls, workspace)
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
        )


class VerificationGates:
    """Orchestrates the three-tier verification pipeline."""

    def __init__(
        self,
        model_router: ModelRouter,
        prompt_assembler: PromptAssembler,
        config: VerificationConfig,
        process: ProcessDefinition | None = None,
    ):
        self._config = config
        validator = ResponseValidator()
        self._tier1 = DeterministicVerifier(process=process)
        self._tier2 = LLMVerifier(model_router, prompt_assembler, validator)
        self._tier3 = VotingVerifier(self._tier2, config.tier3_vote_count)

    async def verify(
        self,
        subtask: Subtask,
        result_summary: str,
        tool_calls: list,
        workspace: Path | None,
        tier: int = 1,
    ) -> VerificationResult:
        """Run verification up to the specified tier.

        Each tier must pass before proceeding to the next.
        """
        # Tier 1 always runs
        if self._config.tier1_enabled:
            t1 = await self._tier1.verify(subtask, result_summary, tool_calls, workspace)
            if not t1.passed:
                return t1

        if tier < 2 or not self._config.tier2_enabled:
            # Tier 1 either passed or was skipped; no higher-tier check requested
            return VerificationResult(
                tier=1,
                passed=True,
                confidence=0.7 if self._config.tier1_enabled else 0.5,
            )

        # Tier 2: independent LLM check
        t2 = await self._tier2.verify(subtask, result_summary, tool_calls, workspace)
        if not t2.passed:
            return t2

        if tier < 3 or not self._config.tier3_enabled:
            return t2

        # Tier 3: voting
        t3 = await self._tier3.verify(subtask, result_summary, tool_calls, workspace)
        return t3
