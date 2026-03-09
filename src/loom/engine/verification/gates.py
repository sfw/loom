"""Verification gate orchestration and voting flow."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING

from loom.config import CompactorLimitsConfig, VerificationConfig, VerifierLimitsConfig
from loom.events.bus import EventBus
from loom.models.router import ModelRouter, ResponseValidator
from loom.prompts.assembler import PromptAssembler
from loom.state.task_state import Subtask

from . import claims as verification_claims
from . import events as verification_events
from . import placeholder_guard as verification_placeholder_guard
from . import policy as verification_policy
from .tier1 import DeterministicVerifier
from .tier2 import LLMVerifier
from .types import Check, VerificationResult

if TYPE_CHECKING:
    from loom.processes.schema import ProcessDefinition

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
        return verification_placeholder_guard.expected_deliverables_for_subtask(
            self,
            subtask,
        )

    @staticmethod
    def _files_changed(tool_calls: list) -> list[str]:
        return verification_placeholder_guard.files_changed(tool_calls)

    @staticmethod
    def _to_nonempty_int(
        value: object,
        default: int,
        *,
        minimum: int,
        maximum: int,
    ) -> int:
        return verification_placeholder_guard.to_nonempty_int(
            value,
            default,
            minimum=minimum,
            maximum=maximum,
        )

    @staticmethod
    def _to_bool(raw: object, default: bool) -> bool:
        return verification_placeholder_guard.to_bool(raw, default)

    @classmethod
    def _normalized_scan_suffixes(cls, raw: object) -> tuple[str, ...]:
        return verification_placeholder_guard.normalized_scan_suffixes(cls, raw)

    @classmethod
    def _normalize_candidate_path(
        cls,
        *,
        workspace: Path | None,
        raw_path: object,
    ) -> str | None:
        return verification_placeholder_guard.normalize_candidate_path(
            workspace=workspace,
            raw_path=raw_path,
        )

    @classmethod
    def _normalize_candidate_bucket(
        cls,
        *,
        workspace: Path | None,
        raw_paths: list[str],
    ) -> list[str]:
        return verification_placeholder_guard.normalize_candidate_bucket(
            workspace=workspace,
            raw_paths=raw_paths,
        )

    @staticmethod
    def _evidence_artifact_paths(evidence_records: list[dict] | None) -> list[str]:
        return verification_placeholder_guard.evidence_artifact_paths(evidence_records)

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
        return verification_placeholder_guard.build_placeholder_scan_candidates(
            cls,
            subtask=subtask,
            workspace=workspace,
            tool_calls=tool_calls,
            evidence_tool_calls=evidence_tool_calls,
            evidence_records=evidence_records,
            expected_deliverables=expected_deliverables,
        )

    @classmethod
    def _path_has_symlink_component(
        cls,
        *,
        workspace: Path,
        path: Path,
    ) -> bool:
        return verification_placeholder_guard.path_has_symlink_component(
            workspace=workspace,
            path=path,
        )

    @classmethod
    def _is_placeholder_claim_failure(cls, result: VerificationResult) -> bool:
        return verification_placeholder_guard.is_placeholder_claim_failure(cls, result)

    def _scan_placeholder_markers(
        self,
        *,
        workspace: Path | None,
        candidate_data: dict[str, object],
    ) -> dict[str, object]:
        return verification_placeholder_guard.scan_placeholder_markers(
            self,
            workspace=workspace,
            candidate_data=candidate_data,
        )

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
        return await verification_placeholder_guard.apply_placeholder_contradiction_guard(
            self,
            subtask=subtask,
            result=result,
            workspace=workspace,
            tool_calls=tool_calls,
            evidence_tool_calls=evidence_tool_calls,
            evidence_records=evidence_records,
        )

    @staticmethod
    def _normalize_claim_type(text: str) -> str:
        return verification_claims.normalize_claim_type(text)

    @staticmethod
    def _claim_id(text: str) -> str:
        return verification_claims.claim_id(text)

    @staticmethod
    def _claim_status_from_fact_verdict(verdict: str) -> tuple[str, str]:
        return verification_claims.claim_status_from_fact_verdict(verdict)

    @staticmethod
    def _critical_claim_types(validity_contract: dict[str, object]) -> set[str]:
        return verification_claims.critical_claim_types(validity_contract)

    def _extract_claim_lifecycle(
        self,
        *,
        tool_calls: list,
        result: VerificationResult,
        validity_contract: dict[str, object],
    ) -> list[dict[str, object]]:
        return verification_claims.extract_claim_lifecycle(
            self,
            tool_calls=tool_calls,
            result=result,
            validity_contract=validity_contract,
        )

    @staticmethod
    def _claim_counts(claims: list[dict[str, object]]) -> dict[str, int]:
        return verification_claims.claim_counts(claims)

    def _attach_claim_lifecycle(
        self,
        *,
        task_id: str,
        subtask_id: str,
        result: VerificationResult,
        tool_calls: list,
        validity_contract: dict[str, object],
    ) -> VerificationResult:
        return verification_claims.attach_claim_lifecycle(
            self,
            task_id=task_id,
            subtask_id=subtask_id,
            result=result,
            tool_calls=tool_calls,
            validity_contract=validity_contract,
        )

    def _emit_outcome_event(
        self,
        *,
        task_id: str,
        subtask_id: str,
        result: VerificationResult,
    ) -> None:
        verification_events.emit_verification_outcome(
            self._event_bus,
            task_id=task_id,
            subtask_id=subtask_id,
            result=result,
        )

    def _emit_verification_started(
        self,
        *,
        task_id: str,
        subtask_id: str,
        target_tier: int,
    ) -> None:
        verification_events.emit_verification_started(
            self._event_bus,
            task_id=task_id,
            subtask_id=subtask_id,
            target_tier=target_tier,
        )

    def _emit_verification_terminal(
        self,
        *,
        task_id: str,
        subtask_id: str,
        result: VerificationResult,
    ) -> None:
        verification_events.emit_verification_terminal(
            self._event_bus,
            task_id=task_id,
            subtask_id=subtask_id,
            result=result,
        )

    def _emit_placeholder_findings_event(
        self,
        *,
        task_id: str,
        subtask_id: str,
        result: VerificationResult,
    ) -> None:
        verification_events.emit_placeholder_findings_extracted(
            self._event_bus,
            task_id=task_id,
            subtask_id=subtask_id,
            result=result,
        )

    @staticmethod
    def classify_shadow_diff(
        legacy_result: VerificationResult,
        result: VerificationResult,
    ) -> str:
        return verification_policy.classify_shadow_diff(legacy_result, result)

    def _emit_instrumentation_events(
        self,
        *,
        task_id: str,
        subtask_id: str,
        result: VerificationResult,
        legacy_result: VerificationResult | None = None,
    ) -> None:
        verification_events.emit_instrumentation_events(
            self._event_bus,
            task_id=task_id,
            subtask_id=subtask_id,
            result=result,
            legacy_result=legacy_result,
            classify_shadow_diff_fn=self.classify_shadow_diff,
        )

    @staticmethod
    def _aggregate_non_failing(results: list[VerificationResult]) -> VerificationResult:
        return verification_policy.aggregate_non_failing(results)

    @staticmethod
    def _legacy_result_from_tiers(results: list[VerificationResult]) -> VerificationResult:
        return verification_policy.legacy_result_from_tiers(results)

    @staticmethod
    def _fallback_from_tier1_for_inconclusive_tier2(
        *,
        tier1_result: VerificationResult | None,
        tier2_result: VerificationResult,
    ) -> VerificationResult | None:
        return verification_policy.fallback_from_tier1_for_inconclusive_tier2(
            tier1_result=tier1_result,
            tier2_result=tier2_result,
        )

    async def verify(
        self,
        subtask: Subtask,
        result_summary: str,
        tool_calls: list,
        workspace: Path | None,
        evidence_tool_calls: list | None = None,
        evidence_records: list[dict] | None = None,
        retry_writable_deliverables: list[str] | None = None,
        validity_contract: dict[str, object] | None = None,
        tier: int = 1,
        task_id: str = "",
    ) -> VerificationResult:
        """Run verification up to the specified tier.

        Each tier must pass before proceeding to the next.
        """
        policy_enabled = bool(getattr(self._config, "policy_engine_enabled", True))
        results: list[VerificationResult] = []
        effective_validity_contract: dict[str, object] = {}
        if isinstance(validity_contract, dict) and validity_contract:
            effective_validity_contract = dict(validity_contract)
        elif self._process is not None:
            resolver = getattr(self._process, "resolve_validity_contract_for_phase", None)
            if callable(resolver):
                phase_hint = str(getattr(subtask, "phase_id", "") or "").strip() or subtask.id
                try:
                    resolved = resolver(phase_hint, is_synthesis=bool(subtask.is_synthesis))
                except TypeError:
                    resolved = resolver(phase_hint)
                if isinstance(resolved, dict):
                    effective_validity_contract = dict(resolved)

        def finalize(result: VerificationResult) -> VerificationResult:
            return self._attach_claim_lifecycle(
                task_id=task_id,
                subtask_id=subtask.id,
                result=result,
                tool_calls=tool_calls,
                validity_contract=effective_validity_contract,
            )

        self._emit_verification_started(
            task_id=task_id,
            subtask_id=subtask.id,
            target_tier=max(1, int(tier or 1)),
        )

        # Tier 1 always runs when enabled.
        if self._config.tier1_enabled:
            t1 = await self._tier1.verify(
                subtask,
                result_summary,
                tool_calls,
                evidence_tool_calls=evidence_tool_calls,
                evidence_records=evidence_records,
                workspace=workspace,
                retry_writable_deliverables=retry_writable_deliverables,
            )
            results.append(t1)
            self._emit_placeholder_findings_event(
                task_id=task_id,
                subtask_id=subtask.id,
                result=t1,
            )
            if not t1.passed:
                t1 = finalize(t1)
                self._emit_outcome_event(
                    task_id=task_id,
                    subtask_id=subtask.id,
                    result=t1,
                )
                self._emit_verification_terminal(
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
            result = finalize(result)
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
            self._emit_verification_terminal(
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
            inconclusive_fallback = finalize(inconclusive_fallback)
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
            self._emit_verification_terminal(
                task_id=task_id,
                subtask_id=subtask.id,
                result=inconclusive_fallback,
            )
            return inconclusive_fallback
        if not t2.passed:
            t2 = finalize(t2)
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
            self._emit_verification_terminal(
                task_id=task_id,
                subtask_id=subtask.id,
                result=t2,
            )
            return t2

        if tier < 3 or not self._config.tier3_enabled:
            result = t2 if not policy_enabled else self._aggregate_non_failing(results)
            result = finalize(result)
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
            self._emit_verification_terminal(
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
            t3 = finalize(t3)
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
            self._emit_verification_terminal(
                task_id=task_id,
                subtask_id=subtask.id,
                result=t3,
            )
            return t3

        result = t3 if not policy_enabled else self._aggregate_non_failing(results)
        result = finalize(result)
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
        self._emit_verification_terminal(
            task_id=task_id,
            subtask_id=subtask.id,
            result=result,
        )
        return result
