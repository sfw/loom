"""Process definition schema, loading, and validation.

A process definition is the unit of domain specialization. It can be
a single YAML file or a directory (package) containing a ``process.yaml``
plus optional bundled tools, templates, and examples.

The engine remains fully generic — the process definition injects domain
intelligence into existing prompt templates via the PromptAssembler.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from loom.engine.verification_helpers import (
    get_verification_helper,
    verification_helper_is_bound,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ProcessNotFoundError(Exception):
    """Raised when a process definition cannot be found."""

    def __init__(self, name: str, available: list[str] | None = None):
        available = available or []
        msg = f"Process not found: {name!r}"
        if available:
            msg += f". Available: {', '.join(sorted(available))}"
        super().__init__(msg)
        self.name = name
        self.available = available


class ProcessValidationError(Exception):
    """Raised when a process definition fails validation."""

    def __init__(self, errors: list[str], path: Path | None = None):
        prefix = f"{path}: " if path else ""
        msg = f"{prefix}{len(errors)} validation error(s):\n"
        msg += "\n".join(f"  - {e}" for e in errors)
        super().__init__(msg)
        self.errors = errors
        self.path = path


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class IterationBudget:
    """Bounded per-phase budget for gate-driven iteration loops."""

    max_wall_clock_seconds: int = 0
    max_tokens: int = 0
    max_tool_calls: int = 0


@dataclass
class IterationGate:
    """One gate evaluation rule used by phase iteration policies."""

    id: str
    type: str
    blocking: bool = False
    operator: str = ""
    value: Any = None
    tool: str = ""
    metric_path: str = ""
    target: str = ""
    pattern: str = ""
    expect_match: bool = True
    verifier_field: str = ""
    command: list[str] = field(default_factory=list)
    timeout_seconds: int = 60


@dataclass
class IterationPolicy:
    """Phase-level iteration contract."""

    enabled: bool = False
    max_attempts: int = 4
    strategy: str = "targeted_remediation"
    stop_on_no_improvement_attempts: int = 0
    max_total_runner_invocations: int = 0
    max_replans_after_exhaustion: int = 2
    replan_on_exhaustion: bool = True
    budget: IterationBudget = field(default_factory=IterationBudget)
    gates: list[IterationGate] = field(default_factory=list)


@dataclass
class OutputCoordination:
    """Process-level output ownership and publish coordination policy."""

    strategy: str = "direct"  # "direct" | "fan_in"
    intermediate_root: str = ".loom/phase-artifacts"
    enforce_single_writer: bool = True
    publish_mode: str = "transactional"  # "transactional" | "best_effort"
    conflict_policy: str = "defer_fifo"  # "defer_fifo" | "fail_fast"
    finalizer_input_policy: str = "require_all_workers"  # "require_all_workers" | "allow_partial"


@dataclass
class PhaseTemplate:
    """A blueprint phase in the process definition."""

    id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    model_tier: int = 2
    verification_tier: int = 1
    is_critical_path: bool = False
    is_synthesis: bool = False
    output_strategy: str = ""  # "" | "direct" | "fan_in"
    finalizer_input_policy: str = ""  # "" | "require_all_workers" | "allow_partial"
    acceptance_criteria: str = ""
    deliverables: list[str] = field(default_factory=list)
    validity_contract: dict[str, Any] = field(default_factory=dict)
    iteration: IterationPolicy | None = None


@dataclass
class VerificationRule:
    """A domain-specific verification check."""

    name: str
    description: str
    check: str
    severity: str = "warning"  # "warning" | "error"
    type: str = "llm"  # "llm" | "regex"
    target: str = "output"  # "output" | "deliverables"
    enforcement: str = ""  # "" | "hard" | "advisory"
    failure_class: str = ""  # "" | "hard_integrity" | "recoverable_placeholder" | "semantic"
    remediation_mode: str = ""  # Optional remediation routing hint
    scope: str = ""  # "" | "phase" | "global"
    applies_to_phases: list[str] = field(default_factory=list)  # [] | ["phase-a"] | ["*"]
    requires_exact_cardinality: bool = False
    min_count: int | None = None


@dataclass
class MemoryType:
    """A domain-specific memory extraction type."""

    type: str
    description: str


@dataclass
class PlannerExample:
    """A few-shot example for the planner."""

    goal: str
    subtasks: list[dict] = field(default_factory=list)


@dataclass
class ProcessTestAcceptance:
    """Acceptance contract for a process test case."""

    phases_must_include: list[str] = field(default_factory=list)
    deliverables_must_exist: list[str] = field(default_factory=list)
    verification_forbidden_patterns: list[str] = field(default_factory=list)


@dataclass
class ProcessTestCase:
    """Declarative test case embedded in process.yaml."""

    id: str
    mode: str = "deterministic"  # "deterministic" | "live"
    goal: str = ""
    timeout_seconds: int = 900
    requires_network: bool = False
    requires_tools: list[str] = field(default_factory=list)
    acceptance: ProcessTestAcceptance = field(
        default_factory=ProcessTestAcceptance,
    )


@dataclass
class ToolRequirements:
    """Tool availability and restriction configuration."""

    guidance: str = ""
    required: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)


@dataclass
class AuthRequirement:
    """Auth resource requirement for process preflight."""

    provider: str
    source: str = "api"  # "api" | "mcp"
    modes: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    required_env_keys: list[str] = field(default_factory=list)
    mcp_server: str = ""
    resource_ref: str = ""
    resource_id: str = ""


@dataclass
class AuthRequirements:
    """Auth requirements declared by process contract."""

    required: list[AuthRequirement] = field(default_factory=list)


@dataclass
class VerificationPolicyContract:
    """Structured verification policy from process contract v2."""

    mode: str = "llm_first"  # "llm_first" | "static_first"
    static_checks: dict[str, Any] = field(default_factory=dict)
    semantic_checks: list[dict[str, Any]] = field(default_factory=list)
    output_contract: dict[str, Any] = field(default_factory=dict)
    outcome_policy: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationRemediationContract:
    """Remediation strategy contract from process contract v2."""

    strategies: list[dict[str, Any]] = field(default_factory=list)
    default_strategy: str = ""
    critical_path_behavior: str = ""
    retry_budget: dict[str, Any] = field(default_factory=dict)
    transient_retry_policy: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceContract:
    """Evidence extraction and schema contract from process contract v2."""

    record_schema: dict[str, Any] = field(default_factory=dict)
    extraction: dict[str, Any] = field(default_factory=dict)
    summarization: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptContracts:
    """Prompt-level behavioral constraints from process contract v2."""

    executor_constraints: str = ""
    verifier_constraints: str = ""
    remediation_instructions: str | dict[str, Any] = ""
    evidence_contract: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessDefinition:
    """Complete process definition — the unit of domain specialization.

    Loaded from YAML and injected into the PromptAssembler to specialize
    the engine for a particular domain without any Python code changes.
    """

    # Metadata
    name: str
    schema_version: int = 1
    version: str = "1.0"
    description: str = ""
    author: str = ""
    tags: list[str] = field(default_factory=list)
    risk_level: str = ""

    # Behavior
    persona: str = ""
    tool_guidance: str = ""
    tools: ToolRequirements = field(default_factory=ToolRequirements)
    auth: AuthRequirements = field(default_factory=AuthRequirements)
    phase_mode: str = "guided"  # "strict" | "guided" | "suggestive"
    output_coordination: OutputCoordination = field(default_factory=OutputCoordination)
    phases: list[PhaseTemplate] = field(default_factory=list)
    verification_rules: list[VerificationRule] = field(
        default_factory=list,
    )
    memory_types: list[MemoryType] = field(default_factory=list)
    extraction_guidance: str = ""
    workspace_scan: list[str] = field(default_factory=list)
    workspace_guidance: str = ""
    planner_examples: list[PlannerExample] = field(default_factory=list)
    tests: list[ProcessTestCase] = field(default_factory=list)
    replanning_triggers: str = ""
    replanning_guidance: str = ""
    verification_policy: VerificationPolicyContract = field(
        default_factory=VerificationPolicyContract,
    )
    verification_remediation: VerificationRemediationContract = field(
        default_factory=VerificationRemediationContract,
    )
    evidence: EvidenceContract = field(default_factory=EvidenceContract)
    validity_contract: dict[str, Any] = field(default_factory=dict)
    prompt_contracts: PromptContracts = field(default_factory=PromptContracts)

    # Package metadata
    dependencies: list[str] = field(default_factory=list)

    # Resolved paths (set by loader)
    source_path: Path | None = None
    package_dir: Path | None = None
    _HIGH_RISK_INTENT_TOKENS = frozenset({
        "investment",
        "investing",
        "finance",
        "financial",
        "medical",
        "medicine",
        "healthcare",
        "health",
        "legal",
        "law",
        "compliance",
    })
    _RISK_LEVEL_ALIASES = {
        "normal": "medium",
        "med": "medium",
        "mid": "medium",
        "std": "medium",
        "standard": "medium",
    }
    _VALID_RISK_LEVELS = frozenset({"low", "medium", "high", "critical"})
    _OUTPUT_STRATEGIES = frozenset({"direct", "fan_in"})
    _OUTPUT_PUBLISH_MODES = frozenset({"transactional", "best_effort"})
    _OUTPUT_CONFLICT_POLICIES = frozenset({"defer_fifo", "fail_fast"})
    _FINALIZER_INPUT_POLICIES = frozenset({"require_all_workers", "allow_partial"})
    _VERIFIER_CAPABILITIES = frozenset({
        "artifact_static",
        "command_execution",
        "service_runtime",
        "browser_runtime",
        "report_rendering",
    })
    _PHASE_FINALIZER_SUFFIX = "__finalize_output"

    def has_phases(self) -> bool:
        return bool(self.phases)

    def has_verification_rules(self) -> bool:
        return bool(self.verification_rules)

    @classmethod
    def _normalize_output_strategy(cls, value: object, *, default: str = "direct") -> str:
        text = str(value or "").strip().lower()
        if text in cls._OUTPUT_STRATEGIES:
            return text
        return default

    def phase_output_strategy(self, phase_id: str) -> str:
        default_strategy = self._normalize_output_strategy(
            getattr(self.output_coordination, "strategy", "direct"),
            default="direct",
        )
        normalized_phase_id = str(phase_id or "").strip()
        if not normalized_phase_id:
            return default_strategy
        for phase in self.phases:
            if str(getattr(phase, "id", "")).strip() != normalized_phase_id:
                continue
            override = str(getattr(phase, "output_strategy", "") or "").strip().lower()
            if override in self._OUTPUT_STRATEGIES:
                return override
            break
        return default_strategy

    def phase_finalizer_input_policy(self, phase_id: str) -> str:
        default_policy = str(
            getattr(self.output_coordination, "finalizer_input_policy", "require_all_workers"),
        ).strip().lower() or "require_all_workers"
        if default_policy not in self._FINALIZER_INPUT_POLICIES:
            default_policy = "require_all_workers"
        normalized_phase_id = str(phase_id or "").strip()
        if not normalized_phase_id:
            return default_policy
        for phase in self.phases:
            if str(getattr(phase, "id", "")).strip() != normalized_phase_id:
                continue
            override = str(getattr(phase, "finalizer_input_policy", "") or "").strip().lower()
            if override in self._FINALIZER_INPUT_POLICIES:
                return override
            break
        return default_policy

    @classmethod
    def phase_finalizer_id(cls, phase_id: str) -> str:
        normalized_phase_id = str(phase_id or "").strip()
        if not normalized_phase_id:
            return ""
        return f"{normalized_phase_id}{cls._PHASE_FINALIZER_SUFFIX}"

    def is_contract_v2(self) -> bool:
        return int(self.schema_version or 1) >= 2

    def verifier_mode(self) -> str:
        """Return the normalized verification policy mode."""
        mode = str(self.verification_policy.mode or "llm_first").strip().lower()
        if mode not in {"llm_first", "static_first"}:
            return "llm_first"
        return mode

    def verifier_static_checks(self) -> dict[str, Any]:
        """Return normalized static verification policy settings."""
        static_checks = self.verification_policy.static_checks
        if isinstance(static_checks, dict):
            return dict(static_checks)
        return {}

    def verifier_semantic_checks(self) -> list[dict[str, Any]]:
        """Return normalized semantic verification policy settings."""
        semantic_checks = self.verification_policy.semantic_checks
        if not isinstance(semantic_checks, list):
            return []
        return [item for item in semantic_checks if isinstance(item, dict)]

    def verifier_semantic_check_contracts(self) -> list[dict[str, Any]]:
        """Return normalized semantic verifier check contracts."""
        contracts: list[dict[str, Any]] = []
        for item in self.verifier_semantic_checks():
            name = str(item.get("name", "") or "").strip()
            capability = str(item.get("capability", "") or "").strip().lower()
            target = str(item.get("target", "") or "").strip()
            helper = str(item.get("helper", "") or "").strip().lower()
            kind = str(item.get("kind", "") or "").strip().lower()
            description = str(item.get("description", "") or "").strip()
            optional = self._to_bool(item.get("optional", False), default=False)
            contract: dict[str, Any] = {}
            if name:
                contract["name"] = name
            if capability in self._VERIFIER_CAPABILITIES:
                contract["capability"] = capability
            if target:
                contract["target"] = target
            if helper:
                contract["helper"] = helper
            if kind:
                contract["kind"] = kind
            if description:
                contract["description"] = description
            if optional:
                contract["optional"] = True
            if contract:
                contracts.append(contract)
        return contracts

    def verifier_helper_contracts(self) -> list[dict[str, Any]]:
        """Return semantic verifier checks that declare registered helpers."""
        contracts: list[dict[str, Any]] = []
        for item in self.verifier_capability_contracts():
            helper = str(item.get("helper", "") or "").strip().lower()
            if helper:
                contracts.append(dict(item))
        return contracts

    def verifier_helper_specs(self) -> list[dict[str, object]]:
        """Return unique helper specs with descriptions for prompts/tooling."""
        specs: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        for item in self.verifier_helper_contracts():
            helper = str(item.get("helper", "") or "").strip().lower()
            capability = str(item.get("capability", "") or "").strip().lower()
            if not helper or (helper, capability) in seen:
                continue
            helper_spec = get_verification_helper(helper)
            description = helper_spec.description if helper_spec else ""
            specs.append({
                "helper": helper,
                "capability": capability,
                "description": str(description or "").strip(),
                "bound": bool(verification_helper_is_bound(helper)),
            })
            seen.add((helper, capability))
        return specs

    def verifier_output_contract(self) -> dict[str, Any]:
        output = self.verification_policy.output_contract
        if isinstance(output, dict):
            return dict(output)
        return {}

    def verifier_outcome_policy(self) -> dict[str, Any]:
        """Return normalized outcome verification policy settings."""
        outcome_policy = self.verification_policy.outcome_policy
        if isinstance(outcome_policy, dict):
            return dict(outcome_policy)
        return {}

    def verifier_required_response_fields(self) -> list[str]:
        if not self.is_contract_v2():
            return ["passed"]

        required = [
            "passed",
            "outcome",
            "reason_code",
            "severity_class",
            "confidence",
            "feedback",
            "issues",
            "metadata",
        ]
        output = self.verifier_output_contract()
        extra = output.get("required_fields", [])
        if isinstance(extra, list):
            for field_name in extra:
                text = str(field_name or "").strip()
                if text:
                    required.append(text)
        metadata_fields = output.get("metadata_fields", [])
        if isinstance(metadata_fields, list) and metadata_fields and "metadata" not in required:
            required.append("metadata")
        return list(dict.fromkeys(required))

    def verifier_metadata_fields(self) -> list[str]:
        output = self.verifier_output_contract()
        fields: list[str] = []
        for key in ("metadata_fields", "counters", "flags"):
            raw = output.get(key, [])
            if not isinstance(raw, list):
                continue
            for item in raw:
                text = str(item or "").strip()
                if text:
                    fields.append(text)
        return list(dict.fromkeys(fields))

    def verifier_optional_capabilities(self) -> list[str]:
        """Return normalized optional verifier capabilities from outcome policy."""
        normalized: list[str] = []
        raw = self.verifier_outcome_policy().get("optional_capabilities", [])
        if isinstance(raw, list):
            for item in raw:
                text = str(item or "").strip().lower()
                if text and text not in normalized:
                    normalized.append(text)
        for item in self.verifier_semantic_check_contracts():
            if not bool(item.get("optional", False)):
                continue
            text = str(item.get("capability", "") or "").strip().lower()
            if text and text not in normalized:
                normalized.append(text)
        return normalized

    def verifier_required_capabilities(self) -> list[str]:
        """Return normalized required verifier capabilities from semantic checks."""
        normalized: list[str] = []
        for item in self.verifier_semantic_check_contracts():
            if bool(item.get("optional", False)):
                continue
            text = str(item.get("capability", "") or "").strip().lower()
            if text and text not in normalized:
                normalized.append(text)
        return normalized

    def verifier_prefers_behavior_over_style(self) -> bool:
        """Return True when the verifier should prefer behavior checks over style."""
        return self._to_bool(
            self.verifier_outcome_policy().get("prefer_behavior_over_style", False),
            default=False,
        )

    def verifier_capability_contracts(self) -> list[dict[str, Any]]:
        """Return semantic verifier checks that declare concrete capabilities."""
        contracts: list[dict[str, Any]] = []
        for item in self.verifier_semantic_check_contracts():
            capability = str(item.get("capability", "") or "").strip().lower()
            if capability:
                contracts.append(dict(item))
        return contracts

    def verifier_treat_infra_as_warning(self) -> bool:
        """Return True when verifier infra failures can downgrade to warnings."""
        return self._to_bool(
            self.verifier_outcome_policy().get("treat_verifier_infra_as_warning", False),
            default=False,
        )

    def is_adhoc_process(self) -> bool:
        """Return True for generated ad hoc process definitions."""
        if any(str(tag or "").strip().lower() == "adhoc" for tag in self.tags):
            return True

        version = str(self.version or "").strip().lower()
        if version.startswith("adhoc-"):
            return True

        name = str(self.name or "").strip().lower()
        return name.endswith("-adhoc")

    def verifier_tool_success_policy(
        self,
        *,
        include_adhoc_fallback: bool = False,
    ) -> str:
        """Return the normalized tier-1 tool-failure policy."""
        policy = str(
            self.verifier_static_checks().get("tool_success_policy", "") or "",
        ).strip().lower()
        if policy in {
            "all_tools_hard",
            "development_balanced",
            "safety_integrity_only",
        }:
            return policy
        if include_adhoc_fallback and self.is_adhoc_process():
            return "safety_integrity_only"
        return "all_tools_hard"

    def verification_policy_payload(self) -> dict[str, Any]:
        """Serialize the structured verification policy into plain JSON-safe data."""
        return {
            "mode": self.verifier_mode(),
            "static_checks": self.verifier_static_checks(),
            "semantic_checks": self.verifier_semantic_checks(),
            "output_contract": self.verifier_output_contract(),
            "outcome_policy": self.verifier_outcome_policy(),
        }

    @staticmethod
    def _to_bool(value: object, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        lowered = str(value or "").strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
        return bool(default)

    @staticmethod
    def _to_ratio(value: object, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = float(default)
        return max(0.0, min(1.0, parsed))

    @staticmethod
    def _to_non_negative_int(value: object, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = int(default)
        return max(0, parsed)

    @classmethod
    def _normalize_risk_level(cls, value: object) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        normalized = cls._RISK_LEVEL_ALIASES.get(text, text)
        if normalized in cls._VALID_RISK_LEVELS:
            return normalized
        return ""

    @classmethod
    def _normalize_validity_contract(cls, raw: dict[str, Any] | None) -> dict[str, Any]:
        payload = dict(raw or {})
        claim_extraction_raw = payload.get("claim_extraction", {})
        if isinstance(claim_extraction_raw, bool):
            claim_extraction_raw = {"enabled": claim_extraction_raw}
        if not isinstance(claim_extraction_raw, dict):
            claim_extraction_raw = {}

        final_gate_raw = payload.get("final_gate", {})
        if not isinstance(final_gate_raw, dict):
            final_gate_raw = {}
        temporal_raw = final_gate_raw.get("temporal_consistency", {})
        if not isinstance(temporal_raw, dict):
            temporal_raw = {}

        critical_claim_types_raw = payload.get("critical_claim_types", [])
        if isinstance(critical_claim_types_raw, str):
            critical_claim_types_raw = [critical_claim_types_raw]
        if not isinstance(critical_claim_types_raw, list):
            critical_claim_types_raw = []
        critical_claim_types = list(dict.fromkeys(
            str(item or "").strip().lower()
            for item in critical_claim_types_raw
            if str(item or "").strip()
        ))

        prune_mode = str(payload.get("prune_mode", "drop") or "").strip().lower()
        if prune_mode not in {"drop", "rewrite_uncertainty"}:
            prune_mode = "drop"

        return {
            "enabled": cls._to_bool(payload.get("enabled", False), False),
            "claim_extraction": {
                "enabled": cls._to_bool(claim_extraction_raw.get("enabled", False), False),
            },
            "critical_claim_types": critical_claim_types,
            "min_supported_ratio": cls._to_ratio(payload.get("min_supported_ratio", 0.75), 0.75),
            "max_unverified_ratio": cls._to_ratio(payload.get("max_unverified_ratio", 0.25), 0.25),
            "max_contradicted_count": cls._to_non_negative_int(
                payload.get("max_contradicted_count", 0),
                0,
            ),
            "prune_mode": prune_mode,
            "require_fact_checker_for_synthesis": cls._to_bool(
                payload.get("require_fact_checker_for_synthesis", False),
                False,
            ),
            "final_gate": {
                "enforce_verified_context_only": cls._to_bool(
                    final_gate_raw.get("enforce_verified_context_only", True),
                    True,
                ),
                "synthesis_min_verification_tier": max(
                    1,
                    cls._to_non_negative_int(
                        final_gate_raw.get("synthesis_min_verification_tier", 2),
                        2,
                    ),
                ),
                "critical_claim_support_ratio": cls._to_ratio(
                    final_gate_raw.get("critical_claim_support_ratio", 1.0),
                    1.0,
                ),
                "temporal_consistency": {
                    "enabled": cls._to_bool(temporal_raw.get("enabled", False), False),
                    "require_as_of_alignment": cls._to_bool(
                        temporal_raw.get("require_as_of_alignment", False),
                        False,
                    ),
                    "enforce_cross_claim_date_conflict_check": cls._to_bool(
                        temporal_raw.get("enforce_cross_claim_date_conflict_check", False),
                        False,
                    ),
                    "max_source_age_days": cls._to_non_negative_int(
                        temporal_raw.get("max_source_age_days", 0),
                        0,
                    ),
                    "as_of": str(temporal_raw.get("as_of", "") or "").strip(),
                },
            },
        }

    @classmethod
    def _merge_validity_contract(
        cls,
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> dict[str, Any]:
        merged = cls._normalize_validity_contract(base)
        incoming = cls._normalize_validity_contract(override)

        merged["enabled"] = bool(merged.get("enabled", False) or incoming.get("enabled", False))
        merged["claim_extraction"] = {
            "enabled": bool(
                merged.get("claim_extraction", {}).get("enabled", False)
                or incoming.get("claim_extraction", {}).get("enabled", False)
            ),
        }
        merged["critical_claim_types"] = list(dict.fromkeys([
            *list(merged.get("critical_claim_types", [])),
            *list(incoming.get("critical_claim_types", [])),
        ]))
        merged["min_supported_ratio"] = max(
            cls._to_ratio(merged.get("min_supported_ratio", 0.75), 0.75),
            cls._to_ratio(incoming.get("min_supported_ratio", 0.75), 0.75),
        )
        merged["max_unverified_ratio"] = min(
            cls._to_ratio(merged.get("max_unverified_ratio", 0.25), 0.25),
            cls._to_ratio(incoming.get("max_unverified_ratio", 0.25), 0.25),
        )
        merged["max_contradicted_count"] = min(
            cls._to_non_negative_int(merged.get("max_contradicted_count", 0), 0),
            cls._to_non_negative_int(incoming.get("max_contradicted_count", 0), 0),
        )
        incoming_prune_mode = str(incoming.get("prune_mode", "") or "").strip().lower()
        if incoming_prune_mode in {"drop", "rewrite_uncertainty"}:
            merged["prune_mode"] = incoming_prune_mode
        merged["require_fact_checker_for_synthesis"] = bool(
            merged.get("require_fact_checker_for_synthesis", False)
            or incoming.get("require_fact_checker_for_synthesis", False)
        )

        final_gate = dict(merged.get("final_gate", {}))
        incoming_final_gate = dict(incoming.get("final_gate", {}))
        final_gate["enforce_verified_context_only"] = bool(
            final_gate.get("enforce_verified_context_only", True)
            or incoming_final_gate.get("enforce_verified_context_only", True)
        )
        final_gate["synthesis_min_verification_tier"] = max(
            1,
            max(
                int(final_gate.get("synthesis_min_verification_tier", 2) or 2),
                int(incoming_final_gate.get("synthesis_min_verification_tier", 2) or 2),
            ),
        )
        final_gate["critical_claim_support_ratio"] = max(
            cls._to_ratio(final_gate.get("critical_claim_support_ratio", 1.0), 1.0),
            cls._to_ratio(incoming_final_gate.get("critical_claim_support_ratio", 1.0), 1.0),
        )
        temporal = dict(final_gate.get("temporal_consistency", {}))
        incoming_temporal = dict(incoming_final_gate.get("temporal_consistency", {}))
        temporal["enabled"] = bool(
            temporal.get("enabled", False)
            or incoming_temporal.get("enabled", False)
        )
        temporal["require_as_of_alignment"] = bool(
            temporal.get("require_as_of_alignment", False)
            or incoming_temporal.get("require_as_of_alignment", False)
        )
        temporal["enforce_cross_claim_date_conflict_check"] = bool(
            temporal.get("enforce_cross_claim_date_conflict_check", False)
            or incoming_temporal.get("enforce_cross_claim_date_conflict_check", False)
        )
        base_age = cls._to_non_negative_int(temporal.get("max_source_age_days", 0), 0)
        incoming_age = cls._to_non_negative_int(incoming_temporal.get("max_source_age_days", 0), 0)
        if base_age > 0 and incoming_age > 0:
            temporal["max_source_age_days"] = min(base_age, incoming_age)
        else:
            temporal["max_source_age_days"] = max(base_age, incoming_age)
        incoming_as_of = str(incoming_temporal.get("as_of", "") or "").strip()
        base_as_of = str(temporal.get("as_of", "") or "").strip()
        temporal["as_of"] = incoming_as_of or base_as_of
        final_gate["temporal_consistency"] = temporal
        merged["final_gate"] = final_gate
        return merged

    def _domain_tokens(self) -> set[str]:
        tokens: set[str] = set()
        for raw in [self.name, *list(self.tags or [])]:
            text = str(raw or "").strip().lower()
            if not text:
                continue
            for token in re.split(r"[^a-z0-9]+", text):
                if token:
                    tokens.add(token)
        return tokens

    def _is_high_risk_intent(self) -> bool:
        explicit_risk = self._normalize_risk_level(self.risk_level)
        if explicit_risk in {"high", "critical"}:
            return True
        if explicit_risk in {"low", "medium"}:
            return False
        return bool(self._domain_tokens() & self._HIGH_RISK_INTENT_TOKENS)

    def _default_validity_contract_floor(self, *, is_synthesis: bool) -> dict[str, Any]:
        high_risk = self._is_high_risk_intent()
        if high_risk:
            min_supported_ratio = 0.8
            max_unverified_ratio = 0.2
            require_fact_checker = True
            critical_support_ratio = 1.0
            temporal_consistency = {
                "enabled": True,
                "require_as_of_alignment": True,
                "enforce_cross_claim_date_conflict_check": True,
                "max_source_age_days": 365,
            }
        else:
            min_supported_ratio = 0.65
            max_unverified_ratio = 0.35
            require_fact_checker = False
            critical_support_ratio = 0.9
            temporal_consistency = {
                "enabled": False,
                "require_as_of_alignment": False,
                "enforce_cross_claim_date_conflict_check": False,
                "max_source_age_days": 0,
            }
        return self._normalize_validity_contract({
            "enabled": True,
            "claim_extraction": {"enabled": True},
            "critical_claim_types": ["numeric", "date", "entity_fact"],
            "min_supported_ratio": min_supported_ratio,
            "max_unverified_ratio": max_unverified_ratio,
            "max_contradicted_count": 0,
            "prune_mode": "rewrite_uncertainty",
            "require_fact_checker_for_synthesis": require_fact_checker,
            "final_gate": {
                "enforce_verified_context_only": bool(is_synthesis),
                "synthesis_min_verification_tier": 2,
                "critical_claim_support_ratio": critical_support_ratio,
                "temporal_consistency": temporal_consistency,
            },
        })

    def resolve_validity_contract_for_phase(
        self,
        phase_id: str,
        *,
        is_synthesis: bool = False,
    ) -> dict[str, Any]:
        contract = self._default_validity_contract_floor(
            is_synthesis=bool(is_synthesis),
        )
        if isinstance(self.validity_contract, dict) and self.validity_contract:
            contract = self._merge_validity_contract(contract, self.validity_contract)
        normalized_phase_id = str(phase_id or "").strip()
        if normalized_phase_id:
            for phase in self.phases:
                if str(getattr(phase, "id", "")).strip() != normalized_phase_id:
                    continue
                phase_contract = getattr(phase, "validity_contract", {})
                if isinstance(phase_contract, dict) and phase_contract:
                    contract = self._merge_validity_contract(contract, phase_contract)
                break
        if is_synthesis:
            final_gate = dict(contract.get("final_gate", {}))
            final_gate["enforce_verified_context_only"] = True
            final_gate["synthesis_min_verification_tier"] = max(
                1,
                int(final_gate.get("synthesis_min_verification_tier", 2) or 2),
            )
            contract["final_gate"] = final_gate
        return contract

    def evidence_facets(self) -> list[str]:
        record_schema = self.evidence.record_schema
        if not isinstance(record_schema, dict):
            return []
        raw = record_schema.get("facets", [])
        if not isinstance(raw, list):
            return []
        return [
            str(item).strip()
            for item in raw
            if isinstance(item, str) and str(item).strip()
        ]

    def requires_evidence_contract(self, subtask_id: str) -> bool:
        evidence_contract = self.prompt_contracts.evidence_contract
        if isinstance(evidence_contract, dict) and evidence_contract:
            enabled = bool(evidence_contract.get("enabled", True))
            if not enabled:
                return False
            applies = evidence_contract.get("applies_to_phases", [])
            if isinstance(applies, str):
                applies = [applies]
            if isinstance(applies, list) and applies:
                normalized = [str(item or "").strip() for item in applies]
                normalized = [item for item in normalized if item]
                if "*" in normalized:
                    return True
                return subtask_id in normalized
            return True
        if self.is_contract_v2():
            return bool(self.evidence.record_schema or self.evidence.extraction)
        return False

    def prompt_executor_constraints(self) -> str:
        return str(self.prompt_contracts.executor_constraints or "").strip()

    def prompt_verifier_constraints(self) -> str:
        return str(self.prompt_contracts.verifier_constraints or "").strip()

    def prompt_remediation_instructions(self, strategy: str = "") -> str:
        raw = self.prompt_contracts.remediation_instructions
        if isinstance(raw, str):
            return raw.strip()
        if isinstance(raw, dict):
            preferred = str(strategy or "").strip().lower()
            if preferred:
                candidate = raw.get(preferred)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            default = raw.get("default")
            if isinstance(default, str):
                return default.strip()
        return ""

    def remediation_critical_path_behavior(self) -> str:
        raw = str(
            self.verification_remediation.critical_path_behavior or "",
        ).strip().lower()
        if raw in {"block", "confirm_or_prune_then_queue", "queue_follow_up"}:
            return raw
        return "block"

    def regex_rules(self) -> list[VerificationRule]:
        """Return only regex-type verification rules."""
        return [r for r in self.verification_rules if r.type == "regex"]

    def llm_rules(self) -> list[VerificationRule]:
        """Return only LLM-type verification rules."""
        return [r for r in self.verification_rules if r.type == "llm"]

    @staticmethod
    def _rule_applies_to_subtask(
        rule: VerificationRule,
        subtask_id: str,
        *,
        phase_scope_default: str = "global",
    ) -> bool:
        """Return True when rule should apply to current subtask."""
        normalized_scope_default = str(phase_scope_default or "global").strip().lower()
        if normalized_scope_default not in {"global", "current_phase"}:
            normalized_scope_default = "global"

        normalized_subtask = str(subtask_id or "").strip()
        if not normalized_subtask:
            return False

        applies = [str(p or "").strip() for p in (rule.applies_to_phases or [])]
        applies = [p for p in applies if p]
        if applies:
            if "*" in applies:
                return True
            return normalized_subtask in applies

        scope = str(rule.scope or "").strip().lower()
        if scope == "global":
            return True
        if scope == "phase":
            # Rule is explicitly phase-scoped but no explicit phase list exists.
            # Interpret as "apply to the currently executing phase/subtask".
            return bool(normalized_subtask)

        # Legacy rule (no scope metadata)
        if normalized_scope_default == "global":
            return True
        # Migration mode: treat legacy rules as current-phase checks.
        return bool(normalized_subtask)

    def regex_rules_for_subtask(
        self,
        subtask_id: str,
        *,
        phase_scope_default: str = "global",
    ) -> list[VerificationRule]:
        """Return regex rules applicable to the current subtask."""
        return [
            r for r in self.regex_rules()
            if self._rule_applies_to_subtask(
                r,
                subtask_id,
                phase_scope_default=phase_scope_default,
            )
        ]

    def llm_rules_for_subtask(
        self,
        subtask_id: str,
        *,
        phase_scope_default: str = "global",
    ) -> list[VerificationRule]:
        """Return llm rules applicable to the current subtask."""
        return [
            r for r in self.llm_rules()
            if self._rule_applies_to_subtask(
                r,
                subtask_id,
                phase_scope_default=phase_scope_default,
            )
        ]

    def get_deliverables(self) -> dict[str, list[str]]:
        """Return {phase_id: [deliverable_filenames]} for all phases."""
        result: dict[str, list[str]] = {}
        for phase in self.phases:
            if phase.deliverables:
                # Extract just the filename from "filename — description"
                filenames = []
                for d in phase.deliverables:
                    fname = d.split("—")[0].split(" — ")[0].strip()
                    filenames.append(fname)
                result[phase.id] = filenames
        return result

    def auth_required_resources(self) -> list[dict[str, Any]]:
        """Return normalized auth.required records as plain dicts."""
        items: list[dict[str, Any]] = []
        for requirement in self.auth.required:
            payload: dict[str, Any] = {
                "provider": str(requirement.provider or "").strip(),
                "source": str(requirement.source or "api").strip().lower() or "api",
            }
            if requirement.modes:
                payload["modes"] = [
                    str(item).strip()
                    for item in requirement.modes
                    if str(item).strip()
                ]
            if requirement.scopes:
                payload["scopes"] = [
                    str(item).strip()
                    for item in requirement.scopes
                    if str(item).strip()
                ]
            if requirement.required_env_keys:
                payload["required_env_keys"] = [
                    str(item).strip()
                    for item in requirement.required_env_keys
                    if str(item).strip()
                ]
            if str(requirement.mcp_server or "").strip():
                payload["mcp_server"] = str(requirement.mcp_server).strip()
            if str(requirement.resource_ref or "").strip():
                payload["resource_ref"] = str(requirement.resource_ref).strip()
            if str(requirement.resource_id or "").strip():
                payload["resource_id"] = str(requirement.resource_id).strip()
            if (
                payload.get("provider")
                or payload.get("resource_ref")
                or payload.get("resource_id")
            ):
                items.append(payload)
        return items


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class ProcessLoader:
    """Discovers and loads process definition files.

    Search order (later entries take precedence):
    1. Built-in processes shipped with Loom
    2. User-global (~/.loom/processes/)
    3. Workspace-local (./loom-processes/ or ./.loom/processes/)

    Supports both single YAML files and directory-based packages.
    """

    BUILTIN_DIR = Path(__file__).parent / "builtin"
    _ITERATION_STRATEGIES = frozenset({"targeted_remediation", "full_rerun"})
    _ARTIFACT_REGEX_TARGETS = frozenset({
        "",
        "auto",
        "deliverables",
        "changed_files",
        "summary",
        "output",
    })
    _COMMAND_EXIT_ALLOWLIST_PREFIXES = (
        ("pytest",),
        ("uv", "run", "pytest"),
        ("python", "-m", "pytest"),
        ("python3", "-m", "pytest"),
        ("ruff", "check"),
        ("npm", "test"),
        ("pnpm", "test"),
        ("bun", "test"),
        ("go", "test"),
        ("cargo", "test"),
        ("make", "test"),
    )

    def __init__(
        self,
        workspace: Path | None = None,
        extra_search_paths: list[Path] | None = None,
        *,
        require_rule_scope_metadata: bool = False,
        require_v2_contract: bool = False,
    ):
        self._workspace = workspace
        self._extra = extra_search_paths or []
        self._require_rule_scope_metadata = bool(require_rule_scope_metadata)
        self._require_v2_contract = bool(require_v2_contract)

    @property
    def search_paths(self) -> list[Path]:
        """Return search paths in precedence order (last wins)."""
        paths = [self.BUILTIN_DIR]
        paths.append(Path.home() / ".loom" / "processes")
        for p in self._extra:
            paths.append(Path(p).expanduser())
        if self._workspace:
            paths.append(self._workspace / ".loom" / "processes")
            paths.append(self._workspace / "loom-processes")
        return paths

    def discover(self) -> dict[str, Path]:
        """Find all available process definitions.

        Returns ``{name: path}`` where path is the YAML file or the
        package directory.  Later search paths override earlier ones.
        """
        processes: dict[str, Path] = {}
        for search_dir in self.search_paths:
            if not search_dir.exists():
                continue
            # Single YAML files
            for yaml_file in search_dir.glob("*.yaml"):
                name = self._quick_parse_name(yaml_file)
                if name:
                    processes[name] = yaml_file
            # Directory-based packages (contain process.yaml)
            for subdir in search_dir.iterdir():
                if subdir.is_dir():
                    manifest = subdir / "process.yaml"
                    if manifest.exists():
                        name = self._quick_parse_name(manifest)
                        if name:
                            processes[name] = subdir
        return processes

    def load(self, name_or_path: str) -> ProcessDefinition:
        """Load a process definition by name or file/directory path."""
        path = Path(name_or_path)
        if path.exists():
            return self._load_from_path(path)

        # Search by name
        available = self.discover()
        if name_or_path not in available:
            raise ProcessNotFoundError(name_or_path, list(available.keys()))
        return self._load_from_path(available[name_or_path])

    def list_available(self) -> list[dict[str, str]]:
        """List all available processes with metadata.

        Only parses YAML for metadata — does NOT import bundled tools.
        """
        result = []
        for name, path in sorted(self.discover().items()):
            try:
                defn = self._load_metadata_only(path)
                result.append({
                    "name": defn.name,
                    "version": defn.version,
                    "description": defn.description,
                    "author": defn.author,
                    "path": str(path),
                })
            except Exception as e:
                logger.warning("Failed to load process %s: %s", name, e)
                result.append({
                    "name": name,
                    "version": "?",
                    "description": "(failed to load)",
                    "author": "",
                    "path": str(path),
                })
        return result

    # --- Internal ---

    def _load_metadata_only(self, path: Path) -> ProcessDefinition:
        """Parse a process definition for metadata only (no bundled tool imports)."""
        if path.is_dir():
            yaml_path = path / "process.yaml"
            if not yaml_path.exists():
                raise ProcessValidationError(
                    ["Package directory missing process.yaml"],
                    path,
                )
            defn = self._parse(yaml_path)
            defn.package_dir = path
            # Deliberately skip _register_bundled_tools
            return defn
        else:
            return self._parse(path)

    def _load_from_path(self, path: Path) -> ProcessDefinition:
        """Load from a YAML file or package directory."""
        if path.is_dir():
            yaml_path = path / "process.yaml"
            if not yaml_path.exists():
                raise ProcessValidationError(
                    ["Package directory missing process.yaml"],
                    path,
                )
            defn = self._parse(yaml_path)
            defn.package_dir = path
            # Activate per-package isolated dependencies if present.
            self._activate_isolated_dependencies(path)
            # Register bundled tools if present
            self._register_bundled_tools(path)
            return defn
        else:
            return self._parse(path)

    def _parse(self, path: Path) -> ProcessDefinition:
        """Parse a YAML file into a ProcessDefinition."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        if not raw or not isinstance(raw, dict):
            raise ProcessValidationError(
                ["Empty or invalid YAML file"], path,
            )

        defn = self._build_definition(raw)
        defn.source_path = path

        errors = self._validate(defn)
        if errors:
            raise ProcessValidationError(errors, path)
        return defn

    def _build_definition(self, raw: dict[str, Any]) -> ProcessDefinition:
        """Construct a ProcessDefinition from parsed YAML."""
        schema_version_raw = raw.get("schema_version", 1)
        try:
            schema_version = int(schema_version_raw)
        except (TypeError, ValueError):
            schema_version = 1
        if schema_version < 1:
            schema_version = 1
        risk_level_raw = str(raw.get("risk_level", "") or "").strip().lower()

        # Phases
        phases = []
        for p in raw.get("phases", []):
            phase_validity_contract = p.get("validity_contract", {})
            if isinstance(phase_validity_contract, bool):
                phase_validity_contract = {"enabled": phase_validity_contract}
            if not isinstance(phase_validity_contract, dict):
                phase_validity_contract = {}
            phases.append(PhaseTemplate(
                id=p.get("id", ""),
                description=p.get("description", ""),
                depends_on=p.get("depends_on", []),
                model_tier=p.get("model_tier", 2),
                verification_tier=p.get("verification_tier", 1),
                is_critical_path=p.get("is_critical_path", False),
                is_synthesis=p.get("is_synthesis", False),
                output_strategy=str(p.get("output_strategy", "") or "").strip().lower(),
                finalizer_input_policy=str(
                    p.get("finalizer_input_policy", "") or "",
                ).strip().lower(),
                acceptance_criteria=p.get("acceptance_criteria", ""),
                deliverables=p.get("deliverables", []),
                validity_contract=phase_validity_contract,
                iteration=self._parse_iteration_policy(p.get("iteration")),
            ))

        # Output coordination policy
        output_coordination_raw = raw.get("output_coordination", {})
        if not isinstance(output_coordination_raw, dict):
            output_coordination_raw = {}
        output_coordination = OutputCoordination(
            strategy=str(output_coordination_raw.get("strategy", "direct") or "")
            .strip()
            .lower(),
            intermediate_root=str(
                output_coordination_raw.get("intermediate_root", ".loom/phase-artifacts")
                or "",
            ).strip(),
            enforce_single_writer=self._bool_or_default(
                output_coordination_raw.get("enforce_single_writer", True),
                True,
            ),
            publish_mode=str(output_coordination_raw.get("publish_mode", "transactional") or "")
            .strip()
            .lower(),
            conflict_policy=str(output_coordination_raw.get("conflict_policy", "defer_fifo") or "")
            .strip()
            .lower(),
            finalizer_input_policy=str(
                output_coordination_raw.get("finalizer_input_policy", "require_all_workers")
                or "",
            ).strip().lower(),
        )

        # Verification rules + contract v2 verification blocks
        rules = []
        verif = raw.get("verification", {})
        if not isinstance(verif, dict):
            verif = {}
        for r in verif.get("rules", []):
            applies_to = r.get("applies_to_phases", [])
            if isinstance(applies_to, str):
                applies_to = [applies_to]
            if not isinstance(applies_to, list):
                applies_to = []
            rules.append(VerificationRule(
                name=r.get("name", ""),
                description=r.get("description", ""),
                check=r.get("check", ""),
                severity=r.get("severity", "warning"),
                type=r.get("type", "llm"),
                target=r.get("target", "output"),
                enforcement=r.get("enforcement", ""),
                failure_class=str(r.get("failure_class", "") or "").strip().lower(),
                remediation_mode=str(r.get("remediation_mode", "") or "").strip().lower(),
                scope=r.get("scope", ""),
                applies_to_phases=[str(p).strip() for p in applies_to if str(p).strip()],
                requires_exact_cardinality=bool(
                    r.get("requires_exact_cardinality", False)
                ),
                min_count=r.get("min_count"),
            ))

        policy_raw = verif.get("policy", {})
        if not isinstance(policy_raw, dict):
            policy_raw = {}
        static_checks = policy_raw.get("static_checks", {})
        if not isinstance(static_checks, dict):
            static_checks = {}
        semantic_checks = policy_raw.get("semantic_checks", [])
        if not isinstance(semantic_checks, list):
            semantic_checks = []
        semantic_checks = [
            item for item in semantic_checks
            if isinstance(item, dict)
        ]
        output_contract = policy_raw.get("output_contract", {})
        if not isinstance(output_contract, dict):
            output_contract = {}
        outcome_policy = policy_raw.get("outcome_policy", {})
        if not isinstance(outcome_policy, dict):
            outcome_policy = {}
        verification_policy = VerificationPolicyContract(
            mode=str(policy_raw.get("mode", "llm_first") or "llm_first")
            .strip()
            .lower(),
            static_checks=static_checks,
            semantic_checks=semantic_checks,
            output_contract=output_contract,
            outcome_policy=outcome_policy,
        )

        remediation_raw = verif.get("remediation", {})
        if not isinstance(remediation_raw, dict):
            remediation_raw = {}
        strategies = remediation_raw.get("strategies", [])
        if not isinstance(strategies, list):
            strategies = []
        strategies = [
            item for item in strategies
            if isinstance(item, dict)
        ]
        retry_budget = remediation_raw.get("retry_budget", {})
        if not isinstance(retry_budget, dict):
            retry_budget = {}
        transient_retry_policy = remediation_raw.get("transient_retry_policy", {})
        if not isinstance(transient_retry_policy, dict):
            transient_retry_policy = {}
        verification_remediation = VerificationRemediationContract(
            strategies=strategies,
            default_strategy=str(remediation_raw.get("default_strategy", "") or "").strip(),
            critical_path_behavior=str(
                remediation_raw.get("critical_path_behavior", "") or "",
            ).strip().lower(),
            retry_budget=retry_budget,
            transient_retry_policy=transient_retry_policy,
        )

        # Memory types
        memory_raw = raw.get("memory", {})
        memory_types = []
        for m in memory_raw.get("extract_types", []) if isinstance(
            memory_raw, dict,
        ) else []:
            memory_types.append(MemoryType(
                type=m.get("type", ""),
                description=m.get("description", ""),
            ))

        # Planner examples
        examples = []
        for ex in raw.get("planner_examples", []):
            examples.append(PlannerExample(
                goal=ex.get("goal", ""),
                subtasks=ex.get("subtasks", []),
            ))

        # Process tests
        tests: list[ProcessTestCase] = []
        tests_raw = raw.get("tests", [])
        if isinstance(tests_raw, list):
            for t in tests_raw:
                if not isinstance(t, dict):
                    continue
                timeout_raw = t.get("timeout_seconds", 900)
                try:
                    timeout_seconds = int(timeout_raw)
                except (TypeError, ValueError):
                    timeout_seconds = -1
                acceptance_raw = t.get("acceptance", {})
                if not isinstance(acceptance_raw, dict):
                    acceptance_raw = {}

                phases_raw = acceptance_raw.get("phases", {})
                if not isinstance(phases_raw, dict):
                    phases_raw = {}
                deliverables_raw = acceptance_raw.get("deliverables", {})
                if not isinstance(deliverables_raw, dict):
                    deliverables_raw = {}
                verification_raw = acceptance_raw.get("verification", {})
                if not isinstance(verification_raw, dict):
                    verification_raw = {}

                tests.append(ProcessTestCase(
                    id=str(t.get("id", "")).strip(),
                    mode=str(t.get("mode", "deterministic")).strip(),
                    goal=str(t.get("goal", "")).strip(),
                    timeout_seconds=timeout_seconds,
                    requires_network=bool(t.get("requires_network", False)),
                    requires_tools=[
                        item
                        for item in t.get("requires_tools", [])
                        if isinstance(item, str) and item.strip()
                    ] if isinstance(t.get("requires_tools", []), list) else [],
                    acceptance=ProcessTestAcceptance(
                        phases_must_include=[
                            item
                            for item in phases_raw.get("must_include", [])
                            if isinstance(item, str) and item.strip()
                        ] if isinstance(phases_raw.get("must_include", []), list) else [],
                        deliverables_must_exist=[
                            item
                            for item in deliverables_raw.get("must_exist", [])
                            if isinstance(item, str) and item.strip()
                        ] if isinstance(deliverables_raw.get("must_exist", []), list) else [],
                        verification_forbidden_patterns=[
                            item
                            for item in verification_raw.get("forbidden_patterns", [])
                            if isinstance(item, str) and item.strip()
                        ] if isinstance(
                            verification_raw.get("forbidden_patterns", []), list
                        ) else [],
                    ),
                ))

        # Tool requirements
        tool_raw = raw.get("tools", {})
        if isinstance(tool_raw, dict):
            tools = ToolRequirements(
                guidance=tool_raw.get("guidance", ""),
                required=tool_raw.get("required", []),
                excluded=tool_raw.get("excluded", []),
            )
        else:
            tools = ToolRequirements()

        # Auth requirements
        auth_raw = raw.get("auth", {})
        auth_required: list[AuthRequirement] = []
        if isinstance(auth_raw, dict):
            required_raw = auth_raw.get("required", [])
            if not isinstance(required_raw, list):
                required_raw = []
            for entry in required_raw:
                if not isinstance(entry, dict):
                    continue
                modes_raw = entry.get("modes", [])
                if isinstance(modes_raw, str):
                    modes_raw = [modes_raw]
                if not isinstance(modes_raw, list):
                    modes_raw = []
                scopes_raw = entry.get("scopes", [])
                if isinstance(scopes_raw, str):
                    scopes_raw = [scopes_raw]
                if not isinstance(scopes_raw, list):
                    scopes_raw = []
                env_keys_raw = entry.get("required_env_keys", [])
                if isinstance(env_keys_raw, str):
                    env_keys_raw = [env_keys_raw]
                if not isinstance(env_keys_raw, list):
                    env_keys_raw = []
                auth_required.append(AuthRequirement(
                    provider=str(entry.get("provider", "")).strip(),
                    source=str(entry.get("source", "api") or "api").strip().lower(),
                    modes=[
                        str(item).strip()
                        for item in modes_raw
                        if str(item).strip()
                    ],
                    scopes=[
                        str(item).strip()
                        for item in scopes_raw
                        if str(item).strip()
                    ],
                    required_env_keys=[
                        str(item).strip()
                        for item in env_keys_raw
                        if str(item).strip()
                    ],
                    mcp_server=str(entry.get("mcp_server", "")).strip(),
                    resource_ref=str(entry.get("resource_ref", "")).strip(),
                    resource_id=str(entry.get("resource_id", "")).strip(),
                ))
        auth = AuthRequirements(required=auth_required)

        # Workspace analysis
        ws_raw = raw.get("workspace_analysis", {})
        ws_scan = ws_raw.get("scan_for", []) if isinstance(
            ws_raw, dict,
        ) else []
        ws_guidance = ws_raw.get("guidance", "") if isinstance(
            ws_raw, dict,
        ) else ""

        # Replanning
        replan_raw = raw.get("replanning", {})
        replan_triggers = replan_raw.get("triggers", "") if isinstance(
            replan_raw, dict,
        ) else ""
        replan_guidance = replan_raw.get("guidance", "") if isinstance(
            replan_raw, dict,
        ) else ""

        # Evidence contract v2
        evidence_raw = raw.get("evidence", {})
        if not isinstance(evidence_raw, dict):
            evidence_raw = {}
        evidence_record_schema = evidence_raw.get("record_schema", {})
        if not isinstance(evidence_record_schema, dict):
            evidence_record_schema = {}
        evidence_extraction = evidence_raw.get("extraction", {})
        if not isinstance(evidence_extraction, dict):
            evidence_extraction = {}
        evidence_summarization = evidence_raw.get("summarization", {})
        if not isinstance(evidence_summarization, dict):
            evidence_summarization = {}
        evidence_contract = EvidenceContract(
            record_schema=evidence_record_schema,
            extraction=evidence_extraction,
            summarization=evidence_summarization,
        )

        # Validity contract (claim-level verification / pruning / synthesis gates)
        validity_contract_raw = raw.get("validity_contract", {})
        if isinstance(validity_contract_raw, bool):
            validity_contract_raw = {"enabled": validity_contract_raw}
        if not isinstance(validity_contract_raw, dict):
            validity_contract_raw = {}

        # Prompt contracts v2
        prompt_contracts_raw = raw.get("prompt_contracts", {})
        if not isinstance(prompt_contracts_raw, dict):
            prompt_contracts_raw = {}
        remediation_instructions_raw = prompt_contracts_raw.get(
            "remediation_instructions",
            "",
        )
        if not isinstance(remediation_instructions_raw, (str, dict)):
            remediation_instructions_raw = str(remediation_instructions_raw)
        evidence_contract_raw = prompt_contracts_raw.get("evidence_contract", {})
        if isinstance(evidence_contract_raw, bool):
            evidence_contract_raw = {"enabled": evidence_contract_raw}
        if not isinstance(evidence_contract_raw, dict):
            evidence_contract_raw = {}
        prompt_contracts = PromptContracts(
            executor_constraints=str(
                prompt_contracts_raw.get("executor_constraints", "") or "",
            ).strip(),
            verifier_constraints=str(
                prompt_contracts_raw.get("verifier_constraints", "") or "",
            ).strip(),
            remediation_instructions=remediation_instructions_raw,
            evidence_contract=evidence_contract_raw,
        )

        # Top-level tool_guidance (legacy compat — also check tools.guidance)
        tool_guidance = raw.get("tool_guidance", "") or tools.guidance

        # Dependencies
        deps_raw = raw.get("dependencies", [])
        if not isinstance(deps_raw, list):
            deps_raw = []

        if not verification_policy.output_contract:
            verification_policy.output_contract = {
                "required_fields": [
                    "passed",
                    "outcome",
                    "reason_code",
                    "severity_class",
                    "confidence",
                    "feedback",
                    "issues",
                    "metadata",
                ],
                "metadata_fields": [],
            }

        return ProcessDefinition(
            name=raw.get("name", ""),
            schema_version=schema_version,
            version=str(raw.get("version", "1.0")),
            description=raw.get("description", ""),
            author=raw.get("author", ""),
            tags=raw.get("tags", []),
            risk_level=risk_level_raw,
            persona=raw.get("persona", ""),
            tool_guidance=tool_guidance,
            tools=tools,
            auth=auth,
            phase_mode=raw.get("phase_mode", "guided"),
            output_coordination=output_coordination,
            phases=phases,
            verification_rules=rules,
            memory_types=memory_types,
            extraction_guidance=memory_raw.get(
                "extraction_guidance", "",
            ) if isinstance(memory_raw, dict) else "",
            workspace_scan=ws_scan,
            workspace_guidance=ws_guidance,
            planner_examples=examples,
            tests=tests,
            replanning_triggers=replan_triggers,
            replanning_guidance=replan_guidance,
            verification_policy=verification_policy,
            verification_remediation=verification_remediation,
            evidence=evidence_contract,
            validity_contract=validity_contract_raw,
            prompt_contracts=prompt_contracts,
            dependencies=deps_raw,
        )

    @staticmethod
    def _int_or_default(value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _bool_or_default(value: object, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1", "on"}:
                return True
            if lowered in {"false", "no", "0", "off"}:
                return False
        return default

    @staticmethod
    def _command_tokens(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                return [token for token in shlex.split(text) if token]
            except ValueError:
                return []
        return []

    @classmethod
    def _command_exit_prefix_allowlisted(cls, command: list[str]) -> bool:
        tokens = [str(token).strip() for token in command if str(token).strip()]
        if not tokens:
            return False
        first = Path(tokens[0]).name.lower()
        normalized = [first] + [str(token).strip() for token in tokens[1:]]
        command_tuple = tuple(normalized)
        for prefix in cls._COMMAND_EXIT_ALLOWLIST_PREFIXES:
            if len(command_tuple) < len(prefix):
                continue
            if command_tuple[: len(prefix)] == prefix:
                return True
        return False

    @staticmethod
    def _normalize_workspace_relative_path(value: object) -> tuple[str, str]:
        text = str(value or "").strip()
        if not text:
            return "", "must be a non-empty workspace-relative path"
        if text.startswith("~"):
            return "", "must be workspace-relative, not home-relative"
        if re.match(r"^[A-Za-z]:[/\\\\]", text):
            return "", "must be workspace-relative, not an absolute drive path"

        normalized = text.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute():
            return "", "must be workspace-relative, not absolute"

        normalized_parts: list[str] = []
        for part in path.parts:
            if part in {"", "."}:
                continue
            if part == "..":
                return "", "must not contain parent path segments ('..')"
            normalized_parts.append(part)

        if not normalized_parts:
            return "", "must resolve to a non-empty workspace-relative path"
        return str(PurePosixPath(*normalized_parts)), ""

    def _parse_iteration_policy(self, raw: object) -> IterationPolicy | None:
        if not isinstance(raw, dict):
            return None

        budget_raw = raw.get("iteration_budget", {})
        if not isinstance(budget_raw, dict):
            budget_raw = {}
        budget = IterationBudget(
            max_wall_clock_seconds=self._int_or_default(
                budget_raw.get("max_wall_clock_seconds", 0),
                0,
            ),
            max_tokens=self._int_or_default(
                budget_raw.get("max_tokens", 0),
                0,
            ),
            max_tool_calls=self._int_or_default(
                budget_raw.get("max_tool_calls", 0),
                0,
            ),
        )

        gates: list[IterationGate] = []
        gates_raw = raw.get("gates", [])
        if isinstance(gates_raw, list):
            for index, gate_raw in enumerate(gates_raw):
                if not isinstance(gate_raw, dict):
                    continue
                gate_id = str(gate_raw.get("id", "")).strip() or f"gate-{index + 1}"
                gates.append(IterationGate(
                    id=gate_id,
                    type=str(gate_raw.get("type", "")).strip().lower(),
                    blocking=bool(gate_raw.get("blocking", False)),
                    operator=str(gate_raw.get("operator", "")).strip().lower(),
                    value=gate_raw.get("value"),
                    tool=str(gate_raw.get("tool", "")).strip(),
                    metric_path=str(gate_raw.get("metric_path", "")).strip(),
                    target=str(gate_raw.get("target", "")).strip().lower(),
                    pattern=str(gate_raw.get("pattern", "")),
                    expect_match=self._bool_or_default(
                        gate_raw.get("expect_match", True),
                        True,
                    ),
                    verifier_field=str(gate_raw.get("field", "")).strip(),
                    command=self._command_tokens(gate_raw.get("command")),
                    timeout_seconds=self._int_or_default(
                        gate_raw.get("timeout_seconds", 60),
                        60,
                    ),
                ))

        return IterationPolicy(
            enabled=bool(raw.get("enabled", False)),
            max_attempts=self._int_or_default(raw.get("max_attempts", 4), 4),
            strategy=str(raw.get("strategy", "targeted_remediation")).strip().lower(),
            stop_on_no_improvement_attempts=self._int_or_default(
                raw.get("stop_on_no_improvement_attempts", 0),
                0,
            ),
            max_total_runner_invocations=self._int_or_default(
                raw.get("max_total_runner_invocations", 0),
                0,
            ),
            max_replans_after_exhaustion=self._int_or_default(
                raw.get("max_replans_after_exhaustion", 2),
                2,
            ),
            replan_on_exhaustion=self._bool_or_default(
                raw.get("replan_on_exhaustion", True),
                True,
            ),
            budget=budget,
            gates=gates,
        )

    @staticmethod
    def _validate_validity_contract(
        contract: dict[str, Any],
        *,
        label: str,
        errors: list[str],
    ) -> None:
        if not isinstance(contract, dict):
            errors.append(f"{label} validity_contract must be a mapping")
            return

        prune_mode = str(contract.get("prune_mode", "") or "").strip().lower()
        if prune_mode and prune_mode not in {"drop", "rewrite_uncertainty"}:
            errors.append(
                f"{label} validity_contract.prune_mode must be one of "
                "{'drop', 'rewrite_uncertainty'}",
            )

        for field_name in ("min_supported_ratio", "max_unverified_ratio"):
            raw_value = contract.get(field_name)
            if raw_value in (None, ""):
                continue
            try:
                parsed = float(raw_value)
            except (TypeError, ValueError):
                errors.append(
                    f"{label} validity_contract.{field_name} must be a number between 0 and 1",
                )
                continue
            if parsed < 0.0 or parsed > 1.0:
                errors.append(
                    f"{label} validity_contract.{field_name} must be between 0 and 1",
                )

        contradicted_raw = contract.get("max_contradicted_count")
        if contradicted_raw not in (None, ""):
            try:
                contradicted_count = int(contradicted_raw)
            except (TypeError, ValueError):
                errors.append(
                    f"{label} validity_contract.max_contradicted_count must be an integer >= 0",
                )
            else:
                if contradicted_count < 0:
                    errors.append(
                        f"{label} validity_contract.max_contradicted_count must be >= 0",
                    )

        claim_extraction = contract.get("claim_extraction", {})
        if claim_extraction not in ({}, None) and not isinstance(claim_extraction, dict):
            errors.append(f"{label} validity_contract.claim_extraction must be a mapping")

        final_gate = contract.get("final_gate", {})
        if final_gate not in ({}, None) and not isinstance(final_gate, dict):
            errors.append(f"{label} validity_contract.final_gate must be a mapping")
            return
        if isinstance(final_gate, dict):
            tier_raw = final_gate.get("synthesis_min_verification_tier")
            if tier_raw not in (None, ""):
                try:
                    tier = int(tier_raw)
                except (TypeError, ValueError):
                    errors.append(
                        f"{label} validity_contract.final_gate.synthesis_min_verification_tier "
                        "must be an integer >= 1",
                    )
                else:
                    if tier < 1:
                        errors.append(
                            f"{label} validity_contract.final_gate.synthesis_min_verification_tier "
                            "must be >= 1",
                        )
            temporal = final_gate.get("temporal_consistency", {})
            if temporal not in ({}, None) and not isinstance(temporal, dict):
                errors.append(
                    f"{label} validity_contract.final_gate.temporal_consistency must be a mapping",
                )
            if isinstance(temporal, dict):
                max_age_raw = temporal.get("max_source_age_days")
                if max_age_raw not in (None, ""):
                    try:
                        max_age = int(max_age_raw)
                    except (TypeError, ValueError):
                        errors.append(
                            f"{label} validity_contract.final_gate.temporal_consistency."
                            "max_source_age_days must be an integer >= 0",
                        )
                    else:
                        if max_age < 0:
                            errors.append(
                                f"{label} validity_contract.final_gate.temporal_consistency."
                                "max_source_age_days must be >= 0",
                            )

    @staticmethod
    def _validate_high_risk_validity_override(
        contract: dict[str, Any],
        *,
        label: str,
        errors: list[str],
    ) -> None:
        if not isinstance(contract, dict) or not contract:
            return
        if "enabled" in contract and not ProcessDefinition._to_bool(
            contract.get("enabled", True),
            True,
        ):
            errors.append(
                f"{label} high-risk validity_contract must not disable validity enforcement",
            )

        claim_extraction = contract.get("claim_extraction")
        if isinstance(claim_extraction, dict) and "enabled" in claim_extraction:
            if not ProcessDefinition._to_bool(
                claim_extraction.get("enabled", True),
                True,
            ):
                errors.append(
                    f"{label} high-risk validity_contract.claim_extraction.enabled "
                    "must be true when explicitly set",
                )

        if "max_contradicted_count" in contract:
            try:
                max_contradicted_count = int(contract.get("max_contradicted_count", 0))
            except (TypeError, ValueError):
                max_contradicted_count = 0
            if max_contradicted_count > 0:
                errors.append(
                    f"{label} high-risk validity_contract.max_contradicted_count "
                    "must be 0",
                )
        if "require_fact_checker_for_synthesis" in contract and not ProcessDefinition._to_bool(
            contract.get("require_fact_checker_for_synthesis", True),
            True,
        ):
            errors.append(
                f"{label} high-risk validity_contract.require_fact_checker_for_synthesis "
                "must be true when explicitly set",
            )

        final_gate = contract.get("final_gate")
        if not isinstance(final_gate, dict):
            return
        if "enforce_verified_context_only" in final_gate:
            if not ProcessDefinition._to_bool(
                final_gate.get("enforce_verified_context_only", True),
                True,
            ):
                errors.append(
                    f"{label} high-risk validity_contract.final_gate."
                    "enforce_verified_context_only must be true when explicitly set",
                )
        if "synthesis_min_verification_tier" in final_gate:
            try:
                min_tier = int(final_gate.get("synthesis_min_verification_tier", 2))
            except (TypeError, ValueError):
                min_tier = 2
            if min_tier < 2:
                errors.append(
                    f"{label} high-risk validity_contract.final_gate."
                    "synthesis_min_verification_tier must be >= 2",
                )
        temporal = final_gate.get("temporal_consistency")
        if isinstance(temporal, dict) and "enabled" in temporal:
            if not ProcessDefinition._to_bool(temporal.get("enabled", True), True):
                errors.append(
                    f"{label} high-risk validity_contract.final_gate.temporal_consistency."
                    "enabled must be true when explicitly set",
                )
        if isinstance(temporal, dict) and "require_as_of_alignment" in temporal:
            if not ProcessDefinition._to_bool(
                temporal.get("require_as_of_alignment", True),
                True,
            ):
                errors.append(
                    f"{label} high-risk validity_contract.final_gate.temporal_consistency."
                    "require_as_of_alignment must be true when explicitly set",
                )
        if (
            isinstance(temporal, dict)
            and "enforce_cross_claim_date_conflict_check" in temporal
            and not ProcessDefinition._to_bool(
                temporal.get("enforce_cross_claim_date_conflict_check", True),
                True,
            )
        ):
            errors.append(
                f"{label} high-risk validity_contract.final_gate.temporal_consistency."
                "enforce_cross_claim_date_conflict_check must be true when explicitly set",
            )

    def _validate(self, defn: ProcessDefinition) -> list[str]:
        """Validate a ProcessDefinition beyond schema structure."""
        errors: list[str] = []

        # Required fields
        if not defn.name:
            errors.append("Missing required field: name")

        if int(defn.schema_version or 1) < 1:
            errors.append("schema_version must be >= 1")

        if str(defn.risk_level or "").strip():
            normalized_risk = ProcessDefinition._normalize_risk_level(defn.risk_level)
            if not normalized_risk:
                errors.append(
                    "risk_level must be one of {'low', 'medium', 'high', 'critical'} "
                    "when provided",
                )
            else:
                defn.risk_level = normalized_risk

        if self._require_v2_contract and not defn.is_contract_v2():
            errors.append(
                "schema_version must be 2 when strict contract-v2 validation is enabled",
            )

        # Name format
        if defn.name and not re.match(r"^[a-z0-9][a-z0-9-]*$", defn.name):
            errors.append(
                f"Invalid name {defn.name!r}: must be lowercase "
                f"alphanumeric with hyphens",
            )

        # Phase mode
        valid_modes = {"strict", "guided", "suggestive"}
        if defn.phase_mode not in valid_modes:
            errors.append(
                f"Invalid phase_mode {defn.phase_mode!r}: "
                f"must be one of {valid_modes}",
            )

        output_coordination = defn.output_coordination
        strategy = str(output_coordination.strategy or "").strip().lower() or "direct"
        if strategy not in ProcessDefinition._OUTPUT_STRATEGIES:
            errors.append(
                "output_coordination.strategy must be one of {'direct', 'fan_in'}",
            )
        else:
            output_coordination.strategy = strategy

        publish_mode = (
            str(output_coordination.publish_mode or "").strip().lower() or "transactional"
        )
        if publish_mode not in ProcessDefinition._OUTPUT_PUBLISH_MODES:
            errors.append(
                "output_coordination.publish_mode must be one of "
                "{'transactional', 'best_effort'}",
            )
        else:
            output_coordination.publish_mode = publish_mode

        conflict_policy = (
            str(output_coordination.conflict_policy or "").strip().lower() or "defer_fifo"
        )
        if conflict_policy not in ProcessDefinition._OUTPUT_CONFLICT_POLICIES:
            errors.append(
                "output_coordination.conflict_policy must be one of "
                "{'defer_fifo', 'fail_fast'}",
            )
        else:
            output_coordination.conflict_policy = conflict_policy

        finalizer_input_policy = (
            str(output_coordination.finalizer_input_policy or "").strip().lower()
            or "require_all_workers"
        )
        if finalizer_input_policy not in ProcessDefinition._FINALIZER_INPUT_POLICIES:
            errors.append(
                "output_coordination.finalizer_input_policy must be one of "
                "{'require_all_workers', 'allow_partial'}",
            )
        else:
            output_coordination.finalizer_input_policy = finalizer_input_policy

        normalized_intermediate_root, root_error = self._normalize_workspace_relative_path(
            output_coordination.intermediate_root,
        )
        if root_error:
            errors.append(
                "output_coordination.intermediate_root "
                f"{root_error}",
            )
        else:
            output_coordination.intermediate_root = normalized_intermediate_root

        if defn.is_contract_v2() or self._require_v2_contract:
            policy_mode = str(defn.verification_policy.mode or "").strip().lower()
            if policy_mode not in {"llm_first", "static_first"}:
                errors.append(
                    "verification.policy.mode must be one of "
                    "{'llm_first', 'static_first'} for schema_version >= 2",
                )

            required_fields = defn.verifier_required_response_fields()
            if "passed" not in required_fields:
                errors.append(
                    "verification.policy.output_contract.required_fields "
                    "must include 'passed'",
                )
            if "metadata" not in required_fields and defn.verifier_metadata_fields():
                errors.append(
                    "verification.policy.output_contract.required_fields must include "
                    "'metadata' when metadata_fields/counters/flags are declared",
                )

            for contract in defn.verifier_semantic_check_contracts():
                helper = str(contract.get("helper", "") or "").strip().lower()
                capability = str(contract.get("capability", "") or "").strip().lower()
                if helper:
                    helper_spec = get_verification_helper(helper)
                    if helper_spec is None:
                        errors.append(
                            "verification.policy.semantic_checks helper "
                            f"{helper!r} is not registered",
                        )
                    elif capability and capability not in helper_spec.capabilities:
                        supported = ", ".join(helper_spec.capabilities)
                        errors.append(
                            "verification.policy.semantic_checks helper "
                            f"{helper!r} does not support capability {capability!r}; "
                            f"supported capabilities: {supported}",
                        )

            evidence_required = defn.evidence.record_schema.get("required_fields", [])
            if evidence_required and not isinstance(evidence_required, list):
                errors.append(
                    "evidence.record_schema.required_fields must be a list when provided",
                )
            evidence_facets = defn.evidence.record_schema.get("facets", [])
            if evidence_facets and not isinstance(evidence_facets, list):
                errors.append(
                    "evidence.record_schema.facets must be a list when provided",
                )

            critical_path_behavior = str(
                defn.verification_remediation.critical_path_behavior or "",
            ).strip().lower()
            if critical_path_behavior and critical_path_behavior not in {
                "block",
                "confirm_or_prune_then_queue",
                "queue_follow_up",
            }:
                errors.append(
                    "verification.remediation.critical_path_behavior must be one of "
                    "{'block', 'confirm_or_prune_then_queue', 'queue_follow_up'}",
                )

        if isinstance(defn.validity_contract, dict) and defn.validity_contract:
            self._validate_validity_contract(
                defn.validity_contract,
                label="process",
                errors=errors,
            )
        if defn._is_high_risk_intent():
            self._validate_high_risk_validity_override(
                defn.validity_contract,
                label="process",
                errors=errors,
            )

        # Phase validation
        phase_ids = set()
        for phase in defn.phases:
            if not phase.id:
                errors.append("Phase missing required field: id")
                continue
            if phase.id in phase_ids:
                errors.append(f"Duplicate phase id: {phase.id!r}")
            phase_ids.add(phase.id)
            if not phase.description:
                errors.append(f"Phase {phase.id!r} has empty description")
            output_strategy = str(getattr(phase, "output_strategy", "") or "").strip().lower()
            if output_strategy and output_strategy not in ProcessDefinition._OUTPUT_STRATEGIES:
                errors.append(
                    f"Phase {phase.id!r}: output_strategy must be one of "
                    "{'direct', 'fan_in'} when provided",
                )
            else:
                phase.output_strategy = output_strategy
            finalizer_input_policy = str(
                getattr(phase, "finalizer_input_policy", "") or "",
            ).strip().lower()
            if (
                finalizer_input_policy
                and finalizer_input_policy not in ProcessDefinition._FINALIZER_INPUT_POLICIES
            ):
                errors.append(
                    f"Phase {phase.id!r}: finalizer_input_policy must be one of "
                    "{'require_all_workers', 'allow_partial'} when provided",
                )
            else:
                phase.finalizer_input_policy = finalizer_input_policy
            if isinstance(phase.validity_contract, dict) and phase.validity_contract:
                self._validate_validity_contract(
                    phase.validity_contract,
                    label=f"phase {phase.id!r}",
                    errors=errors,
                )
            if defn._is_high_risk_intent() and bool(getattr(phase, "is_synthesis", False)):
                self._validate_high_risk_validity_override(
                    phase.validity_contract,
                    label=f"phase {phase.id!r}",
                    errors=errors,
                )
            self._validate_iteration_policy(phase, errors)

        phase_finalizer_owners: dict[str, str] = {}
        for phase in defn.phases:
            phase_id = str(getattr(phase, "id", "") or "").strip()
            if not phase_id:
                continue
            if not bool(getattr(phase, "deliverables", [])):
                continue
            if defn.phase_output_strategy(phase_id) != "fan_in":
                continue
            finalizer_id = ProcessDefinition.phase_finalizer_id(phase_id)
            if not finalizer_id:
                continue
            if finalizer_id in phase_ids:
                errors.append(
                    f"Phase {phase_id!r}: fan_in finalizer id {finalizer_id!r} "
                    "collides with declared phase id; rename the phase id",
                )
            owner_phase = phase_finalizer_owners.get(finalizer_id, "")
            if owner_phase and owner_phase != phase_id:
                errors.append(
                    f"fan_in finalizer id collision: {owner_phase!r} and "
                    f"{phase_id!r} both resolve to {finalizer_id!r}",
                )
            else:
                phase_finalizer_owners[finalizer_id] = phase_id

        if defn._is_high_risk_intent():
            for phase in defn.phases:
                if not bool(getattr(phase, "is_synthesis", False)):
                    continue
                resolved = defn.resolve_validity_contract_for_phase(
                    str(getattr(phase, "id", "") or ""),
                    is_synthesis=True,
                )
                claim_extraction = resolved.get("claim_extraction", {})
                final_gate = resolved.get("final_gate", {})
                if not ProcessDefinition._to_bool(resolved.get("enabled", False), False):
                    errors.append(
                        f"phase {phase.id!r} high-risk resolved validity contract disabled",
                    )
                if not (
                    isinstance(claim_extraction, dict)
                    and ProcessDefinition._to_bool(
                        claim_extraction.get("enabled", False),
                        False,
                    )
                ):
                    errors.append(
                        f"phase {phase.id!r} high-risk resolved contract must enable "
                        "claim extraction",
                    )
                if int(resolved.get("max_contradicted_count", 0) or 0) > 0:
                    errors.append(
                        f"phase {phase.id!r} high-risk resolved contract must enforce "
                        "max_contradicted_count=0",
                    )
                if not (
                    isinstance(final_gate, dict)
                    and ProcessDefinition._to_bool(
                        final_gate.get("enforce_verified_context_only", False),
                        False,
                    )
                ):
                    errors.append(
                        f"phase {phase.id!r} high-risk resolved contract must enforce "
                        "verified-context-only synthesis",
                    )
                if int(final_gate.get("synthesis_min_verification_tier", 1) or 1) < 2:
                    errors.append(
                        f"phase {phase.id!r} high-risk resolved contract must require "
                        "synthesis_min_verification_tier >= 2",
                    )
                if not ProcessDefinition._to_bool(
                    resolved.get("require_fact_checker_for_synthesis", False),
                    False,
                ):
                    errors.append(
                        f"phase {phase.id!r} high-risk resolved contract must require "
                        "fact_checker for synthesis",
                    )
                temporal = final_gate.get("temporal_consistency", {})
                if not (
                    isinstance(temporal, dict)
                    and ProcessDefinition._to_bool(temporal.get("enabled", False), False)
                ):
                    errors.append(
                        f"phase {phase.id!r} high-risk resolved contract must enable "
                        "temporal consistency gate",
                    )
                if not (
                    isinstance(temporal, dict)
                    and ProcessDefinition._to_bool(
                        temporal.get("require_as_of_alignment", False),
                        False,
                    )
                ):
                    errors.append(
                        f"phase {phase.id!r} high-risk resolved contract must require "
                        "as_of alignment",
                    )

        # Dependency graph validation
        for phase in defn.phases:
            for dep in phase.depends_on:
                if dep not in phase_ids:
                    errors.append(
                        f"Phase {phase.id!r} depends on unknown phase "
                        f"{dep!r}",
                    )

        # Cycle detection (topological sort)
        cycle = self._detect_cycles(defn.phases)
        if cycle:
            errors.append(
                f"Dependency cycle detected: {' -> '.join(cycle)}",
            )

        # Deliverable uniqueness
        seen_deliverables: dict[str, str] = {}
        for phase in defn.phases:
            for d in phase.deliverables:
                fname = d.split("—")[0].split(" — ")[0].strip()
                if fname in seen_deliverables:
                    errors.append(
                        f"Duplicate deliverable {fname!r} in phases "
                        f"{seen_deliverables[fname]!r} and {phase.id!r}",
                    )
                seen_deliverables[fname] = phase.id

        # Tool requirements: required and excluded must not overlap
        if defn.tools.required and defn.tools.excluded:
            overlap = set(defn.tools.required) & set(defn.tools.excluded)
            if overlap:
                errors.append(
                    f"Tool(s) both required and excluded: "
                    f"{', '.join(sorted(overlap))}",
                )

        # Auth requirements
        for requirement in defn.auth.required:
            provider = str(requirement.provider or "").strip()
            resource_ref = str(requirement.resource_ref or "").strip()
            resource_id = str(requirement.resource_id or "").strip()
            if not provider and not resource_ref and not resource_id:
                errors.append(
                    "auth.required entry missing provider/resource_ref/resource_id"
                )
                continue
            source = str(requirement.source or "api").strip().lower()
            if source not in {"api", "mcp"}:
                errors.append(
                    f"auth.required entry {provider or resource_ref or resource_id!r}: "
                    f"invalid source {source!r} "
                    "(expected 'api' or 'mcp')",
                )
            if source != "mcp" and str(requirement.mcp_server or "").strip():
                errors.append(
                    f"auth.required entry {provider or resource_ref or resource_id!r}: "
                    "mcp_server is only valid "
                    "when source='mcp'",
                )
            if resource_ref and ":" not in resource_ref:
                errors.append(
                    f"auth.required entry {provider or resource_ref or resource_id!r}: "
                    "resource_ref must look like '<kind>:<key>'",
                )
            for mode in requirement.modes:
                if not str(mode).strip():
                    errors.append(
                        f"auth.required entry {provider or resource_ref or resource_id!r}: "
                        "modes must not contain empty values",
                    )
            for env_key in requirement.required_env_keys:
                key = str(env_key or "").strip()
                if not key:
                    errors.append(
                        f"auth.required entry {provider or resource_ref or resource_id!r}: "
                        "required_env_keys "
                        "must not contain empty values",
                    )
                    continue
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                    errors.append(
                        f"auth.required entry {provider or resource_ref or resource_id!r}: "
                        f"invalid env key {key!r}",
                    )

        # Process test validation
        seen_test_ids: set[str] = set()
        valid_test_modes = {"deterministic", "live"}
        for test_case in defn.tests:
            if not test_case.id:
                errors.append("Process test case missing required field: id")
            elif test_case.id in seen_test_ids:
                errors.append(
                    f"Duplicate process test id: {test_case.id!r}",
                )
            else:
                seen_test_ids.add(test_case.id)

            if test_case.mode not in valid_test_modes:
                errors.append(
                    f"Process test {test_case.id!r}: invalid mode "
                    f"{test_case.mode!r} (expected one of {valid_test_modes})",
                )

            if not test_case.goal:
                errors.append(
                    f"Process test {test_case.id!r} missing required field: goal",
                )

            if test_case.timeout_seconds <= 0:
                errors.append(
                    f"Process test {test_case.id!r}: timeout_seconds must be > 0",
                )

        # Verification rule validation
        for rule in defn.verification_rules:
            if not rule.name:
                errors.append("Verification rule missing name")
            if not rule.check:
                errors.append(
                    f"Verification rule {rule.name!r} missing check",
                )
            if rule.severity not in ("warning", "error"):
                errors.append(
                    f"Rule {rule.name!r}: invalid severity "
                    f"{rule.severity!r}",
                )
            if rule.type not in ("llm", "regex"):
                errors.append(
                    f"Rule {rule.name!r}: invalid type {rule.type!r}",
                )
            if rule.enforcement and rule.enforcement not in ("hard", "advisory"):
                errors.append(
                    f"Rule {rule.name!r}: invalid enforcement {rule.enforcement!r}",
                )
            if rule.failure_class and rule.failure_class not in (
                "hard_integrity",
                "recoverable_placeholder",
                "semantic",
            ):
                errors.append(
                    f"Rule {rule.name!r}: invalid failure_class "
                    f"{rule.failure_class!r}",
                )
            if rule.remediation_mode and rule.remediation_mode not in (
                "none",
                "confirm_or_prune",
                "targeted_remediation",
                "queue_follow_up",
                "remediate_and_retry",
            ):
                errors.append(
                    f"Rule {rule.name!r}: invalid remediation_mode "
                    f"{rule.remediation_mode!r}",
                )
            if rule.scope and rule.scope not in ("phase", "global"):
                errors.append(
                    f"Rule {rule.name!r}: invalid scope {rule.scope!r}",
                )
            if self._require_rule_scope_metadata:
                has_scope = bool(str(rule.scope or "").strip())
                has_explicit_phases = bool(
                    [p for p in (rule.applies_to_phases or []) if str(p).strip()]
                )
                if not has_scope and not has_explicit_phases:
                    errors.append(
                        f"Rule {rule.name!r}: missing scope metadata "
                        "(set scope or applies_to_phases)",
                    )
            for phase_id in rule.applies_to_phases:
                if phase_id == "*":
                    continue
                if phase_id not in phase_ids:
                    errors.append(
                        f"Rule {rule.name!r}: applies_to_phases contains unknown phase "
                        f"{phase_id!r}",
                    )
            if rule.min_count is not None:
                try:
                    min_count = int(rule.min_count)
                except (TypeError, ValueError):
                    errors.append(
                        f"Rule {rule.name!r}: min_count must be an integer",
                    )
                else:
                    if min_count < 0:
                        errors.append(
                            f"Rule {rule.name!r}: min_count must be >= 0",
                        )
            # Validate regex rules compile
            if rule.type == "regex":
                try:
                    re.compile(rule.check)
                except re.error as e:
                    errors.append(
                        f"Rule {rule.name!r}: invalid regex: {e}",
                    )

        return errors

    def _validate_iteration_policy(
        self,
        phase: PhaseTemplate,
        errors: list[str],
    ) -> None:
        policy = phase.iteration
        if policy is None or not bool(policy.enabled):
            return

        phase_id = phase.id or "<unknown>"
        gate_types = {"tool_metric", "command_exit", "artifact_regex", "verifier_field"}
        deterministic_types = {"tool_metric", "command_exit", "artifact_regex"}
        operators = {"gte", "lte", "eq", "contains", "not_contains"}

        if policy.max_attempts < 1 or policy.max_attempts > 6:
            errors.append(
                f"Phase {phase_id!r}: iteration.max_attempts must be between 1 and 6",
            )
        if policy.strategy not in self._ITERATION_STRATEGIES:
            errors.append(
                f"Phase {phase_id!r}: iteration.strategy must be one of "
                "{targeted_remediation,full_rerun}",
            )
        if policy.stop_on_no_improvement_attempts < 0:
            errors.append(
                f"Phase {phase_id!r}: iteration.stop_on_no_improvement_attempts "
                "must be >= 0",
            )
        if policy.max_replans_after_exhaustion < 0:
            errors.append(
                f"Phase {phase_id!r}: iteration.max_replans_after_exhaustion "
                "must be >= 0",
            )
        if policy.max_total_runner_invocations <= 0:
            errors.append(
                f"Phase {phase_id!r}: iteration.max_total_runner_invocations "
                "must be > 0",
            )
        elif policy.max_total_runner_invocations < policy.max_attempts:
            errors.append(
                f"Phase {phase_id!r}: iteration.max_total_runner_invocations must be "
                ">= iteration.max_attempts",
            )
        if policy.budget.max_wall_clock_seconds <= 0:
            errors.append(
                f"Phase {phase_id!r}: iteration.iteration_budget.max_wall_clock_seconds "
                "must be > 0",
            )
        if policy.budget.max_tokens <= 0:
            errors.append(
                f"Phase {phase_id!r}: iteration.iteration_budget.max_tokens must be > 0",
            )
        if policy.budget.max_tool_calls <= 0:
            errors.append(
                f"Phase {phase_id!r}: iteration.iteration_budget.max_tool_calls "
                "must be > 0",
            )
        if not policy.gates:
            errors.append(
                f"Phase {phase_id!r}: iteration.gates must declare at least one gate",
            )
            return

        seen_gate_ids: set[str] = set()
        deterministic_blockers = 0
        score_like_gate = False
        shell_meta_pattern = re.compile(r"[;&|><`]")
        for gate in policy.gates:
            gate_id = str(gate.id or "").strip()
            if not gate_id:
                errors.append(
                    f"Phase {phase_id!r}: iteration gate missing id",
                )
                continue
            if gate_id in seen_gate_ids:
                errors.append(
                    f"Phase {phase_id!r}: duplicate iteration gate id {gate_id!r}",
                )
            seen_gate_ids.add(gate_id)

            gate_type = str(gate.type or "").strip().lower()
            if gate_type not in gate_types:
                errors.append(
                    f"Phase {phase_id!r}: gate {gate_id!r} has invalid type "
                    f"{gate.type!r}",
                )
                continue

            if gate.blocking and gate_type in deterministic_types:
                deterministic_blockers += 1
            if gate_type == "verifier_field" and gate.blocking:
                errors.append(
                    f"Phase {phase_id!r}: gate {gate_id!r} of type verifier_field "
                    "is advisory-only in MVP and cannot be blocking",
                )

            operator = str(gate.operator or "").strip().lower()
            if gate_type in {"tool_metric", "verifier_field"} and operator not in operators:
                errors.append(
                    f"Phase {phase_id!r}: gate {gate_id!r} requires operator in "
                    "{gte,lte,eq,contains,not_contains}",
                )

            if gate_type == "tool_metric":
                if not str(gate.tool or "").strip():
                    errors.append(
                        f"Phase {phase_id!r}: gate {gate_id!r} missing tool",
                    )
                if not str(gate.metric_path or "").strip():
                    errors.append(
                        f"Phase {phase_id!r}: gate {gate_id!r} missing metric_path",
                    )
                path_hint = " ".join([
                    str(gate.metric_path or "").strip().lower(),
                    gate_id.lower(),
                ])
                if "score" in path_hint and isinstance(gate.value, (int, float)):
                    score_like_gate = True

            if gate_type == "artifact_regex":
                if not str(gate.pattern or "").strip():
                    errors.append(
                        f"Phase {phase_id!r}: gate {gate_id!r} missing pattern",
                    )
                else:
                    try:
                        re.compile(str(gate.pattern))
                    except re.error as e:
                        errors.append(
                            f"Phase {phase_id!r}: gate {gate_id!r} has invalid regex: {e}",
                        )
                target = str(gate.target or "").strip().lower()
                if target not in self._ARTIFACT_REGEX_TARGETS:
                    errors.append(
                        f"Phase {phase_id!r}: gate {gate_id!r} target must be one of "
                        "{auto,deliverables,changed_files,summary,output}",
                    )
                if target == "deliverables" and not phase.deliverables:
                    errors.append(
                        f"Phase {phase_id!r}: gate {gate_id!r} targets deliverables "
                        "but phase has no deliverables declared",
                    )

            if gate_type == "command_exit":
                if not gate.command:
                    errors.append(
                        f"Phase {phase_id!r}: gate {gate_id!r} missing command",
                    )
                if gate.timeout_seconds <= 0:
                    errors.append(
                        f"Phase {phase_id!r}: gate {gate_id!r} timeout_seconds "
                        "must be > 0",
                    )
                for token in gate.command:
                    text = str(token or "")
                    if (
                        shell_meta_pattern.search(text)
                        or "$(" in text
                        or "||" in text
                        or "&&" in text
                    ):
                        errors.append(
                            f"Phase {phase_id!r}: gate {gate_id!r} command contains "
                            "shell metacharacters; provide argv tokens only",
                        )
                        break
                if gate.command and not self._command_exit_prefix_allowlisted(gate.command):
                    errors.append(
                        f"Phase {phase_id!r}: gate {gate_id!r} command is not allowlisted "
                        "for MVP command_exit gates",
                    )

            if gate_type == "verifier_field" and not str(gate.verifier_field or "").strip():
                errors.append(
                    f"Phase {phase_id!r}: gate {gate_id!r} missing field for verifier_field",
                )

        if deterministic_blockers == 0:
            errors.append(
                f"Phase {phase_id!r}: iteration requires at least one deterministic "
                "blocking gate",
            )
        if score_like_gate and policy.stop_on_no_improvement_attempts <= 0:
            errors.append(
                f"Phase {phase_id!r}: score-like iteration gates require "
                "stop_on_no_improvement_attempts > 0",
            )
        if (
            policy.stop_on_no_improvement_attempts > 0
            and policy.stop_on_no_improvement_attempts >= policy.max_attempts
        ):
            errors.append(
                f"Phase {phase_id!r}: stop_on_no_improvement_attempts must be < "
                "max_attempts",
            )

    def _detect_cycles(
        self, phases: list[PhaseTemplate],
    ) -> list[str] | None:
        """Detect dependency cycles using DFS. Returns cycle path or None."""
        adj: dict[str, list[str]] = {p.id: p.depends_on for p in phases}
        _white, _gray, _black = 0, 1, 2
        color: dict[str, int] = {pid: _white for pid in adj}
        parent: dict[str, str | None] = {pid: None for pid in adj}

        def dfs(node: str) -> list[str] | None:
            color[node] = _gray
            for neighbor in adj.get(node, []):
                if neighbor not in color:
                    continue
                if color[neighbor] == _gray:
                    # Found a cycle — reconstruct path
                    cycle = [neighbor, node]
                    cur: str | None = node
                    while cur and parent.get(cur) and parent[cur] != neighbor:
                        cur = parent[cur]
                        if cur:
                            cycle.append(cur)
                    cycle.append(neighbor)
                    return list(reversed(cycle))
                if color[neighbor] == _white:
                    parent[neighbor] = node
                    result = dfs(neighbor)
                    if result:
                        return result
            color[node] = _black
            return None

        for node in adj:
            if color[node] == _white:
                result = dfs(node)
                if result:
                    return result
        return None

    @staticmethod
    def _quick_parse_name(yaml_file: Path) -> str | None:
        """Extract just the name field from a YAML file (fast)."""
        try:
            with open(yaml_file) as f:
                # Read just enough to find the name field
                for line in f:
                    line = line.strip()
                    if line.startswith("name:"):
                        val = line.split(":", 1)[1].strip()
                        # Strip quotes
                        return val.strip("'\"") or None
                    if line.startswith("#") or not line:
                        continue
                    # If we hit non-name content, fall back to full parse
                    break
            # Full parse fallback
            with open(yaml_file) as f:
                raw = yaml.safe_load(f)
            if raw and isinstance(raw, dict):
                return raw.get("name")
            return None
        except Exception:
            return None

    @staticmethod
    def _activate_isolated_dependencies(package_dir: Path) -> None:
        """Add isolated package dependency paths to sys.path if available."""
        deps_root = package_dir.parent / ".deps" / package_dir.name
        if not deps_root.exists():
            return

        candidates: list[Path] = []
        if sys.platform.startswith("win"):
            candidates.append(deps_root / "Lib" / "site-packages")
        else:
            lib_root = deps_root / "lib"
            for site_dir in sorted(lib_root.glob("python*/site-packages")):
                candidates.append(site_dir)

        for candidate in candidates:
            if candidate.exists():
                path_str = str(candidate)
                if path_str not in sys.path:
                    sys.path.insert(0, path_str)
                    logger.info(
                        "Activated isolated process dependencies: %s",
                        candidate,
                    )

    @staticmethod
    def _tool_name_from_class(tool_cls: type) -> str | None:
        """Best-effort extraction of a Tool name from a Tool subclass."""
        try:
            tool = tool_cls()
            name = getattr(tool, "name", "")
            if isinstance(name, str) and name.strip():
                return name.strip()
        except Exception:
            return None
        return None

    @staticmethod
    def _registered_tool_name_map() -> dict[str, type]:
        """Return {tool_name: class} for currently registered tool classes."""
        from loom.tools.registry import Tool

        name_map: dict[str, type] = {}
        for cls in Tool._registered_classes:
            name = ProcessLoader._tool_name_from_class(cls)
            if name:
                name_map[name] = cls
        return name_map

    @staticmethod
    def _register_bundled_tools(package_dir: Path) -> None:
        """Import and register tools from a package's tools/ directory."""
        from loom.tools.registry import Tool

        tools_dir = package_dir / "tools"
        if not tools_dir.exists() or not tools_dir.is_dir():
            return
        for py_file in tools_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            before_classes = set(Tool._registered_classes)
            before_name_map = ProcessLoader._registered_tool_name_map()
            # Include package name to avoid collisions between processes
            safe_pkg = package_dir.name.replace("-", "_")
            module_name = f"loom.processes._bundled.{safe_pkg}.{py_file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(
                    module_name, py_file,
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)

                    # Detect and suppress bundled tools that collide on name
                    # with already-registered tools.
                    new_classes = [
                        cls
                        for cls in Tool._registered_classes
                        if cls not in before_classes
                    ]
                    name_owner: dict[str, type] = dict(before_name_map)
                    for cls in new_classes:
                        tool_name = ProcessLoader._tool_name_from_class(cls)
                        if not tool_name:
                            continue
                        existing = name_owner.get(tool_name)
                        if existing is None:
                            name_owner[tool_name] = cls
                            continue
                        Tool._registered_classes.discard(cls)
                        logger.warning(
                            "Bundled tool '%s' from %s conflicts with "
                            "existing tool class %s.%s; skipping bundled tool. "
                            "Rename bundled tools to unique names.",
                            tool_name,
                            py_file,
                            existing.__module__,
                            existing.__name__,
                        )
            except Exception as e:
                logger.warning(
                    "Failed to load bundled tool %s: %s", py_file, e,
                )
