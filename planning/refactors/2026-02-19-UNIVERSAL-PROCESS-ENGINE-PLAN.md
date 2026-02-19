# Universal Process Engine Refactor Plan (2026-02-19)

## Objective
Refactor Loom's process subsystem so the core engine can execute any process definition without domain-specific assumptions in runtime code.

Hard requirement:
1. Process-specific behavior must be declared in process definition YAML.
2. Core engine must contain no process-domain logic (market, finance, legal, etc.).
3. Verification must be LLM-first for semantic and structure interpretation of LLM-produced data.
4. Deterministic/static checks remain only for safety, integrity, and declared hard invariants.

## Non-Negotiable Invariants
1. No hardcoded domain semantics in core runtime modules.
2. No fixed output schema assumptions in verifier unless explicitly declared by the loaded process.
3. No hardcoded remediation wording/structure in orchestrator.
4. No retry targeting keyed to domain labels (for example, "market") in retry manager.
5. Base prompt templates stay generic; process-specific instructions are injected from YAML.

## Current Violations to Eliminate
1. Domain-specific evidence inference in core: `/Users/sfw/Development/loom/src/loom/state/evidence.py`
2. Domain-specific evidence snapshot formatting and policy enforcement in verifier core: `/Users/sfw/Development/loom/src/loom/engine/verification.py`
3. Domain-specific retry targeting fields and parsing: `/Users/sfw/Development/loom/src/loom/recovery/retry.py`
4. Hardcoded synthesis/remediation phrasing in orchestrator: `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`
5. Domain-specific verifier template constraints: `/Users/sfw/Development/loom/src/loom/prompts/templates/verifier.yaml`
6. Domain keyword heuristics for evidence contract activation: `/Users/sfw/Development/loom/src/loom/prompts/assembler.py`

## Target End State
Core engine responsibilities:
1. Execute process phases, dependencies, retries, approvals, persistence.
2. Run safety/integrity checks.
3. Invoke LLM-based verification with process-provided verification contract.
4. Route remediation using process-defined strategy metadata.

Process YAML responsibilities:
1. Phase definitions and critical-path designation.
2. Evidence model and extraction mapping.
3. Verification semantic checks and output policy.
4. Remediation strategy and escalation policy.
5. Output partitioning/section rules for synthesis-like phases.

## Process Contract v2 (Schema Expansion)
Introduce schema versioned process contract with explicit behavior blocks.

### 1) Top-Level
1. `schema_version: 2`
2. `name`, `version`, `description`, `phase_mode`, `phases`, existing fields retained.

### 2) `verification.policy`
1. `mode`: `llm_first` | `static_first`
2. `static_checks`:
   - `tool_success_policy`
   - `file_integrity_policy`
   - `syntax_policy`
   - `deliverable_existence_policy`
3. `semantic_checks`:
   - list of process-defined LLM checks with `id`, `check`, `severity`, `enforcement`, `scope`
4. `output_contract`:
   - verifier response fields required by this process
   - optional process-specific counters/flags
5. `outcome_policy`:
   - pass/fail/partial behavior thresholds
   - recommendation/supporting partition policy if needed

### 3) `verification.remediation`
1. `strategies`: process-defined retry/remediation strategies
2. `default_strategy`
3. `critical_path_behavior`
4. `retry_budget`
5. `transient_retry_policy`

### 4) `evidence`
1. `record_schema`:
   - required generic fields (id, source, timestamp, quality)
   - process-specific `facets` list
2. `extraction`:
   - arg-path mappings from tool calls
   - optional LLM extraction for loose structures
3. `summarization`:
   - how evidence context is summarized for verifier prompts

### 5) `prompt_contracts`
1. `executor_constraints`
2. `verifier_constraints`
3. `remediation_instructions`

## Core Refactor Workstreams

### W1: Evidence Layer Generalization
Scope:
1. Replace fixed `market`/`dimension` fields with generic `facets: {key:value}`.
2. Remove `_infer_market` and domain dimension heuristics from core.
3. Build process-driven extractor using YAML mappings and optional LLM assist.

Files:
1. `/Users/sfw/Development/loom/src/loom/state/evidence.py`
2. `/Users/sfw/Development/loom/src/loom/processes/schema.py`

Exit criteria:
1. Core evidence code references no process-domain terms.
2. Evidence extraction behavior changes only via YAML contract.

### W2: Verification Engine Dehardcoding + LLM-First Policy
Scope:
1. Move semantic policies (unconfirmed thresholds, section partition rules, recommendation logic) out of core into process policy.
2. Keep static checks to safety/integrity only.
3. Evaluate process semantic checks via verifier LLM contract.
4. Preserve tiering/voting framework but make content policy YAML-driven.

Files:
1. `/Users/sfw/Development/loom/src/loom/engine/verification.py`
2. `/Users/sfw/Development/loom/src/loom/prompts/templates/verifier.yaml`
3. `/Users/sfw/Development/loom/src/loom/prompts/assembler.py`

Exit criteria:
1. No process-specific rule text in verifier core.
2. Base verifier template is domain-agnostic.
3. LLM verifier can handle variable shapes without static schema failure.

### W3: Retry/Remediation Generic Contract
Scope:
1. Replace `missing_markets` with generic targeting metadata (for example, `missing_targets`).
2. Remove text parsing keyed to domain words.
3. Classify failures by structured reason code + remediation metadata.
4. Process YAML defines strategy mapping.

Files:
1. `/Users/sfw/Development/loom/src/loom/recovery/retry.py`
2. `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`

Exit criteria:
1. Retry manager has no domain-specific parsing patterns.
2. Remediation strategies are process-configurable.

### W4: Prompt System Hard Separation (Core vs Process)
Scope:
1. Keep shared template constraints generic only.
2. Inject process-specific verifier/executor constraints from YAML.
3. Remove keyword heuristics for evidence contract triggers; use explicit process flags/rules.

Files:
1. `/Users/sfw/Development/loom/src/loom/prompts/assembler.py`
2. `/Users/sfw/Development/loom/src/loom/prompts/templates/verifier.yaml`

Exit criteria:
1. Domain behavior in prompts comes exclusively from loaded process definition.

### W5: Process Schema + Loader Validation
Scope:
1. Add schema validation for new contract blocks.
2. Add strict validation mode to require explicit verification policy and evidence model metadata.
3. Provide compatibility adapter for legacy v1 process files.

Files:
1. `/Users/sfw/Development/loom/src/loom/processes/schema.py`
2. `/Users/sfw/Development/loom/src/loom/config.py`

Exit criteria:
1. Invalid process contracts fail fast with actionable errors.
2. Legacy definitions can run under compatibility mode.

### W6: Built-In Process Migration
Scope:
1. Migrate built-in YAML definitions to contract v2.
2. Provide migration path for existing user/custom process definitions (v1 -> v2).
3. Encode prior behavior as data, not code.

Files:
1. `/Users/sfw/Development/loom/src/loom/processes/builtin/*.yaml`
2. `/Users/sfw/Development/loom/src/loom/processes/schema.py` (compat adapter + migration validation)

Exit criteria:
1. All built-ins run without relying on legacy hardcoded core semantics.
2. Existing v1 process definitions have a documented and test-covered migration path to v2.

### W7: Process Definition Documentation + Authoring Guide Refresh
Scope:
1. Update docs to define process contract v2 as the source of all process behavior.
2. Document LLM-first verification model and limits of static checks.
3. Add explicit migration guide for existing process YAML definitions.
4. Update examples to show how process-specific behavior is declared in YAML (not core code).

Files:
1. `/Users/sfw/Development/loom/docs/creating-packages.md`
2. `/Users/sfw/Development/loom/docs/agent-integration.md`
3. `/Users/sfw/Development/loom/README.md`

Exit criteria:
1. Process author docs and examples match the shipped v2 schema.
2. A maintainer can author a new process without touching core runtime code.
3. Migration steps from v1 to v2 are explicit and validated by docs examples/tests.

## LLM-First Verification Policy (Global Direction)
1. Tier 1 static checks limited to:
   - tool execution integrity
   - artifact confinement/safety
   - file existence/non-empty when declared
   - syntax parse validity where applicable
2. Tier 2 LLM checks own:
   - semantic correctness
   - loose/dynamic data structure interpretation
   - schema conformance relative to process-declared expectations
   - evidence sufficiency judgments
3. Tier 3 voting remains optional for high-criticality workflows.
4. Any domain-specific acceptance logic must be process-defined YAML rules interpreted by verifier.

## Migration and Compatibility Strategy

### Phase A: Introduce Contract v2 in Parallel (No Cutover)
1. Add schema and loader support.
2. Keep existing runtime behavior behind compatibility path.
3. Add instrumentation to compare legacy vs v2 decision outcomes.

### Phase B: Dehardcode Core + Shadow Mode
1. Core starts consuming v2 policy first when available.
2. Legacy hardcoded logic runs shadow-only for diff telemetry.
3. Log reason-code and outcome deltas.

### Phase C: Cutover
1. Enable v2 path by default.
2. Disable legacy domain hardcoding path.
3. Keep rollback flag for one release cycle.

### Phase D: Legacy Removal
1. Delete compatibility shims and dead code paths.
2. Enforce v2 schema for built-ins and new process definitions.

## PR Sequence (Hard Order)
1. PR-U1: Process contract v2 schema + loader validation + feature flags.
2. PR-U2: Evidence layer generalization (`facets` model, YAML-driven extraction).
3. PR-U3: Verifier prompt/template dehardcoding + process-injected constraints.
4. PR-U4: Verification engine policy externalization to process contract.
5. PR-U5: Retry/remediation generic contract and targeting metadata refactor.
6. PR-U6: Orchestrator remediation prompt dehardcoding.
7. PR-U7: Built-in YAML migration to v2.
8. PR-U8: Shadow diff telemetry + rollout controls.
9. PR-U9: Cutover defaults to v2 + legacy path deprecated.
10. PR-U10: Legacy hardcoded behavior removal and cleanup.
11. PR-U11: Documentation refresh for process contract v2 + migration guide.

## Test Strategy

### Unit Tests
1. Evidence extraction from process-defined mappings with arbitrary facet keys.
2. Verifier behavior with process-specific output contracts.
3. Retry strategy mapping from structured metadata only.
4. Prompt assembly ensuring no domain rules appear without process injection.

### Integration Tests
1. Run at least three materially different processes (for example: research, finance, operations) with same core engine.
2. Validate success/failure behavior driven only by YAML policy.
3. Validate partial-verified and remediation queue logic from process policy.

### Regression/Guard Tests
1. Add grep-based guard tests against domain hardcoding in core files:
   - `/Users/sfw/Development/loom/src/loom/state/evidence.py`
   - `/Users/sfw/Development/loom/src/loom/engine/verification.py`
   - `/Users/sfw/Development/loom/src/loom/recovery/retry.py`
   - `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`
   - `/Users/sfw/Development/loom/src/loom/prompts/templates/verifier.yaml`
2. Golden tests for verification outcomes under v2 policies.
3. Shadow-diff thresholds before cutover.

## Acceptance Criteria
1. Core runtime contains zero process-domain assumptions.
2. Process behavior changes are achievable by YAML-only edits.
3. LLM-first verification passes dynamic/loosely-structured outputs without static shape failures.
4. Static checks do not fail tasks for semantic/data-shape interpretation issues.
5. Retry/remediation strategies are process-defined and generic.
6. Built-in processes fully migrated to v2 contract.
7. CLI/API/TUI all execute through the same policy behavior for a given process definition.
8. Process authoring docs explicitly describe contract v2 and v1->v2 migration.

## Risks and Mitigations
1. Risk: Over-flexible contracts reduce quality consistency.
   Mitigation: Strong process schema validation + golden acceptance tests per process.
2. Risk: LLM verifier variance.
   Mitigation: tiered verification, strict output protocol, optional voting, shadow diff telemetry.
3. Risk: Migration churn across built-ins.
   Mitigation: compatibility adapter + staged cutover + rollback flag.
4. Risk: Token/runtime growth.
   Mitigation: prompt compaction budgets, scoped evidence summaries, targeted remediation.

## Definition of Done
1. All hardcoded process-specific logic removed from core runtime modules.
2. All process-specific verification/remediation behavior declared in YAML contract v2.
3. LLM-first verification policy active by default.
4. Built-in process suite passes contract and integration tests under v2.
5. Legacy compatibility code removed or explicitly sunset with dated removal milestone.
