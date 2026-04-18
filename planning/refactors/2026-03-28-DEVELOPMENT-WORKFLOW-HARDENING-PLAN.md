# Development Workflow Hardening Plan (2026-03-28)

## 1) Goal
Harden Loom for software-development and software-verification tasks without degrading research, writing, or hybrid workflows.

This plan focuses on:
1. giving `intent=build` and coding-heavy runs a first-class verification lane
2. separating product failures from verifier/harness failures
3. replacing brittle, style-based gates with capability- and behavior-based checks
4. making development verification observable, reproducible, and safe to roll out incrementally

This is a planning artifact only. No runtime changes are included here.

## 2) Why This Is Needed
Recent incident analysis exposed a familiar failure pattern for development workflows:
1. the generated microsite contained a real verifier mismatch
2. the verifier stack also introduced independent infrastructure failures
3. the orchestrator collapsed both into a terminal `hard_invariant_failed`
4. a human-readable report claimed success while the machine-readable results file recorded failure

Observed incident artifacts:
1. `runtime-validation-results.json` recorded `passed: 15, failed: 1` with the failed check `main.js references window.* data globals`
2. `ui-integration-validation-report.md` claimed `16/16 tests passed (100%)`
3. the run event trail showed:
   - a shell-based local HTTP server command timing out after 120 seconds
   - a `claude_code` browser-verification step timing out after 180 seconds
   - the subtask being marked `hard_invariant_failed`

The important takeaway is not that Loom "cannot code." The system already has meaningful coding and verification infrastructure. The issue is that the current runtime is still too research-shaped for software work:
1. verifier infrastructure failures are not cleanly separated from product failures
2. verification contracts are too generic for build tasks
3. optional UI/browser checks are treated too much like mandatory gates
4. machine-readable and human-readable verification outputs can drift
5. some deterministic checks overfit to style rather than behavior

## 3) What Already Exists
Loom already has several strong building blocks that this plan should reuse instead of replacing.

### A) Intent-aware ad hoc planning
`build` is already a first-class ad hoc intent and requires an implementation phase and a test/verify phase.

Relevant code:
1. `src/loom/tui/app/process_runs/adhoc.py`
2. `adhoc_intent_progression()`
3. intent-shape enforcement in the synthesized ad hoc process flow

### B) Structured verification policy contracts
Process definitions already carry a structured `verification_policy` with:
1. `mode`
2. `static_checks`
3. `semantic_checks`
4. `output_contract`
5. `outcome_policy`

Relevant code:
1. `src/loom/processes/schema.py`
2. `VerificationPolicyContract`

### C) Verification profiles already exist
The verification layer already understands:
1. `research`
2. `coding`
3. `data_ops`
4. `hybrid`

Relevant code:
1. `src/loom/engine/verification/policy.py`
2. `src/loom/engine/orchestrator/profile.py`

### D) Resilience policy routing already exists
The policy engine already supports rollout modes:
1. `enforce`
2. `shadow`
3. `off`

It already routes some reasons differently by profile, including coding-aware synthesis behavior.

Relevant code:
1. `src/loom/engine/verification/policy.py`
2. `src/loom/engine/orchestrator/dispatch.py`
3. `src/loom/engine/orchestrator/validity.py`

### E) Tier-1 verifier already has scoped tool-failure policy
Tier 1 already supports:
1. `all_tools_hard`
2. `safety_integrity_only`

Relevant code:
1. `src/loom/engine/verification/tier1.py`

This means the repo already has the seams needed to harden development flows without a broad rewrite.

## 4) Current Gaps
The core issue is not missing infrastructure. It is that the existing components are not yet composed into a reliable development lane.

### Gap 1: Ad hoc verification defaults are still generic
`adhoc_default_verification_policy()` is not intent-aware today. Ad hoc `build` tasks still inherit a generic policy shape that was designed to avoid brittleness broadly, not to provide strong software verification.

Consequence:
1. development tasks do not automatically receive a deterministic-first verification bundle
2. coding verification expectations are inferred later and inconsistently

### Gap 2: Product failures and verifier failures are mixed together
The current system can route all of the following into one terminal failure path:
1. a real failed product assertion
2. a verifier parse problem
3. a browser harness timeout
4. a shell orchestration mistake
5. a report-generation contradiction

Consequence:
1. a build run can fail "for verification" even when the build artifact is probably usable
2. users cannot reliably tell whether Loom failed to build software or failed to verify it

### Gap 3: `hard_invariant` is still too broad for dev workflows
`hard_invariant` is appropriate for:
1. safety violations
2. integrity/policy violations
3. non-negotiable contract failures

It is not appropriate for:
1. optional browser-verification tooling being unavailable
2. a verifier harness command timing out
3. machine/human report drift
4. style-based source-code heuristics that do not imply a broken runtime

Consequence:
1. development runs are blocked too aggressively
2. remediation logic is invoked in the wrong situations

### Gap 4: Verification artifacts do not have a single source of truth
The incident produced contradictory outputs:
1. machine-readable JSON said failure
2. markdown said success

Consequence:
1. user trust drops sharply
2. orchestrator and UI can disagree
3. false confidence artifacts accumulate inside the workspace

### Gap 5: Development verification uses fragile shell composition
The incident verifier launched a static server using a shell background process:
1. `python3 -m http.server 8080 &`
2. `sleep 2`
3. `curl ...`

That pattern is easy to hang if process lifetime is not managed.

Consequence:
1. verifier commands can time out even when the page would have loaded
2. harness bugs surface as task failures

### Gap 6: Some coding checks are style-heuristic checks rather than behavior checks
Example incident check:
1. `main.js references window.* data globals`

That can be a useful convention check, but it is not by itself a strong runtime-behavior guarantee.

Consequence:
1. development runs can fail for implementation style rather than user-visible correctness
2. the verifier creates unnecessary friction when the delivered artifact is actually functional

### Gap 7: Optional environment-dependent checks are not degraded cleanly
Browser/UI verification depends on runtime capabilities:
1. local server creation
2. port binding
3. headless browser availability
4. screenshot support

Consequence:
1. environment limitations can look like product failures
2. verification results are not portable across execution contexts

## 5) Design Principles
The hardening strategy should follow these rules.

### P1: Scope hardening to development lanes first
Default behavior for research and writing should remain stable unless the user explicitly opts into broader policy changes.

### P2: Deterministic-first for build tasks
For `intent=build` and high-confidence `coding` profiles:
1. run deterministic artifact and command checks first
2. run semantic/model-assisted verification second
3. only escalate to browser/UI checks when the deliverable actually needs them

### P3: Behavior beats style
Prefer:
1. "page loads without console errors"
2. "tests pass"
3. "table has rows"
4. "artifact exists and is non-empty"

Over:
1. "code references `window.foo` rather than `foo`"
2. other source-pattern checks that do not map directly to a user-visible guarantee

### P4: Verifier infra failures are not product failures
Timeouts, unavailable browsers, port-binding failures, parser failures, and report-generation contradictions should become `infra` or `inconclusive` unless the task explicitly required that exact capability.

### P5: Machine-readable output is canonical
The orchestrator, UI, remediation logic, and human-readable reports should all derive from one structured verification result.

### P6: Roll out through shadow mode and feature flags
The system already has resilience policy modes; development hardening should reuse them for a safe rollout path.

## 6) Target End State
For development tasks, Loom should behave like this:
1. ad hoc `build` tasks automatically receive a development verification policy
2. coding-heavy tasks are resolved into the `coding` profile with explicit confidence and fallback behavior
3. deterministic software checks run first and produce structured results
4. optional environment-dependent checks degrade to warnings when capability is absent
5. verifier harness failures are surfaced distinctly from product failures
6. human-readable reports are rendered from structured result files and cannot contradict them
7. only true integrity/safety/contract violations become terminal hard failures

For non-development tasks, default behavior should remain effectively unchanged unless explicitly configured otherwise.

## 7) Proposed Architecture

### A) Introduce an intent-aware development verification profile bundle
Change ad hoc verification defaults from:
1. one generic policy for all intents

To:
1. `research` defaults
2. `writing` defaults
3. `build` defaults

`build` should bias toward:
1. `static_first`
2. stronger deterministic `output_contract`
3. explicit `semantic_checks` for software verification
4. an `outcome_policy` that distinguishes:
   - required checks
   - optional checks
   - infra downgrades

This should be implemented as an additive extension to `verification_policy`, not as a second parallel policy system.

### B) Add a development verification capability model
Development verification should understand a small number of capability classes:
1. `artifact_static`
   - syntax, parse, file presence, schema, non-empty outputs
2. `command_execution`
   - test command, lint, typecheck, build command
3. `service_runtime`
   - local static server or app startup
4. `browser_runtime`
   - headless browser load, DOM assertions, console/network capture
5. `report_rendering`
   - human report generation from structured results

Each check should declare:
1. whether it is `required` or `optional`
2. what capability it depends on
3. what failure class it produces on failure:
   - `product`
   - `infra`
   - `inconclusive`

### C) Narrow `hard_invariant` to true non-negotiables
For development tasks, `hard_invariant` should be limited to:
1. safety/policy violations
2. artifact path and mutation-policy violations
3. required deliverable missing after bounded remediation
4. deterministic contract failures explicitly declared as blocking
5. confirmed contradictions between produced artifact and required contract

It should not be used for:
1. unavailable browser automation
2. verifier shell timeout
3. report-generation mismatch
4. style-only code heuristics

### D) Make structured verification output canonical
Development verification should emit one canonical structured result, for example:
1. overall outcome
2. check list
3. per-check capability
4. per-check required/optional classification
5. failure class
6. evidence references
7. summarized reason codes

Human-readable markdown reports should be pure renderings of this file.

### E) Replace raw-shell verifier harnesses with structured helpers
Instead of freeform shell snippets, development verification should gradually move to helpers such as:
1. `serve_static`
2. `run_build_check`
3. `run_test_suite`
4. `browser_assert`
5. `render_verification_report`

These helpers can still be backed by existing tools internally, but they should have:
1. bounded lifecycles
2. deterministic cleanup
3. structured results
4. explicit capability classification

## 8) Detailed Workstreams

### W1: Intent-Aware Ad Hoc Verification Defaults
Objective:
1. make `build` receive a development verification policy by default

Changes:
1. update `adhoc_default_verification_policy()` to accept intent and possibly risk
2. keep current defaults for `research` and `writing`
3. add `build` defaults:
   - `mode=static_first`
   - `static_checks.tool_success_policy=safety_integrity_only` or a new dev-specific policy
   - development-specific `output_contract`
   - development-specific `outcome_policy`
4. ensure synthesized ad hoc specs preserve this shape through normalization and process-definition creation

Key files:
1. `src/loom/tui/app/process_runs/adhoc.py`
2. `tests/tui/test_process_slash_commands.py`

Success criteria:
1. ad hoc `build` tasks no longer inherit the same verifier defaults as research tasks
2. existing research/writing tests stay green

### W2: Development Check Taxonomy and Failure Classification
Objective:
1. add first-class distinction between product failure and verifier/harness failure

Changes:
1. extend verification metadata conventions for development checks
2. add a stable failure taxonomy:
   - `dev_contract_failed`
   - `dev_test_failed`
   - `dev_build_failed`
   - `dev_browser_check_failed`
   - `dev_verifier_timeout`
   - `dev_verifier_capability_unavailable`
   - `dev_report_contract_violation`
3. map infra/harness failures to `severity_class=infra` or `inconclusive`
4. reserve `hard_invariant` for true policy/integrity issues

Key files:
1. `src/loom/engine/verification/types.py`
2. `src/loom/engine/verification/policy.py`
3. `src/loom/engine/verification/tier1.py`
4. `src/loom/recovery/retry.py`

Success criteria:
1. a browser or harness timeout no longer automatically looks like a broken software deliverable
2. retry routing becomes more targeted for dev failures

### W3: Deterministic-First Development Verification
Objective:
1. strengthen software verification without requiring brittle LLM-first behavior

Changes:
1. define a baseline deterministic development check bundle:
   - output exists
   - syntax parses
   - required scripts/tables/routes/components exist when contract says so
   - declared test command result
   - declared build/lint/typecheck results
2. treat style-based source heuristics as advisory unless the process explicitly marks them required
3. prefer contract- and behavior-based assertions over raw source-pattern assertions

Key files:
1. `src/loom/engine/verification/tier1.py`
2. `src/loom/processes/schema.py`
3. `src/loom/engine/verification/policy.py`

Success criteria:
1. deterministic checks carry most of the signal for build runs
2. style-only rules cannot terminate a run unless explicitly configured to do so

### W4: Optional Browser/UI Verification Lane
Objective:
1. support UI verification without making environment limitations look like product failure

Changes:
1. define browser/UI verification as an optional capability lane unless the task or contract explicitly marks it required
2. add a capability preflight:
   - can a local service be started?
   - is a browser runner available?
   - can screenshots be captured?
3. downgrade missing capability to `infra` or skipped-warning unless the contract requires it
4. ensure browser results are structured and report exact skipped/failed reasons

Key files:
1. `src/loom/engine/verification/policy.py`
2. `src/loom/engine/orchestrator/dispatch.py`
3. future helper/tool module for dev verification

Success criteria:
1. a missing browser environment produces a clear "verification skipped/degraded" result
2. required browser verification can still hard-fail when explicitly requested

### W5: Structured Harness Helpers for Development Verification
Objective:
1. eliminate freeform shell verifier mistakes as a major failure source

Changes:
1. introduce helper abstractions for:
   - static file serving with cleanup
   - command execution with structured exit/timeout metadata
   - browser assertions with structured evidence
2. centralize process lifecycle and cleanup
3. emit structured verifier events with:
   - capability
   - optional/required
   - timeout
   - cleanup success/failure

Key files:
1. new helper module under `src/loom/engine/verification/` or `src/loom/tools/`
2. orchestrator integration points
3. tool-level tests

Success criteria:
1. no background-process shell fragments remain in canonical dev verification flows
2. timeout events identify the harness component precisely

### W6: Canonical Verification Result and Derived Reports
Objective:
1. prevent contradictory success/failure artifacts

Changes:
1. define one machine-readable verification result contract as canonical
2. render markdown reports from that structured result only
3. add consistency enforcement:
   - if rendered report summary disagrees with canonical result, emit `infra_verifier_error`
4. surface the canonical result path in task telemetry/UI

Key files:
1. report-generation path in the relevant verification tooling
2. `src/loom/engine/orchestrator/dispatch.py`
3. tests for canonical-result/report agreement

Success criteria:
1. markdown and JSON verification outputs cannot disagree silently
2. the UI always reflects the canonical result

### W7: Policy Matrix Tightening for Development Profiles
Objective:
1. use the existing resilience policy engine more effectively for coding workflows

Changes:
1. update `resolve_failure_action()` to explicitly recognize development verifier infra reasons
2. ensure `coding` and high-confidence `build` runs:
   - retry infra verifier failures
   - downgrade optional browser/UI misses to warnings
   - block only on true contract/integrity failures
3. preserve research strictness for research-specific coverage/evidence failures

Key files:
1. `src/loom/engine/verification/policy.py`
2. `src/loom/recovery/retry.py`
3. `src/loom/engine/orchestrator/dispatch.py`

Success criteria:
1. development hardening is profile-scoped
2. research behavior remains materially stable

### W8: Telemetry and Auditability
Objective:
1. make development verification regressions measurable

Changes:
1. add counters for:
   - verifier-caused terminal failures
   - product-caused terminal failures
   - optional-check skips
   - browser-capability misses
   - report/result consistency failures
2. add run summary fields that distinguish:
   - product health
   - verifier health
3. expose shadow-mode diffs for dev-policy routing changes

Key files:
1. `src/loom/engine/orchestrator/telemetry.py`
2. task event emitters
3. tests covering new telemetry payload fields

Success criteria:
1. the team can track whether hardening improves dev reliability without inflating bad passes

## 9) Proposed PR Sequence

### PR-D1: Intent-Aware Ad Hoc Verification Defaults
1. make `adhoc_default_verification_policy()` intent-aware
2. add build-policy defaults
3. add ad hoc normalization tests

### PR-D2: Development Failure Taxonomy
1. add structured reason-code and severity mapping for dev verifier failures
2. update retry classification and policy tests

### PR-D3: Canonical Verification Result Contract
1. define canonical machine-readable dev verification result
2. render markdown summaries from the canonical result
3. add mismatch detection tests

### PR-D4: Deterministic Development Check Bundle
1. add contract-driven deterministic dev checks
2. downgrade style-only heuristics to advisory by default
3. extend tier-1 verification tests

### PR-D5: Structured Dev Harness Helpers
1. add safe static-server/test/browser helper abstractions
2. replace raw shell-based verifier flows in canonical development paths
3. add cleanup and timeout tests

### PR-D6: Browser/UI Optionality and Capability Gating
1. add capability preflight
2. add required-vs-optional browser check semantics
3. wire policy decisions for `coding`/`build` flows

### PR-D7: Telemetry and Rollout Controls
1. emit dev verifier health metrics
2. shadow-compare old vs new routing
3. add flags/config for staged enablement

### PR-D8: Cutover and Cleanup
1. enable hardened defaults for build intent
2. remove obsolete shell-only verifier paths where replaced
3. keep research/writing defaults unchanged unless explicitly opted in

## 10) Rollout Strategy
This work should be rolled out narrowly and measurably.

### Phase A: Shadow only
1. keep legacy routing authoritative
2. compute new dev-policy routing in parallel
3. emit telemetry diffs

### Phase B: Build-intent opt-in
1. enable new defaults only for ad hoc `build` runs behind a flag
2. keep research/writing on legacy behavior

### Phase C: Coding-profile default
1. enable for high-confidence `coding` profile runs
2. retain fallback to `hybrid` when profile confidence is low

### Phase D: Broader package adoption
1. allow process packages to opt into the dev profile bundle explicitly
2. keep package-local override support

Recommended flags/config:
1. `verification.dev_profile_enabled`
2. `verification.dev_profile_shadow_only`
3. `verification.dev_browser_checks_default_optional`
4. `verification.dev_report_canonical_result_required`
5. `verification.dev_structured_harness_enabled`

## 11) File Touch Plan
Primary files expected to change:
1. `src/loom/tui/app/process_runs/adhoc.py`
2. `src/loom/processes/schema.py`
3. `src/loom/engine/verification/types.py`
4. `src/loom/engine/verification/policy.py`
5. `src/loom/engine/verification/tier1.py`
6. `src/loom/engine/orchestrator/dispatch.py`
7. `src/loom/recovery/retry.py`
8. `src/loom/engine/orchestrator/telemetry.py`
9. new development-verification helper module(s)
10. report-generation path for structured verification output

Primary tests expected to change/add:
1. `tests/tui/test_process_slash_commands.py`
2. `tests/test_verification.py`
3. `tests/test_verification_policy.py`
4. `tests/test_retry.py`
5. `tests/orchestrator/test_validity_policy.py`
6. `tests/orchestrator/test_verification_profile.py`
7. new tests for structured harness lifecycle and canonical report rendering

## 12) Acceptance Criteria

### Functional
1. ad hoc `build` runs receive an intent-aware verification policy by default
2. optional browser/UI verification can be skipped or downgraded cleanly when capability is absent
3. verifier infra failures no longer surface as product failures by default
4. markdown verification reports cannot contradict canonical machine-readable results
5. style-only code heuristics are advisory unless explicitly made contractual

### Reliability
1. verifier-caused terminal failure rate drops for build/coding runs
2. hard-invariant terminal failures become concentrated in true safety/integrity cases
3. timeout-related dev verification failures become diagnosable by capability and helper

### Safety
1. research and writing defaults remain stable
2. non-dev tasks do not inherit stricter dev verification lanes accidentally
3. mutation/integrity/safety guards remain hard

## 13) Risks and Tradeoffs

### Risk 1: Over-downgrading real product failures
Mitigation:
1. only downgrade failures that are clearly verifier/harness/capability related
2. keep required contract failures blocking
3. use shadow mode to audit false-pass risk before cutover

### Risk 2: Adding too much dev-specific complexity
Mitigation:
1. reuse existing `build` intent and `coding` profile infrastructure
2. extend `verification_policy` rather than inventing a second policy system
3. keep helper APIs narrow and typed

### Risk 3: Research workflows accidentally drift
Mitigation:
1. make the first rollout `build`-only
2. preserve existing defaults for research/writing
3. add regression tests for current research behavior

### Risk 4: Browser verification remains flaky even with helpers
Mitigation:
1. make browser verification optional by default
2. require explicit contracts for mandatory browser validation
3. report capability misses separately from product failures

## 14) Recommended First Cut
The smallest high-leverage implementation slice is:
1. make ad hoc verification defaults intent-aware for `build`
2. add a development failure taxonomy in verification policy
3. make machine-readable verification results canonical over markdown summaries
4. downgrade optional browser/harness failures to `infra` for build/coding runs

This slice should materially reduce the class of false terminal failures seen in the incident without forcing a full verifier-tooling rewrite on day one.

## 15) Summary
Loom does not need a separate "software mode" bolted onto the side. It already has the primitives:
1. build intent
2. coding profile
3. policy modes
4. deterministic verification
5. profile-aware synthesis behavior

The missing piece is composition. The system needs a development-specific policy lane that:
1. activates earlier
2. classifies failures more carefully
3. uses one canonical verification result
4. treats browser/harness checks as capabilities, not assumptions

That is the path to making Loom significantly better at development work without destabilizing the rest of the system.
