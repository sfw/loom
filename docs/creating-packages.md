# Creating Process Packages

This guide explains how to build a Loom process package — a reusable domain specialization that teaches Loom how to work in a specific problem space. Packages are the primary extension mechanism: they inject personas, phased workflows, verification rules, memory types, and optional custom tools without writing Python code in Loom's core.

**Audience:** AI agents and developers building new process packages.

## Package structure

A minimal package is a single `process.yaml` file. A full package is a directory:

```
my-process/
├── process.yaml          # Required — the process definition
├── tools/                # Optional — bundled Python tools
│   └── my_tool.py
├── README.md             # Optional — usage instructions
├── LICENSE               # Optional — license file
└── .gitignore            # Optional
```

The only required file is `process.yaml`. Everything else is optional.

## Quick start

Create `process.yaml`:

```yaml
name: my-process
version: "1.0"
schema_version: 2
description: >
  One-paragraph description of what this process does and when to use it.
author: "Your Name"
tags: [domain, keywords]

persona: |
  You are a [role description]. You [approach description].
  [Constraints and style guidance].

phase_mode: guided    # strict | guided | suggestive
risk_level: medium    # low | medium | high | critical (recommended explicit)

validity_contract:
  enabled: true
  claim_extraction:
    enabled: true
  critical_claim_types: [numeric, date, entity_fact]
  min_supported_ratio: 0.75
  max_unverified_ratio: 0.25
  max_contradicted_count: 0
  prune_mode: rewrite_uncertainty
  final_gate:
    enforce_verified_context_only: true
    synthesis_min_verification_tier: 2
    critical_claim_support_ratio: 1.0

phases:
  - id: research
    description: >
      What this phase accomplishes and how.
    depends_on: []
    model_tier: 2
    verification_tier: 1
    acceptance_criteria: >
      Concrete, measurable criteria for phase completion.
    deliverables:
      - "research-notes.md — description of this deliverable"

  - id: synthesize
    description: >
      Synthesize research into actionable output.
    depends_on: [research]
    model_tier: 3
    verification_tier: 2
    acceptance_criteria: >
      Criteria for the synthesis phase.
    deliverables:
      - "final-report.md — the main output document"
```

Test it locally:

```bash
loom -w /tmp/test-project --process /path/to/my-process/
```

Install it globally:

```bash
loom install /path/to/my-process
```

## Process contract v2 (recommended)

Set `schema_version: 2` in `process.yaml` and declare behavior in data blocks
instead of runtime code:

- `verification.policy` -- LLM-first/static-first mode, static safety checks, semantic checks, verifier output contract.
- `verification.remediation` -- strategy metadata, retry budget, critical-path behavior.
- `evidence` -- record schema, extraction mappings, and summarization controls.
- `validity_contract` -- claim extraction, prune behavior, and final synthesis gates.
- `prompt_contracts` -- process-specific executor/verifier/remediation instructions and evidence-contract activation.

Minimal v2 structure:

```yaml
schema_version: 2

verification:
  policy:
    mode: llm_first
    output_contract:
      required_fields: [passed, outcome, reason_code, severity_class, confidence, feedback, issues, metadata]
  remediation:
    default_strategy: targeted-remediation

evidence:
  record_schema:
    required_fields: [evidence_id, tool, source_url, created_at, quality]
    facets: [target, entity, subject, region, category]

validity_contract:
  enabled: true
  claim_extraction:
    enabled: true
  min_supported_ratio: 0.75
  max_unverified_ratio: 0.25
  max_contradicted_count: 0
  prune_mode: rewrite_uncertainty
  final_gate:
    enforce_verified_context_only: true
    synthesis_min_verification_tier: 2
    critical_claim_support_ratio: 1.0

prompt_contracts:
  evidence_contract:
    enabled: true
    applies_to_phases: ["*"]
```

### Migrating v1 to v2

1. Add `schema_version: 2`.
2. Keep existing `verification.rules` as-is.
3. Add `verification.policy.output_contract.required_fields` including `passed`.
4. Move remediation wording into `prompt_contracts.remediation_instructions`.
5. Add `evidence.record_schema`/`evidence.extraction` and enable `prompt_contracts.evidence_contract`.
6. Add explicit `risk_level` and a process-level `validity_contract`.
7. Add `validity_contract` overrides to synthesis phases only when you need stricter floors than process defaults.

Legacy v1 compatibility remains available during the transition window and is
scheduled for removal on June 30, 2026. New or updated packages should migrate
to `schema_version: 2` now.

## process.yaml reference

### Metadata

```yaml
name: kebab-case-name          # Required. Lowercase alphanumeric + hyphens.
version: "1.0"                 # Semver string.
schema_version: 2              # Process contract version (v2 recommended).
description: >                 # What the process does, when to use it.
  Multi-line description.
author: "Name or Org"
tags: [keyword1, keyword2]     # For discovery and filtering.
risk_level: medium             # Optional, but strongly recommended.
```

**Name rules:** Must match `^[a-z0-9][a-z0-9-]*$`. No underscores, no uppercase. This name is used in `--process my-name` and in the install directory.

### Risk level and rigor floors

`risk_level` controls default validity rigor floors before process/phase overrides
are merged:

| `risk_level` | Default floor profile |
|--------------|-----------------------|
| `low` or `medium` | `min_supported_ratio: 0.65`, `max_unverified_ratio: 0.35`, temporal gate disabled by default |
| `high` or `critical` | `min_supported_ratio: 0.8`, `max_unverified_ratio: 0.2`, temporal gate enabled with as-of + cross-claim conflict checks |

If `risk_level` is omitted, Loom may infer high-risk intent from process `name` and
`tags` tokens (for example `investment`, `financial`, `medical`, `legal`,
`compliance`). Set `risk_level` explicitly to avoid accidental policy drift.

### Dependencies

```yaml
dependencies:
  - pyyaml
  - requests>=2.28
```

Python packages installed automatically by `loom install`. Pin versions when possible — unpinned dependencies are flagged as a security risk during install review.

### Persona

```yaml
persona: |
  You are a senior [domain] specialist. You think in [mental models].
  Recommendations must be [quality bar]. Distinguish between [categories].
  Quantify impact where possible.
```

The persona shapes the model's voice, priorities, and reasoning style for the entire session. Write it in second person ("You are..."). Be specific about domain expertise, quality expectations, and how to handle uncertainty.

### Tool guidance

```yaml
tool_guidance: |
  Use CSV for tabular data. Use Markdown for narrative. Use the calculator
  for financial math. Always show your work.

tools:
  required:
    - calculator
    - spreadsheet
    - document_write
  excluded: []
```

- `tool_guidance` — free-text instructions injected into the system prompt
- `tools.required` — tools that must be available (Loom fails fast if missing)
- `tools.excluded` — tools the model should not use in this process

The required and excluded lists must not overlap.

Task-shaping note:
- Choose required tools based on the package's actual work. Research packages
  often need source-gathering and writing tools; development packages often
  need file mutation, command execution, and structured verification tools;
  operational packages may need workflow- or API-specific tools.
- If the package builds, edits, tests, or verifies software, include
  `verification_helper` in `tools.required` so Loom can route common
  build/test/report/runtime verification steps through the structured helper
  layer instead of relying only on ad hoc shell snippets.
- Keep `shell_execute` available only when the package genuinely needs
  arbitrary command execution beyond the built-in helper routes.

### Auth requirements

If your process needs authenticated APIs/MCPs, declare them explicitly so Loom
can resolve credentials before execution starts:

```yaml
auth:
  required:
    - provider: notion
      source: mcp
      resource_ref: mcp:notion
      modes: [env_passthrough]
      required_env_keys: [NOTION_TOKEN]
      mcp_server: notion
    - provider: acme_issues
      source: api
      resource_ref: api_integration:acme_issues
      modes: [api_key]
```

- `auth.required` — list of required auth resources for preflight
- `provider` — auth provider key from auth profiles (optional when
  `resource_ref` or `resource_id` is supplied)
- `source` — `api` or `mcp`
- `resource_ref` — optional stable resource selector (`mcp:<alias>`,
  `api_integration:<name>`, `tool:<name>`)
- `resource_id` — optional workspace-specific immutable resource id
- `modes` — optional allowed credential modes (`api_key`, `oauth2_pkce`,
  `oauth2_device`, `env_passthrough`, `cli_passthrough`)
- `required_env_keys` — optional env keys that must be present in profile `env`
- `mcp_server` — optional MCP alias constraint (only valid when `source: mcp`)

When exactly one profile matches a required resource, Loom auto-selects it for
the run. If multiple profiles match, Loom requires an explicit selection.

Migration note for older packages: if a tool currently reads keys directly from
`os.environ` without declaring auth requirements, add `auth_requirements` on the
tool (and/or `auth.required` in `process.yaml`) so preflight can fail early with
clear remediation instead of failing mid-run.

### Phase mode

```yaml
phase_mode: guided    # strict | guided | suggestive
```

| Mode | Behavior |
|------|----------|
| `strict` | Phases execute in dependency order. No skipping. |
| `guided` | Phases are recommended but can be reordered or skipped. |
| `suggestive` | Phases are hints. The planner is free to restructure. |

### Output coordination

```yaml
output_coordination:
  strategy: direct                 # direct | fan_in
  intermediate_root: .loom/phase-artifacts
  enforce_single_writer: true
  publish_mode: transactional      # transactional | best_effort
  conflict_policy: defer_fifo      # defer_fifo | fail_fast
  finalizer_input_policy: require_all_workers  # require_all_workers | allow_partial
```

- `strategy` controls canonical deliverable ownership behavior.
- `intermediate_root` must be a normalized workspace-relative path (no absolute or `..` segments).
- `publish_mode` controls fan-in finalizer publish behavior (`transactional` uses staged writes + commit).
- `conflict_policy` controls handling when canonical deliverable writers conflict.
- `finalizer_input_policy` controls whether fan-in finalizers require artifacts from every worker (`require_all_workers`) or can continue with partial worker coverage (`allow_partial`).

When fan-in applies to a phase with deliverables, Loom reserves the deterministic
phase-finalizer ID `<phase_id>__finalize_output`. Avoid declaring a phase with
that ID directly.

### Phases

```yaml
phases:
  - id: phase-id                    # Unique, kebab-case
    description: >                  # What this phase does
      Detailed description.
    depends_on: [other-phase]       # Phase IDs this depends on (DAG)
    model_tier: 2                   # 1 (fast), 2 (standard), 3 (powerful)
    verification_tier: 2            # 1 (light), 2 (thorough)
    is_critical_path: true          # Optional. Default false.
    is_synthesis: false             # Optional. Mark final synthesis phases.
    output_strategy: fan_in         # Optional override: direct | fan_in
    finalizer_input_policy: allow_partial  # Optional override
    acceptance_criteria: >          # Concrete completion criteria
      Measurable criteria.
    deliverables:                   # Expected output files
      - "filename.md — description"
      - "data.csv — description"
    validity_contract:              # Optional per-phase policy floor override
      min_supported_ratio: 0.8
      max_unverified_ratio: 0.2
      final_gate:
        synthesis_min_verification_tier: 2
    iteration:                      # Optional phase-level loop policy
      enabled: true
      max_attempts: 4
      strategy: targeted_remediation
      stop_on_no_improvement_attempts: 2
      max_total_runner_invocations: 6
      iteration_budget:
        max_wall_clock_seconds: 600
        max_tokens: 100000
        max_tool_calls: 20
      gates:
        - id: quality-score
          type: tool_metric
          blocking: true
          tool: humanize_writing
          metric_path: report.humanization_score
          operator: gte
          value: 80
```

**Phase fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique identifier. Kebab-case. |
| `description` | Yes | What the phase accomplishes and how. |
| `depends_on` | Yes | List of phase IDs that must complete first. `[]` for root phases. |
| `model_tier` | No | 1-3. Higher = more capable model. Default 2. |
| `verification_tier` | No | 1-2. Higher = more thorough verification. Default 1. |
| `is_critical_path` | No | If true, failure here blocks the entire process. |
| `is_synthesis` | No | If true, this is a final phase that combines prior work. |
| `output_strategy` | No | Optional phase override for output ownership mode (`direct` or `fan_in`). |
| `finalizer_input_policy` | No | Optional phase override for fan-in finalizer artifact requirements (`require_all_workers` or `allow_partial`). |
| `acceptance_criteria` | No | Concrete criteria for phase completion. |
| `deliverables` | No | List of expected output files with descriptions. |
| `validity_contract` | No | Optional phase-level claim/evidence policy override (merged with process defaults). |
| `iteration` | No | Gate-driven loop policy for repeating the phase until gates pass or budgets exhaust. |

Deliverables are enforced by exact filename at execution and verification time.
If a phase declares `financial-summary.csv`, the model must create/update that
exact path (not `tesla_financial_summary.csv` or other variants).

**Dependency rules:**
- Dependencies form a DAG (directed acyclic graph). Cycles are rejected at load time.
- Independent phases (no shared dependencies) can run in parallel.
- Deliverable names must be unique across all phases.

Mixed-workflow note:
- A single process package can contain both research-oriented and
  development-oriented phases/subtasks. For example, one package can gather
  evidence, design an approach, implement code, and then run validation.
- Today, phases are the main way to express that mixed workflow. The package
  can model different kinds of work in one DAG, but `verification.policy`
  still applies at the process level rather than being fully customizable per
  phase.
- In practice, choose the overall verification policy based on the package's
  dominant or riskiest failure mode, then use phases, acceptance criteria,
  planner examples, and semantic checks to make the mixed workflow explicit.
- If research and development portions need fundamentally different
  verification behavior, consider splitting them into separate packages or
  treating one package as the primary workflow and the other as a follow-on
  process.

**Model tier guidance:**
- **Tier 1** — Simple extraction, formatting, data manipulation
- **Tier 2** — Research, analysis, structured reasoning
- **Tier 3** — Complex synthesis, creative work, multi-source integration

### Validity contract (claim and evidence rigor)

`validity_contract` can be declared at the process level and overridden at the
phase level. Loom resolves it as:

1. Risk-based default floor
2. Process-level `validity_contract`
3. Phase-level `validity_contract` (if present)
4. Synthesis hardening (`enforce_verified_context_only=true` for synthesis phases)

Canonical structure:

```yaml
validity_contract:
  enabled: true
  claim_extraction:
    enabled: true
  critical_claim_types: [numeric, date, entity_fact]
  min_supported_ratio: 0.75
  max_unverified_ratio: 0.25
  max_contradicted_count: 0
  prune_mode: rewrite_uncertainty   # drop | rewrite_uncertainty
  require_fact_checker_for_synthesis: false
  final_gate:
    enforce_verified_context_only: true
    synthesis_min_verification_tier: 2
    critical_claim_support_ratio: 1.0
    temporal_consistency:
      enabled: false
      require_as_of_alignment: false
      enforce_cross_claim_date_conflict_check: false
      max_source_age_days: 0
      as_of: ""
```

Field guidance:

| Field | Type | Purpose |
|-------|------|---------|
| `enabled` | bool | Master switch for validity pipeline. |
| `claim_extraction.enabled` | bool | Enables claim extraction and claim lifecycle tracking. |
| `critical_claim_types` | list[str] | Claim categories that must meet stricter support floors. |
| `min_supported_ratio` | float (0..1) | Minimum supported/extracted ratio required by gate checks. |
| `max_unverified_ratio` | float (0..1) | Maximum unresolved/extracted ratio allowed. |
| `max_contradicted_count` | int (`>=0`) | Maximum contradicted claims allowed (set `0` for strict use). |
| `prune_mode` | enum | `drop` removes unresolved claims, `rewrite_uncertainty` keeps uncertainty notes while excluding unresolved claims from synthesis context. |
| `require_fact_checker_for_synthesis` | bool | Requires fact checker use before synthesis passes. |
| `final_gate.enforce_verified_context_only` | bool | Excludes unresolved claims from synthesis input. |
| `final_gate.synthesis_min_verification_tier` | int (`>=1`) | Minimum verification tier for synthesis phases. |
| `final_gate.critical_claim_support_ratio` | float (0..1) | Required support ratio for critical claim types. |
| `final_gate.temporal_consistency.*` | mapping | Temporal checks (`as_of` alignment, stale-source age, cross-claim date conflicts). |

Operational behavior:

- Claim lifecycle states emitted by the runtime are:
  `extracted`, `supported`, `contradicted`, `insufficient_evidence`, `pruned`.
- Intermediate phases are non-fatal by default when unresolved claims are
  prunable and post-prune thresholds still pass.
- Final synthesis fails if critical claim thresholds are not met or if
  unresolved/contradicted claims violate final gate limits.
- High-risk and critical processes cannot weaken core safety constraints in
  explicit overrides (for example `max_contradicted_count > 0` is rejected).
- Write-path artifacts (`write_file`, `document_write`, final synthesis output)
  are captured with provenance metadata and included in run-level validity events.

### Phase iteration loops (optional)

Use `iteration` when one pass is usually not enough and you need explicit
convergence criteria (for example: score >= 80, tests pass, no placeholders).

Runtime loop shape:
1. Execute phase.
2. Evaluate gates.
3. Retry the phase if blocking gates fail and limits remain.
4. Exit when all blocking gates pass, or limits/budgets are exhausted.

Example:

```yaml
phases:
  - id: rewrite
    description: Improve draft quality.
    deliverables:
      - "draft.md — rewritten draft"
    iteration:
      enabled: true
      max_attempts: 4
      strategy: targeted_remediation       # targeted_remediation | full_rerun
      stop_on_no_improvement_attempts: 2
      max_total_runner_invocations: 6
      max_replans_after_exhaustion: 2
      replan_on_exhaustion: true
      iteration_budget:
        max_wall_clock_seconds: 600
        max_tokens: 100000
        max_tool_calls: 20
      gates:
        - id: score
          type: tool_metric
          blocking: true
          tool: humanize_writing
          metric_path: report.humanization_score
          operator: gte
          value: 80
        - id: no-placeholders
          type: artifact_regex
          blocking: true
          target: deliverables             # auto | deliverables | changed_files | summary | output
          pattern: "\\[TBD\\]|\\[TODO\\]|\\[PLACEHOLDER\\]"
          expect_match: false
```

Iteration policy fields:

| Field | Required when `iteration.enabled=true` | Description |
|-------|----------------------------------------|-------------|
| `max_attempts` | Yes | Loop attempt cap. Must be `1..6`. |
| `strategy` | No | `targeted_remediation` or `full_rerun`. |
| `stop_on_no_improvement_attempts` | No | Stop when score-like gates do not improve for N attempts. |
| `max_total_runner_invocations` | Yes | Hard cap across loop + retry churn. Must be `>= max_attempts`. |
| `max_replans_after_exhaustion` | No | Bounded replans after loop exhaustion. |
| `replan_on_exhaustion` | No | If false, terminate instead of requesting replan. |
| `iteration_budget.max_wall_clock_seconds` | Yes | Per-loop wall-clock cap. Must be `> 0`. |
| `iteration_budget.max_tokens` | Yes | Per-loop token cap. Must be `> 0`. |
| `iteration_budget.max_tool_calls` | Yes | Per-loop tool call cap. Must be `> 0`. |
| `gates` | Yes | At least one gate is required. |

Gate types:

| Gate type | Purpose | Key fields |
|-----------|---------|------------|
| `tool_metric` | Compare structured tool output metric | `tool`, `metric_path`, `operator`, `value` |
| `artifact_regex` | Regex check over files or output text | `target`, `pattern`, `expect_match` |
| `command_exit` | Run allowlisted command and compare exit code | `command`, `operator`, `value`, `timeout_seconds` |
| `verifier_field` | Compare structured verifier fields | `field`, `operator`, `value` |

Loader-enforced MVP constraints:
1. At least one deterministic blocking gate is required (`tool_metric`, `artifact_regex`, or `command_exit`).
2. `verifier_field` gates are advisory-only in MVP and cannot be blocking.
3. Score-like gates require `stop_on_no_improvement_attempts > 0`.
4. `artifact_regex.target` must be one of `auto`, `deliverables`, `changed_files`, `summary`, or `output`.
5. `command_exit` must use argv tokens (no shell metacharacters) and an allowlisted command prefix.

`command_exit` allowlisted prefixes (MVP):
- `pytest`
- `uv run pytest`
- `python -m pytest`
- `python3 -m pytest`
- `ruff check`
- `npm test`
- `pnpm test`
- `bun test`
- `go test`
- `cargo test`
- `make test`

Runtime note:
- `command_exit` gates are disabled by default at runtime.
- Enable with `execution.enable_iteration_command_exit_gate = true` in `loom.toml`.
- Process iteration loops are also feature-flagged by `execution.enable_process_iteration_loops = true`.
- You can override `command_exit` runtime allowlists via
  `execution.iteration_command_exit_allowlisted_prefixes`.
- Global loop-replan cap is controlled by
  `execution.max_iteration_replans_after_exhaustion`.

### Verification rules

In contract v2, keep rule definitions in `verification.rules`, and add
`verification.policy`/`verification.remediation` for runtime behavior.

```yaml
verification:
  policy:
    mode: llm_first
    output_contract:
      required_fields: [passed, outcome, reason_code, severity_class, confidence, feedback, issues, metadata]
  remediation:
    default_strategy: targeted-remediation
  rules:
    - name: no-placeholders
      description: "No placeholder text in deliverables"
      check: "\\[TBD\\]|\\[TODO\\]|\\[INSERT\\]|\\[PLACEHOLDER\\]|XX%"
      severity: error
      type: regex

    - name: claims-sourced
      description: "All factual claims must cite sources"
      check: >
        Scan for factual claims. Each must have an inline citation
        mapping to the references. Flag any unsupported claims.
      severity: warning
      type: llm
```

**Rule types:**

| Type | `check` field | When it runs | Cost |
|------|---------------|--------------|------|
| `regex` | Regular expression pattern | Deterministic, runs instantly | Free |
| `llm` | Natural language prompt | Sent to a model for evaluation | Token cost |

**Severity levels:**
- `error` — Verification fails. Phase result is rejected.
- `warning` — Noted but does not block.

**Target** (optional, defaults to `output`):
- `output` — Check the phase's text output
- `deliverables` — Check the delivered files

Every process should include the `no-placeholders` regex rule. It catches the most common failure mode (model leaving `[TBD]` stubs).

### Choosing a verification shape by task type

Package verification should match the package's dominant task shape rather than
assuming every process is primarily research- or development-oriented.

Common patterns:

- Research and analysis packages:
  Use `mode: llm_first` when the core task is evaluating sources, writing
  narrative analysis, or judging domain quality. Pair it with deterministic
  placeholder/structure rules and strong `validity_contract` settings.
- Development and build packages:
  Use `mode: static_first` with deterministic checks first, semantic checks
  second, and capability-aware runtime verification when the package edits or
  validates software artifacts.
- Mixed packages:
  Keep the overall process shaped around the final deliverable. For example, a
  market-research package that generates a small dashboard can still be
  research-first, while a code-generation package that includes some discovery
  phases should still use the stronger development-oriented verification policy.
- Important current limitation:
  mixed packages are supported, but Loom does not yet support a completely
  separate `verification.policy` per phase. Think of verification shape as a
  process-level default that all subtasks inherit, with subtask/phase wording
  and capability contracts helping the runtime infer the right behavior inside
  that shared policy.
- Other operational packages:
  Choose the simplest policy that reliably catches the failure modes that
  matter in that domain. For workflow automation or data maintenance packages,
  deterministic file, schema, or API-contract checks are usually better than
  generic prose-quality rules.

Use these questions to decide:

- Is the package's main risk unsupported claims, stale evidence, or poor
  synthesis? Start with research-oriented verification.
- Is the package's main risk broken artifacts, failed commands, or runtime
  regressions? Start with development-oriented verification.
- Does the package have both? Bias toward the riskier failure mode, then use
  phases, acceptance criteria, and semantic checks to keep the other mode
  represented.

### Development-focused verification policies

Packages that produce or verify software should not rely on a purely
research-style verification shape. Prefer a build-oriented policy that:

- Uses `mode: static_first`
- Uses `static_checks.tool_success_policy: development_balanced`
- Treats verifier infrastructure failures as warnings instead of product
  failures when appropriate
- Prefers behavior checks over style checks
- Declares semantic capability contracts for browser/service/runtime checks
- Uses registered helpers (`run_test_suite`, `run_build_check`,
  `serve_static`, `http_assert`, `browser_assert`, `browser_session`,
  `render_verification_report`) instead of one-off verifier shell flows when
  possible

Recommended baseline:

```yaml
tools:
  required:
    - write_file
    - shell_execute
    - verification_helper

verification:
  policy:
    mode: static_first
    static_checks:
      tool_success_policy: development_balanced
    semantic_checks:
      - name: test-suite
        capability: command_execution
        helper: run_test_suite
      - name: build-check
        capability: command_execution
        helper: run_build_check
      - name: local-service-smoke
        capability: service_runtime
        helper: serve_static
      - name: browser-smoke
        capability: browser_runtime
        helper: browser_session
        optional: true
    output_contract:
      required_fields: [passed, outcome, reason_code, severity_class, confidence, feedback, issues, metadata]
    outcome_policy:
      treat_verifier_infra_as_warning: true
      prefer_behavior_over_style: true
      optional_capabilities: [browser_runtime]
```

Field guidance for development packages:

- `development_balanced` keeps real product failures like broken tests or build
  failures blocking, while downgrading verifier harness issues such as probe
  timeouts or missing optional browser capability.
- `semantic_checks[].capability` declares what kind of verification is being
  requested. Common values are `command_execution`, `service_runtime`,
  `browser_runtime`, and `report_rendering`.
- `semantic_checks[].helper` should name a registered verification helper. Loom
  validates helper names and capability compatibility at load time.
- `semantic_checks[].optional: true` is the right shape for checks that improve
  confidence but should not fail the package when the environment lacks that
  capability.
- `outcome_policy.optional_capabilities` is the process-level escape hatch for
  capabilities such as `browser_runtime` that may be unavailable in some
  environments.
- `prefer_behavior_over_style: true` tells Loom to bias toward behavioral
  checks like "page loads and table renders" instead of brittle source-style
  rules like "must reference `window.*` explicitly."

Browser verification guidance:

- Use `browser_session` for richer localhost browser checks when the package
  needs navigation, clicks, form filling, assertions, network capture, or
  screenshots.
- Loom prefers a Playwright-backed browser session when the optional browser
  addon is installed and falls back to a static HTTP/DOM engine otherwise.
- Treat browser checks as optional unless the package truly requires full
  browser verification to satisfy its contract.
- Users can inspect addon availability with `loom doctor` and enforce it with
  `loom doctor --require-addon browser`.
- Install the addon with `uv sync --extra browser`; install Playwright browser
  binaries with `uv run playwright install`.

Subtask design guidance for development packages:

- Keep subtasks aligned to software lifecycle boundaries such as `implement`,
  `build`, `test`, `runtime-verify`, and `report`.
- Put deterministic checks as early as possible so failures localize quickly.
- Reserve browser/runtime subtasks for concrete behavioral validation, not
  style policing.
- If a package includes both research and build subtasks, keep the
  build-oriented phases on the stricter development verification policy instead
  of inheriting a single research-shaped verifier for the entire package.

General subtask design guidance for all package types:

- Keep subtasks aligned to meaningful task boundaries in the domain rather than
  arbitrary output chunks.
- Put deterministic checks as early as possible when they can cheaply localize
  failures.
- Use semantic checks for quality dimensions that cannot be expressed as simple
  file/regex/exit-code checks.
- If the package mixes task types, reflect that in the subtasks and examples:
  separate research, transformation, implementation, validation, and synthesis
  work instead of flattening them into one generic execution phase.

### Memory types

```yaml
memory:
  extract_types:
    - type: evidence_finding
      description: "Specific fact or data point with source and credibility"
    - type: research_gap
      description: "Area where evidence is insufficient or conflicting"
  extraction_guidance: |
    Prioritize findings that answer the question. Always include source
    attribution.
```

Memory types tell the extraction model what kinds of knowledge to pull from each phase's output and store in Loom's long-term memory. Define types that are specific to your domain.

### Workspace analysis

```yaml
workspace_analysis:
  scan_for:
    - "*.csv — data files with metrics or evidence"
    - "*.json — configuration exports or API responses"
    - "*.md — existing documentation or prior reports"
  guidance: |
    Look for existing data files. Reference actual data rather than making
    assumptions. If prior reports exist, build on their findings.
```

This tells Loom what to look for in the workspace before planning. The `scan_for` patterns are displayed to the planner; the `guidance` is injected into the system prompt.

### Planner examples

```yaml
planner_examples:
  - goal: "Audit GA4 for a B2B SaaS with free trial funnel"
    subtasks:
      - id: tracking-audit
        description: "Audit GA4 event taxonomy, check enhanced measurement"
        depends_on: []
        model_tier: 2
      - id: data-quality
        description: "Assess consent mode impact, check cross-domain tracking"
        depends_on: [tracking-audit]
        model_tier: 2
      - id: funnel-analysis
        description: "Map trial signup → activation → paid conversion funnel"
        depends_on: [data-quality]
        model_tier: 3
```

Provide 2-3 concrete examples showing how the planner should decompose a goal into subtasks that follow your phases. Each example should show the full dependency DAG. These are few-shot examples injected into the planner prompt.

### Replanning

```yaml
replanning:
  triggers: |
    - Research question is too broad to answer meaningfully
    - Critical sub-question has no available evidence
    - Early findings suggest the question should be reframed
  guidance: |
    If too broad, narrow scope and restart evidence gathering. If evidence
    is unavailable, note as limitation and adjust conclusions.
```

Declares when the engine should re-plan and how to adapt. Without replanning configuration, the engine follows the original plan rigidly.

### Process test manifest

Process packages can embed runnable test contracts directly in `process.yaml`:

```yaml
tests:
  - id: smoke
    mode: deterministic         # deterministic | live
    goal: "Analyze Tesla for investment"
    timeout_seconds: 900
    requires_network: false
    requires_tools:
      - write_file
    acceptance:
      phases:
        must_include:
          - company-screening
      deliverables:
        must_exist:
          - company-overview.md
      verification:
        forbidden_patterns:
          - "deliverable_.* not found"
```

`mode: deterministic` runs with a scripted model backend for reproducible CI checks.  
`mode: live` runs against configured real providers and external tools/sources.

Run package tests:

```bash
loom process test <name-or-path>
loom process test <name-or-path> --case smoke
loom process test <name-or-path> --live
```

Validation rules:
- `id` is required and must be unique per package.
- `mode` must be `deterministic` or `live`.
- `goal` is required.
- `timeout_seconds` must be greater than zero.

## Bundled tools

Packages can include custom Python tools in a `tools/` directory. These are auto-discovered and registered when the process loads.

### Writing a tool

Create `tools/my_tool.py`:

```python
"""Description of what this tool does."""

from __future__ import annotations

from typing import Any

from loom.tools.registry import Tool, ToolContext, ToolResult


class MyDomainTool(Tool):
    """One-line summary shown in tool listings."""

    @property
    def name(self) -> str:
        return "my_domain_tool"    # Unique tool name

    @property
    def description(self) -> str:
        return (
            "What this tool does and what operations are available. "
            "Be specific — the model reads this to decide when to use it."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["calc_x", "calc_y"],
                    "description": "The operation to perform.",
                },
                "args": {
                    "type": "object",
                    "description": "Arguments for the operation.",
                },
            },
            "required": ["operation", "args"],
        }

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        operation = args.get("operation", "")
        op_args = args.get("args", {})

        try:
            if operation == "calc_x":
                result = do_calculation(op_args)
                return ToolResult.ok(
                    f"Result: {result}",
                    data={"value": result},
                )
            else:
                return ToolResult.fail(
                    f"Unknown operation: {operation}. Use: calc_x, calc_y"
                )
        except (KeyError, TypeError) as e:
            return ToolResult.fail(f"Missing or invalid argument: {e}")
        except (ValueError, ZeroDivisionError) as e:
            return ToolResult.fail(f"Calculation error: {e}")
```

### Tool requirements

1. **Extend `Tool`** from `loom.tools.registry`
2. **Implement four abstract members:** `name`, `description`, `parameters`, `execute`
3. **`parameters`** must be a valid JSON Schema object
4. **`execute`** is async and receives `args` (dict) and `ctx` (ToolContext)
5. **Return `ToolResult.ok(output)`** on success, **`ToolResult.fail(error)`** on failure
6. **File names** in `tools/` must not start with `_` (those are skipped)
7. **One class per file** is conventional but not required — all `Tool` subclasses in the module are registered
8. **Workspace-writing tools must declare `is_mutating = True`** so mutation policy is enforced preflight
9. **Workspace-writing tools must return accurate `files_changed`** (workspace-relative paths) on success
10. **If write targets use non-standard arg keys, expose `mutation_target_arg_keys`** so path policy can find them (for example `output_path`, `output_json_path`)

Mutation policy note:
- Sealed artifact protection is enforced before mutating tool execution.
- Successful mutations on tracked sealed files trigger reseal/provenance updates.
- If a writing tool omits mutating metadata or `files_changed`, runtime safety and synthesis seal checks can drift.

### Upgrading existing bundled tools

If your package already has tools that write files, upgrade them to the
current mutation contract before release:

1. Add `is_mutating` and return `True` for every workspace-writing tool.
2. Ensure every successful write returns accurate workspace-relative
   `files_changed` paths.
3. If writes target keys other than `path`, expose
   `mutation_target_arg_keys` (for example `output_path`,
   `output_json_path`).
4. Keep write targets inside `ctx.workspace` and use `_resolve_path(...)`
   for normalization/safety.
5. Add/refresh package tests so each writer confirms `is_mutating` and
   `files_changed` behavior for success paths.

Minimal before/after migration example:

```python
# Before
async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
    out = str(args.get("output_path", "")).strip()
    # ...writes file...
    return ToolResult.ok("done")

# After
@property
def is_mutating(self) -> bool:
    return True

@property
def mutation_target_arg_keys(self) -> tuple[str, ...]:
    return ("output_path",)

async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
    out = str(args.get("output_path", "")).strip()
    # ...writes file...
    return ToolResult.ok("done", files_changed=[out])
```

### Tool execution surfaces

Tools can declare where they are allowed to run:

```python
@property
def supported_execution_surfaces(self) -> tuple[str, ...]:
    return ("tui",)  # e.g., interactive-only tools
```

- Valid values are `tui`, `api`, and `cli`
- If omitted, Loom defaults to all three surfaces for backward compatibility
- Use this for tools that require interactivity or a specific runtime context

If a tool calls an authenticated API, declare that requirement in the tool:

```python
@property
def auth_requirements(self) -> list[dict[str, object]]:
    return [
        {
            "provider": "acme_issues",
            "source": "api",
            "resource_ref": "api_integration:acme_issues",
            "modes": ["api_key"],
            "required_env_keys": ["ACME_API_KEY"],
        }
    ]
```

- Use the same shape as `auth.required` in `process.yaml`
- `process.yaml` `auth.required` is still supported as a fallback
- Prefer tool-level declarations when possible so Loom can infer auth needs from
  `tools.required` automatically

For tools that require OAuth, make the declaration explicit and resolve tokens
through `ctx.auth_context` at execution time:

```python
@property
def auth_requirements(self) -> list[dict[str, object]]:
    return [
        {
            "provider": "notion",
            "source": "api",
            "resource_ref": "api_integration:notion",
            "modes": ["oauth2_pkce", "oauth2_device"],
        }
    ]

async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
    auth_ctx = getattr(ctx, "auth_context", None)
    if auth_ctx is None:
        return ToolResult.fail("Auth context is missing.")
    profile = auth_ctx.profile_for_selector("api_integration:notion")
    if profile is None:
        profile = auth_ctx.profile_for_provider("notion")
    if profile is None:
        return ToolResult.fail("No selected profile for Notion auth resource.")
    token = auth_ctx.resolve_secret_ref(profile.token_ref)
    # Use token in your API client...
    return ToolResult.ok("done")
```

Runtime UX note:
- Users do not need to run `/auth select ...` before `/run`.
- Loom resolves required auth at run start and prompts for selection only when
  multiple profiles match.
- Non-interactive clients must provide explicit `auth_profile_overrides` in task
  metadata when resolution is ambiguous.

### Tool naming and collision policy

- Bundled tool names must be globally unique across built-in tools and all loaded packages.
- If a bundled tool name collides with an existing tool, Loom logs a warning and skips loading the colliding bundled tool.
- Use clear prefixes for package tools (for example, `ga_`, `fin_`, `crm_`) to avoid conflicts.

### ToolContext

The `ctx` object provides:

| Field | Type | Description |
|-------|------|-------------|
| `workspace` | `Path \| None` | The user's workspace directory |
| `scratch_dir` | `Path \| None` | Temp directory for intermediate files |
| `subtask_id` | `str` | Current subtask identifier |
| `auth_context` | `RunAuthContext \| None` | Run-scoped selected auth profiles and secret resolver |
| `execution_surface` | `"tui" \| "api" \| "cli"` | Current run surface for this tool invocation |

Use `workspace` to resolve file paths. Use `_resolve_path(raw, workspace)` for safe path resolution that prevents directory traversal.

### ToolResult

| Constructor | When to use |
|-------------|-------------|
| `ToolResult.ok(output, data={...})` | Success. `output` is the text shown to the model. `data` is optional structured data. |
| `ToolResult.fail(error)` | Failure. `error` describes what went wrong. |
| `ToolResult.multimodal(output, blocks)` | Success with rich content (images, etc). |

For workspace-writing tools, include `files_changed` on success:

```python
@property
def is_mutating(self) -> bool:
    return True

@property
def mutation_target_arg_keys(self) -> tuple[str, ...]:
    return ("output_path",)

async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
    relpath = str(args.get("output_path", "")).strip()
    # ...write file under ctx.workspace...
    return ToolResult.ok(
        f"Wrote {relpath}",
        files_changed=[relpath],
    )
```

## Installation and distribution

### Installing locally

```bash
# From a directory
loom install /path/to/my-process

# Into a specific workspace (instead of global ~/.loom/processes/)
loom install /path/to/my-process -w /path/to/project

# Install dependencies in an isolated per-process environment
loom install /path/to/my-process --isolated-deps
```

### Installing from GitHub

```bash
# Full URL
loom install https://github.com/user/loom-process-my-domain

# Shorthand
loom install user/loom-process-my-domain
```

### Where packages live

| Location | Scope |
|----------|-------|
| `~/.loom/processes/` | User-global (default install target) |
| `./loom-processes/` | Workspace-local |
| `./.loom/processes/` | Workspace-local (alternative) |
| `src/loom/processes/builtin/` | Built-in (lowest precedence; user/workspace can override by name) |

Later entries in the search order override earlier ones. A workspace-local process with the same name as a global one takes precedence.

### Security review

When installing, Loom shows a security review:

```
Package: my-process v1.0
Author:  Your Name
Source:  /path/to/my-process
Target:  ~/.loom/processes/my-process

Dependencies (1):
  requests>=2.28

Bundled code (1 file):
  ⚠  tools/my_tool.py — executes automatically when process is loaded

Risk: MEDIUM — bundled code present
```

Risk levels:
- **LOW** — YAML-only, no dependencies, no code
- **MEDIUM** — Has dependencies or bundled code (but not both)
- **HIGH** — Has both dependencies and bundled code

Users must explicitly approve before installation proceeds.

Dependency install modes:
- Default: dependencies install into the current Python environment.
- Isolated: `--isolated-deps` installs dependencies into `<install-target>/.deps/<process-name>/`.

## Design patterns

### Choosing a phase mode

- **`strict`** — Use for regulated or high-stakes domains where steps must not be skipped (financial analysis, compliance audits)
- **`guided`** — Use for most analytical workflows where order matters but flexibility helps (research, marketing, consulting)
- **`suggestive`** — Use for exploratory or opportunistic work where the planner should adapt freely (competitive intelligence, brainstorming)

### Structuring phases

1. Start with **data gathering** phases (tier 2) that have no dependencies
2. Build **analysis** phases (tier 2-3) that depend on gathered data
3. End with a **synthesis** phase (tier 3, `is_synthesis: true`) that depends on all analysis
4. Mark phases that block everything if they fail as `is_critical_path: true`
5. Allow parallel execution by giving independent phases the same (or no) dependencies

Example DAG:

```
research ──┐
            ├── synthesize ── report
analysis ──┘
```

`research` and `analysis` run in parallel. `synthesize` waits for both.

### Building process YAMLs for reliability (detailed)

Use this build sequence when authoring new process definitions:

1. Define intent and risk first:
   - Set `risk_level` explicitly.
   - Use `high`/`critical` for finance, medical, legal, compliance, or any output
     that can cause material harm.
2. Design the phase graph around evidence flow:
   - Early phases gather and normalize sources.
   - Middle phases perform claim-bearing analysis.
   - Final synthesis phase (`is_synthesis: true`) consumes only verified context.
3. Attach clear acceptance criteria to each phase:
   - Prefer measurable criteria ("contains source-backed numeric comparison for
     each vendor") over style-only criteria.
4. Configure `validity_contract` before tuning prompts:
   - Set claim coverage thresholds (`min_supported_ratio`,
     `max_unverified_ratio`, `max_contradicted_count`).
   - Choose `prune_mode` for intermediate behavior (`rewrite_uncertainty` is
     generally safer than `drop` because it preserves explicit uncertainty notes).
   - For final phases, keep `final_gate.enforce_verified_context_only: true`.
5. Enable temporal gates where recency matters:
   - Use `final_gate.temporal_consistency.enabled: true`.
   - Add `max_source_age_days` for freshness-sensitive workflows.
   - Set `as_of` when an explicit reporting date is required.
6. Keep synthesis verification strict:
   - Set synthesis phase `verification_tier` >= 2.
   - Keep `final_gate.synthesis_min_verification_tier` >= 2.
   - Keep `max_contradicted_count: 0` for production-grade processes.
7. Add deterministic + semantic verification appropriate to the task:
   - Regex rules for structure/placeholders.
   - LLM rules for domain quality and claim-source alignment when judgment is
     the main quality bar.
   - Capability- and artifact-based checks when the package produces software,
     automations, or other executable/runtime-facing outputs.
8. Validate migration and behavior in CI:
   - Add `tests:` cases in the package.
   - Include at least one failure-path test for unsupported or contradicted claims.
   - Add failure-path tests that match the package's dominant risks. For
     example: unsupported claims for research packages, failing build/test cases
     for software packages, or schema/contract mismatches for automation/data
     packages.

Suggested policy baselines:

| Context | Starting thresholds |
|---------|---------------------|
| Low/medium risk exploratory work | `min_supported_ratio: 0.65`, `max_unverified_ratio: 0.35`, `prune_mode: rewrite_uncertainty` |
| High/critical risk deliverables | `min_supported_ratio: 0.8`, `max_unverified_ratio: 0.2`, `max_contradicted_count: 0`, temporal gate enabled |
| Final synthesis (all risk levels) | `enforce_verified_context_only: true`, `synthesis_min_verification_tier: 2`, `critical_claim_support_ratio: 1.0` for high/critical |

### Writing verification rules

Every process should include at minimum:

```yaml
verification:
  rules:
    - name: no-placeholders
      description: "No placeholder text in deliverables"
      check: "\\[TBD\\]|\\[TODO\\]|\\[INSERT\\]|\\[PLACEHOLDER\\]|XX%|N/A"
      severity: error
      type: regex
```

Add domain-specific LLM rules for quality checks the model tends to miss:
- Math consistency (funnel rates multiply correctly, budgets sum correctly)
- Specificity (recommendations reference actual products/features, not generic advice)
- Completeness (all required sections present, all sub-questions answered)

### Writing good planner examples

- Provide 2-3 examples covering different scenarios in your domain
- Each example should have a realistic `goal` (one sentence)
- Subtasks should mirror your phase IDs exactly
- Show the full dependency DAG with model tiers
- Make descriptions specific to the example scenario, not generic

### Memory type design

Define 3-6 memory types that capture the most reusable knowledge from your domain:
- What would be valuable if the user runs this process again on a different project?
- What facts or decisions would help related processes?
- Be specific enough to be actionable, general enough to be reusable

## Validation

Loom validates `process.yaml` at load time with structural, contract, and
policy-floor checks. Common errors:

| Error | Fix |
|-------|-----|
| `name must match ^[a-z0-9][a-z0-9-]*$` | Use lowercase kebab-case |
| `duplicate phase id` | Each phase needs a unique `id` |
| `dependency cycle detected` | Check `depends_on` for circular references |
| `unknown dependency` | Referenced phase ID doesn't exist |
| `duplicate deliverable` | Each deliverable filename must be unique across all phases |
| `required and excluded tools overlap` | A tool can't be in both lists |
| `invalid regex in verification rule` | Test your regex pattern |
| `iteration.max_attempts must be between 1 and 6` | Lower attempt cap |
| `iteration requires at least one deterministic blocking gate` | Add blocking `tool_metric`/`artifact_regex`/`command_exit` gate |
| `command is not allowlisted for MVP command_exit gates` | Use an allowlisted test/lint command prefix |

### Ad hoc process generation checklist (model input)

If you feed this doc to a model to generate ad hoc processes, require the model
to satisfy this checklist before returning YAML:

1. Output valid `process.yaml` only (no prose around it).
2. Use `schema_version: 2`.
3. Set explicit `risk_level` (`low|medium|high|critical`).
4. Include a process-level `validity_contract` with claim extraction enabled.
5. Provide at least one phase with unique `id` values and valid `depends_on`.
6. Mark final phase(s) with `is_synthesis: true` and keep synthesis verification tier >= 2.
7. If `iteration.enabled: true`, include all required iteration fields and at least one deterministic blocking gate.
8. Never emit blocking `verifier_field` gates.
9. For `command_exit`, use allowlisted prefixes only and tokenized argv lists.
10. Keep deliverable filenames canonical and stable across phases.
11. If you define bundled workspace-writing tools, set `is_mutating: true` behavior in code and return accurate `files_changed` for every successful write.

### Migration guide: previous definition -> current schema

Use this when upgrading older process definitions (v1 or early v2 without
validity contract fields).

Step 1: Set contract version and risk intent.

```yaml
schema_version: 2
risk_level: medium   # or high/critical for high-stakes domains
```

Step 2: Preserve your existing phase DAG and verification rules.
- Keep `phases`, `depends_on`, `deliverables`, and `verification.rules`.
- Keep existing `verification.policy` and `verification.remediation`.

Step 3: Add process-level validity contract.

```yaml
validity_contract:
  enabled: true
  claim_extraction: { enabled: true }
  critical_claim_types: [numeric, date, entity_fact]
  min_supported_ratio: 0.75
  max_unverified_ratio: 0.25
  max_contradicted_count: 0
  prune_mode: rewrite_uncertainty
  final_gate:
    enforce_verified_context_only: true
    synthesis_min_verification_tier: 2
    critical_claim_support_ratio: 1.0
    temporal_consistency:
      enabled: false
      require_as_of_alignment: false
      enforce_cross_claim_date_conflict_check: false
      max_source_age_days: 0
```

Step 4: Add phase-level overrides only where needed.
- Tighten synthesis or high-stakes phases with per-phase `validity_contract`.
- Avoid loosening below process defaults for high-risk domains.

Step 5: Verify with loader + process tests.
- Run `loom process test <name-or-path>`.
- Add at least one case where unsupported claims are pruned in intermediate
  phases and blocked in final synthesis if thresholds fail.

Test validation without running:

```bash
python -c "
from pathlib import Path
from loom.processes.schema import ProcessLoader
loader = ProcessLoader(workspace=Path('.'))
defn = loader.load('/path/to/process.yaml')
print(f'Loaded: {defn.name} ({len(defn.phases)} phases)')
"
```

## Complete example

See [`packages/google-analytics/`](../packages/google-analytics/) for a full package with:
- 6 guided phases with dependency DAG
- 4 verification rules (1 regex, 3 LLM)
- 6 domain-specific memory types
- Bundled `ga_metrics` tool with 5 operations
- Workspace analysis configuration
- 2 planner examples
- Replanning triggers and guidance

## Process Package Compatibility Checklist

Before publishing a package:

- [ ] `name` is lowercase kebab-case and descriptive
- [ ] `description` is one clear paragraph
- [ ] `schema_version` is `2`
- [ ] `risk_level` is explicitly set (`low|medium|high|critical`)
- [ ] `persona` gives the model a specific expert identity
- [ ] Phases form a valid DAG (no cycles)
- [ ] Each phase has `acceptance_criteria` and `deliverables`
- [ ] Process-level `validity_contract` is present with claim extraction enabled
- [ ] Synthesis phase(s) are marked `is_synthesis: true`
- [ ] `final_gate.enforce_verified_context_only` is true for synthesis
- [ ] `max_contradicted_count` is `0` unless you have a documented exception
- [ ] If `iteration.enabled`, loop policy includes budgets, gates, and bounded caps
- [ ] If `command_exit` gate is used, command prefix is allowlisted and runtime flags are documented
- [ ] `no-placeholders` regex rule is included in verification
- [ ] At least one domain-specific verification rule (LLM or regex)
- [ ] 2-3 planner examples with realistic goals
- [ ] `replanning` section describes when to adapt
- [ ] Bundled tools (if any) handle errors gracefully and return `ToolResult.fail`
- [ ] Bundled workspace-writing tools set `is_mutating = True` and return accurate `files_changed`
- [ ] Bundled tool names are unique (no collisions with built-ins or other packages)
- [ ] Dependencies are version-pinned
- [ ] `loom install /path/to/package` succeeds
- [ ] `loom install /path/to/package --isolated-deps` succeeds (if dependencies are declared)
- [ ] `loom -w /tmp/test --process my-process` loads without errors
- [ ] README.md explains usage, deliverables, and input data

## Troubleshooting

- `Process '<name>' requires missing tool(s): ...`:
Install or enable the required tools, or remove them from `tools.required`.

- `Bundled tool '<name>' ... conflicts with existing tool class ...; skipping bundled tool`:
Rename the bundled tool to a globally unique name and reinstall the package.

- `Failed to create isolated dependency environment`:
Ensure the selected Python executable can create virtual environments (`python -m venv`).

- `Failed to install isolated dependencies`:
Check package names/versions and network/index access in the isolated environment.
