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
```

**Name rules:** Must match `^[a-z0-9][a-z0-9-]*$`. No underscores, no uppercase. This name is used in `--process my-name` and in the install directory.

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

### Phase mode

```yaml
phase_mode: guided    # strict | guided | suggestive
```

| Mode | Behavior |
|------|----------|
| `strict` | Phases execute in dependency order. No skipping. |
| `guided` | Phases are recommended but can be reordered or skipped. |
| `suggestive` | Phases are hints. The planner is free to restructure. |

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
    acceptance_criteria: >          # Concrete completion criteria
      Measurable criteria.
    deliverables:                   # Expected output files
      - "filename.md — description"
      - "data.csv — description"
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
| `acceptance_criteria` | No | Concrete criteria for phase completion. |
| `deliverables` | No | List of expected output files with descriptions. |

Deliverables are enforced by exact filename at execution and verification time.
If a phase declares `financial-summary.csv`, the model must create/update that
exact path (not `tesla_financial_summary.csv` or other variants).

**Dependency rules:**
- Dependencies form a DAG (directed acyclic graph). Cycles are rejected at load time.
- Independent phases (no shared dependencies) can run in parallel.
- Deliverable names must be unique across all phases.

**Model tier guidance:**
- **Tier 1** — Simple extraction, formatting, data manipulation
- **Tier 2** — Research, analysis, structured reasoning
- **Tier 3** — Complex synthesis, creative work, multi-source integration

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

Use `workspace` to resolve file paths. Use `_resolve_path(raw, workspace)` for safe path resolution that prevents directory traversal.

### ToolResult

| Constructor | When to use |
|-------------|-------------|
| `ToolResult.ok(output, data={...})` | Success. `output` is the text shown to the model. `data` is optional structured data. |
| `ToolResult.fail(error)` | Failure. `error` describes what went wrong. |
| `ToolResult.multimodal(output, blocks)` | Success with rich content (images, etc). |

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
| `src/loom/processes/builtin/` | Built-in (cannot be overwritten) |

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

Loom validates `process.yaml` at load time with 22 rules. Common errors:

| Error | Fix |
|-------|-----|
| `name must match ^[a-z0-9][a-z0-9-]*$` | Use lowercase kebab-case |
| `duplicate phase id` | Each phase needs a unique `id` |
| `dependency cycle detected` | Check `depends_on` for circular references |
| `unknown dependency` | Referenced phase ID doesn't exist |
| `duplicate deliverable` | Each deliverable filename must be unique across all phases |
| `required and excluded tools overlap` | A tool can't be in both lists |
| `invalid regex in verification rule` | Test your regex pattern |

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
- [ ] `persona` gives the model a specific expert identity
- [ ] Phases form a valid DAG (no cycles)
- [ ] Each phase has `acceptance_criteria` and `deliverables`
- [ ] `no-placeholders` regex rule is included in verification
- [ ] At least one domain-specific verification rule (LLM or regex)
- [ ] 2-3 planner examples with realistic goals
- [ ] `replanning` section describes when to adapt
- [ ] Bundled tools (if any) handle errors gracefully and return `ToolResult.fail`
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
