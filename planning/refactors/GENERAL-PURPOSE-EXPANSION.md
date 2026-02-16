# Loom General-Purpose Expansion Plan

**Status: IMPLEMENTED** — R4 (Process Definition Plugin Architecture) is fully built and shipped. See `src/loom/processes/` for the implementation and `PLAN.md` Phase 7 for the summary.

**Goal:** Transform Loom from a code-only orchestration engine into a general-purpose AI work engine capable of handling complex white-collar tasks: marketing strategy, investment analysis, financial modeling, consulting engagements, policy analysis, and research workflows.

**Guiding principle:** Loom's orchestration layer (plan → execute → verify → learn) is domain-agnostic. The code-specific assumptions live in the _tools_, _prompts_, _verification_, and _workspace model_. This plan abstracts those layers behind domain-aware interfaces without breaking the existing code workflow.

---

## Part 1: Use-Case Research Summary

### 1.1 Marketing Strategy & Market Research

**Real-world workflow (7 phases):**

| Phase | Depends On | Deliverables | Verification |
|-------|-----------|-------------|-------------|
| 1. Market Sizing (TAM/SAM/SOM) | — | Sizing spreadsheet, industry report | Top-down vs. bottom-up triangulation; ±15% agreement |
| 2. Competitive Analysis | — | Competitive matrix, SWOT, battlecards | Cross-reference traffic data (SimilarWeb) against SEO data (SEMrush) |
| 3. Customer Segmentation | Phase 1 (for segment sizing) | Persona documents, segmentation model, journey maps | Segment sizes validated against CRM data; A/B test messaging per segment |
| 4. Positioning & Messaging | Phases 1, 2, 3 | Positioning statement, messaging framework, taglines | Audience testing; differentiation vs. competitive matrix |
| 5. Channel Strategy | Phase 4 | Channel plan with budget allocation, media plan | ROI projections sanity-checked against industry benchmarks |
| 6. Campaign Planning | Phase 5 | Creative briefs, ad copy, email sequences, timelines | Brand guideline compliance; audience/channel match |
| 7. Measurement & Optimization | Phase 6 | KPI dashboards, attribution analysis, performance reports | MTA + MMM + incrementality testing triangulation |

**Key insight from user:** "Good marketing strategy also requires a dependency graph as early phases are inputs into latter phases." This is exactly the fan-out/fan-in pattern. Phases 1-2 are parallel research. Phase 3 depends on Phase 1. Phase 4 is the synthesis bottleneck requiring all three research phases. Then 5→6→7 is a sequential refinement chain with Phase 7 feeding back to all prior phases.

**Data sources:** SimilarWeb, SEMrush, Ahrefs, Google Trends, Statista, IBISWorld, Google Analytics, social listening (Brandwatch, Brand24), CRM data (Salesforce, HubSpot), survey tools (Qualtrics, SurveyMonkey).

### 1.2 Investment & Financial Analysis

**Real-world workflow (8 phases):**

| Phase | Depends On | Deliverables | Verification |
|-------|-----------|-------------|-------------|
| 1. Screening | — | Watchlist, screening output | Filter criteria consistency check |
| 2. Business & Industry Analysis | Phase 1 | Business overview, Porter's Five Forces, management assessment | Multiple-source cross-check |
| 3. Financial Statement Analysis | Phase 1 | Historical financials spreadsheet, ratio analysis, quality-of-earnings | Ratios vs. industry benchmarks; check against prior guidance |
| 4. Financial Modeling | Phases 2, 3 | 3-statement model, revenue build-up | Formula audit; circular reference check; balances balance |
| 5. Valuation | Phase 4 | DCF model, comps table, valuation summary | DCF vs. comps vs. precedent transactions agreement |
| 6. Risk & Scenarios | Phase 5 | Sensitivity tables, bull/base/bear cases | Scenario range covers historical actuals; terminal value < 85% of DCF |
| 7. Investment Memo | Phases 1-6 | 10-25 page memo, IC presentation | Red team critique; MECE completeness |
| 8. Portfolio Construction | Phase 7 | Position sizing, allocation model | Correlation check; risk budget adherence |

**Key insight:** This workflow is heavily sequential with Phases 2-3 as the only parallel segment. Phase 7 (Investment Memo) is the synthesis bottleneck. Phase 8 creates the monitoring feedback loop.

**Data sources:** SEC EDGAR, financial data APIs (Alpha Vantage, Yahoo Finance, Financial Modeling Prep), earnings transcripts (Koyfin, Seeking Alpha), industry reports, alternative data (web traffic, app downloads).

### 1.3 Cross-Domain Patterns

Seven universal patterns emerge across all white-collar knowledge work:

1. **Fan-out/Fan-in:** Parallel research workstreams converging into synthesis
2. **Sequential refinement chain:** Each phase refines and depends on prior output
3. **Verification/triangulation gates:** Cross-check using independent methods
4. **Feedback loops:** Measurement phase feeds back to research phase
5. **Synthesis bottleneck:** All workstreams converge into one narrative deliverable
6. **Hypothesis-driven oscillation:** Top-down hypothesis tested against bottom-up data
7. **Gated decision points:** Go/no-go gates that can terminate or redirect the workflow

**Common deliverable types across all domains:**
- Spreadsheet models (financial models, market sizing, competitive matrices, budgets)
- Narrative documents (memos, briefs, reports, policies)
- Presentation decks (IC presentations, strategy decks, steerco presentations)
- Dashboards (KPI tracking, portfolio analytics, compliance monitoring)
- Structured data (watchlists, checklists, risk registers, gap matrices)

---

## Part 2: Architecture Audit Summary

### Code-specific blockers identified in current Loom:

| Component | File(s) | Blocker | Priority |
|-----------|---------|---------|----------|
| **Prompt templates** | `prompts/templates/*.yaml` | Every template references "file tools", "shell", "code" | P0 |
| **Orchestrator planning** | `orchestrator.py:451-466` | Hard-coded `analyze_directory()` for code structure | P0 |
| **Tool system** | `tools/*.py`, `registry.py` | `files_changed`, filesystem paths, code-only tools | P0 |
| **Verification tier 1** | `verification.py:100-139` | Syntax checks only for .py/.json/.yaml | P1 |
| **Workspace model** | `workspace.py` | File snapshots, unified diffs, filesystem-only | P1 |
| **Config** | `config.py` | `workspace_path` filesystem-only, no domain field | P1 |
| **Memory extraction** | `runner.py:256-335` | `tool_result`/`artifact` types; code-centric prompt | P2 |
| **Task state** | `task_state.py` | `WorkspaceChanges` counts files only | P3 |
| **Learning** | `learning/manager.py` | Tool-based patterns only | P3 |
| **Database schema** | `schema.sql` | No domain field, no stakeholders/budget/deadline | P3 |

**Key finding:** Loom's orchestration layer (plan → execute → verify → learn cycle, dependency scheduling, parallel dispatch, retry escalation, approval gates) is already domain-agnostic. The expansion is primarily a _tool_, _prompt_, and _verification_ problem.

---

## Part 3: Implementation Plan

### Phase 0: Domain Abstraction Layer (Foundation)

**Goal:** Introduce domain awareness without breaking existing code workflow.

#### 0.1 Domain Registry

Create `src/loom/domains/` package with a domain registry pattern:

```
src/loom/domains/
    __init__.py          # DomainRegistry, BaseDomain
    code.py              # CodeDomain (current behavior)
    marketing.py         # MarketingDomain
    finance.py           # FinanceDomain
    consulting.py        # ConsultingDomain
    research.py          # ResearchDomain
```

Each domain defines:
- **Name and description** — for model context
- **Available tools** — which tools are relevant (code domain gets shell/git; marketing gets web_search/spreadsheet)
- **Prompt template set** — domain-specific planning/execution/verification prompts
- **Verification rules** — what deterministic checks apply
- **Memory entry types** — what kinds of knowledge to extract
- **Artifact types** — what deliverables this domain produces
- **Workspace analyzers** — how to analyze an existing workspace for planning context

```python
class BaseDomain:
    name: str
    description: str

    def get_tools(self) -> list[str]:
        """Tool names available in this domain."""

    def get_prompt_templates(self) -> dict[str, str]:
        """Override paths for planner/executor/verifier/extractor templates."""

    def get_verification_rules(self) -> list[VerificationRule]:
        """Deterministic verification rules for this domain."""

    def get_memory_types(self) -> list[str]:
        """Valid memory entry types for this domain."""

    def get_artifact_types(self) -> list[str]:
        """Deliverable types this domain produces."""

    def analyze_workspace(self, path: Path) -> str:
        """Domain-specific workspace analysis for planning context."""
```

**Changes to existing code:**
- `config.py`: Add `domain: str = "auto"` to top-level config. When "auto", detect from workspace contents (has `.py`/`.js` files → code; has `.xlsx` files → finance; etc.)
- `orchestrator.py`: Load domain from config, pass to planning and tool resolution
- `registry.py`: Filter tools by domain during `create_default_registry()`

#### 0.2 Generalize ToolResult

In `registry.py`, rename `files_changed` to `artifacts_changed` with backward compatibility:

```python
@dataclass
class ToolResult:
    success: bool
    output: str
    artifacts_changed: list[str] = field(default_factory=list)  # was files_changed
    data: dict = field(default_factory=dict)

    @property
    def files_changed(self) -> list[str]:
        """Backward compat alias."""
        return self.artifacts_changed
```

#### 0.3 Remove Hard-Coded Code Analysis from Orchestrator

In `orchestrator.py:451-466`, replace:
```python
code_analysis = await self._analyze_workspace(workspace_path)
```
with:
```python
workspace_context = await self.domain.analyze_workspace(workspace_path)
```

The `CodeDomain.analyze_workspace()` preserves the current `analyze_directory()` behavior. Other domains provide their own analysis.

**Estimated scope:** ~400 lines new code, ~50 lines modified in existing files. No test breakage — `CodeDomain` preserves all current behavior as the default.

---

### Phase 1: Domain-Aware Prompt Templates

**Goal:** Each domain gets its own set of prompt templates that speak its language.

#### 1.1 Template Directory Structure

```
src/loom/prompts/templates/
    code/
        planner.yaml
        executor.yaml
        verifier.yaml
        extractor.yaml
        replanner.yaml
    marketing/
        planner.yaml      # Marketing strategy planning language
        executor.yaml      # Marketing execution constraints
        verifier.yaml      # Brand/audience verification
        extractor.yaml     # Marketing knowledge extraction
        replanner.yaml     # Stakeholder-feedback replanning
    finance/
        planner.yaml       # Financial analysis planning
        executor.yaml      # Data accuracy constraints
        verifier.yaml      # Model validation, ratio checks
        extractor.yaml     # Financial insight extraction
        replanner.yaml     # Market-change replanning
    consulting/
        planner.yaml       # Issue-tree decomposition
        executor.yaml      # MECE and hypothesis-driven
        verifier.yaml      # Completeness and rigor
        extractor.yaml     # Decision/recommendation extraction
        replanner.yaml     # Stakeholder-redirect replanning
    general/
        planner.yaml       # Fallback for unrecognized domains
        executor.yaml
        verifier.yaml
        extractor.yaml
        replanner.yaml
```

#### 1.2 Marketing Prompt Templates (Key Differences)

**Planner:**
- Role: "You are a marketing strategist who decomposes complex marketing projects into research phases, strategy phases, and execution phases."
- Instead of "file tools and shell": "You have access to web research, data analysis, document creation, and spreadsheet tools."
- Subtask model_tier guidance: Tier 1 for data gathering, Tier 2 for synthesis and strategy, Tier 3 for creative and final review.
- Verification_tier guidance: Tier 1 for data accuracy checks, Tier 2 for strategic coherence review.
- Dependency graph awareness: "Research phases (market sizing, competitive analysis, customer segmentation) can run in parallel. Strategy phases (positioning, channel selection) depend on research completion. Execution phases depend on strategy."

**Executor:**
- Constraints: "Cite all data sources. Verify claims with multiple sources. Maintain brand voice consistency. Flag assumptions explicitly."
- Instead of "Read actual files before editing": "Cross-reference existing research before making claims. Use the most recent available data."

**Verifier:**
- Checks: "Is the market sizing triangulated (top-down vs. bottom-up)? Does the competitive analysis cover direct and indirect competitors? Does positioning differentiate from identified competitors? Does the channel strategy match the target audience's media consumption?"

**Extractor:**
- Memory types: `market_insight`, `competitor_move`, `audience_preference`, `brand_guideline`, `channel_performance`, `decision`, `assumption`
- Instead of `tool_result`: `research_finding`

#### 1.3 Finance Prompt Templates (Key Differences)

**Planner:**
- Role: "You are a financial analyst who decomposes investment analysis into screening, research, modeling, valuation, and risk assessment phases."
- Subtask model_tier: Tier 1 for data gathering, Tier 2 for modeling, Tier 3 for memo synthesis.
- Dependency: "Business analysis and financial statement analysis can run in parallel after screening. Financial modeling depends on both. Valuation depends on the model. Risk assessment depends on valuation. The investment memo synthesizes everything."

**Executor:**
- Constraints: "All financial data must cite its source and date. Models must balance (assets = liabilities + equity). Projections must state underlying assumptions. Never extrapolate a single data point."

**Verifier:**
- Checks: "Do the three financial statements link correctly? Does the DCF terminal value represent < 85% of total value? Are projected margins within historical range? Do multiple valuation methods agree within 20%? Are key assumptions documented?"

**Extractor:**
- Memory types: `assumption`, `risk_factor`, `metric`, `forecast_adjustment`, `management_signal`, `industry_trend`, `decision`

#### 1.4 PromptAssembler Changes

Modify `prompts/assembler.py` to accept domain and load templates from the domain's directory:

```python
class PromptAssembler:
    def __init__(self, domain: str = "code", templates_dir: Path | None = None):
        if templates_dir is None:
            base = Path(__file__).parent / "templates"
            self.templates_dir = base / domain
            if not self.templates_dir.exists():
                self.templates_dir = base / "general"
```

**Estimated scope:** ~5 new YAML template files per domain (20 total across 4 domains), ~30 lines changed in `assembler.py`.

---

### Phase 2: General-Purpose Tools

**Goal:** Add tools that serve non-code domains without removing code tools.

#### 2.1 New Tools for Knowledge Work

| Tool | Description | Domains | Implementation |
|------|-------------|---------|---------------|
| `spreadsheet` | Create, read, edit spreadsheets (CSV/XLSX). Supports formulas, pivot summaries, charting data. | Finance, Marketing | Uses `openpyxl` for XLSX, `csv` stdlib for CSV |
| `document_write` | Create structured documents (Markdown → PDF/DOCX). Sections, headers, tables, citations. | All | Markdown generation with optional `python-docx` export |
| `data_analysis` | Statistical analysis, trend detection, correlation, regression on tabular data. | Finance, Marketing | Uses `pandas` + `numpy` (optional deps) |
| `web_research` | Enhanced web search with source aggregation. Fetches multiple sources, extracts key facts, cites origins. | All | Builds on existing `web_search` + `web_fetch` |
| `calculator` | Financial calculations: NPV, IRR, CAGR, DCF, ratios, amortization, compound interest. | Finance | Pure Python, no deps |
| `template_fill` | Fill document templates with data. Supports variable substitution in Markdown/DOCX templates. | Marketing, Consulting | Jinja2 or simple string replacement |
| `chart_data` | Generate chart-ready data structures (JSON) from tabular data. Bar, line, pie, scatter, waterfall. | All | Pure Python data transformation |
| `comparison_matrix` | Build structured comparison tables: features vs. competitors, options vs. criteria, weighted scoring. | Marketing, Consulting | Pure Python |

#### 2.2 Tool Implementation Strategy

Each new tool follows the existing `Tool` base class pattern:
1. Subclass `Tool`
2. Define `name`, `description`, `parameters`, `timeout_seconds`
3. Implement `async execute(self, args, ctx) -> ToolResult`
4. Place in `src/loom/tools/` — auto-discovered by `discover_tools()`

**Domain filtering:** The tool registry's `create_default_registry()` accepts an optional `domain` parameter. Tools declare which domains they serve via a new optional property:

```python
class Tool:
    @property
    def domains(self) -> list[str] | None:
        """Which domains this tool serves. None = all domains."""
        return None  # default: available everywhere
```

Code-specific tools (`shell`, `git`, `code_analysis`) return `["code"]`. New tools return their target domains. Generic tools (`web_search`, `web_fetch`, `read_file`, `task_tracker`) return `None` (available everywhere).

#### 2.3 Data Sources & API Integration Pattern

For tools that need external data (financial APIs, market research APIs), use a clean adapter pattern:

```python
class DataSourceAdapter:
    """Abstract adapter for external data sources."""
    async def query(self, params: dict) -> dict: ...
    async def validate_credentials(self) -> bool: ...
```

Concrete adapters:
- `AlphaVantageAdapter` — stock data, financial statements
- `SECEdgarAdapter` — company filings (free, no API key needed)
- `GoogleTrendsAdapter` — search trend data
- `StaticDataAdapter` — bundled reference data (industry benchmarks, financial formulas)

Config in `loom.toml`:
```toml
[data_sources.alpha_vantage]
api_key = "your-key"
rate_limit = 5  # calls/minute

[data_sources.sec_edgar]
user_agent = "YourName your@email.com"  # SEC requires this
```

**Estimated scope:** ~200-300 lines per new tool (8 tools = ~2000 lines), ~100 lines for data source adapters, ~50 lines for domain filtering in registry.

---

### Phase 3: Domain-Aware Verification

**Goal:** Extend the three-tier verification system to validate non-code deliverables.

#### 3.1 Verification Plugin Interface

```python
class VerificationPlugin:
    """Domain-specific deterministic verification."""

    def can_verify(self, artifact_path: str, artifact_type: str) -> bool:
        """Whether this plugin handles this artifact type."""

    async def verify(self, artifact_path: str, context: dict) -> list[VerificationCheck]:
        """Run deterministic checks. Returns pass/fail checks."""
```

#### 3.2 Verification Plugins by Domain

**Code (existing, extracted into plugin):**
- Python syntax (`compile()`)
- JSON validity (`json.loads()`)
- YAML validity (`yaml.safe_load()`)
- File existence and non-empty

**Finance:**
- Spreadsheet balance check: Assets = Liabilities + Equity (for 3-statement models)
- Formula integrity: no `#REF`, `#DIV/0`, `#VALUE!` errors
- Data type validation: financial columns are numeric, dates are valid
- Range reasonableness: margins 0-100%, growth rates -50% to +200%, WACC 5-15%
- Cross-sheet consistency: revenue in income statement matches revenue in DCF

**Marketing:**
- Document structure: required sections present (Executive Summary, Target Audience, Competitive Landscape, etc.)
- Data freshness: cited data sources are within 2 years
- Completeness: TAM/SAM/SOM all populated if market sizing task
- Brand consistency: company name spelled correctly throughout

**Consulting:**
- MECE check: subtask decomposition covers the full problem space without overlap
- Evidence-backed: each recommendation cites supporting data
- Stakeholder coverage: all identified stakeholders addressed

**General:**
- Word/character count within specified range
- Required sections present (configurable per task)
- No placeholder text remaining ("[TBD]", "TODO", "PLACEHOLDER")
- Citation format consistency

#### 3.3 Changes to DeterministicVerifier

In `verification.py`, replace the hard-coded syntax checks with a plugin dispatch:

```python
class DeterministicVerifier:
    def __init__(self, plugins: list[VerificationPlugin] | None = None):
        self.plugins = plugins or []

    async def verify(self, subtask, result) -> VerificationResult:
        checks = []
        # Existing generic checks (tool success, file existence) stay
        checks.extend(self._check_tool_success(result))
        checks.extend(self._check_artifacts(result))

        # Dispatch to domain plugins
        for artifact in result.artifacts_changed:
            for plugin in self.plugins:
                if plugin.can_verify(artifact, result.data.get("artifact_type", "")):
                    checks.extend(await plugin.verify(artifact, result.data))

        passed = all(c.passed for c in checks)
        return VerificationResult(tier=1, passed=passed, checks=checks, ...)
```

**Estimated scope:** ~150 lines for plugin interface + base, ~100-200 lines per domain plugin (4 domains = ~600 lines), ~50 lines refactored in existing `DeterministicVerifier`.

---

### Phase 4: Workspace Model Generalization

**Goal:** Support non-filesystem artifacts in the workspace and changelog system.

#### 4.1 Artifact Abstraction

```python
@dataclass
class Artifact:
    """A unit of work output, regardless of storage backend."""
    id: str                           # Unique identifier
    type: str                         # "file", "spreadsheet", "document", "data"
    path: str                         # Filesystem path, or logical path for non-file artifacts
    format: str                       # "py", "xlsx", "md", "csv", "json", "pdf"
    metadata: dict = field(default_factory=dict)  # Domain-specific metadata
```

#### 4.2 Changelog Generalization

The existing `ChangelogEntry` in `workspace.py` tracks file operations. Generalize:

```python
@dataclass
class ChangelogEntry:
    timestamp: str
    operation: str            # "create", "modify", "delete", "rename", "analyze", "review"
    artifact: Artifact        # What was changed (replaces raw file path)
    before_snapshot: str      # For files: path to copy. For data: serialized prior state.
    description: str          # Human-readable description of change
    subtask_id: str | None
```

New operations beyond file CRUD:
- `"analyze"` — ran analysis on existing data (no modification, but valuable context)
- `"review"` — verified/reviewed an artifact (quality gate passage)
- `"synthesize"` — combined multiple artifacts into one (e.g., research → memo)

#### 4.3 Diff Generalization

The `DiffGenerator` currently uses `difflib.unified_diff` for text files. Add format-aware diffing:

```python
class DiffGenerator:
    def generate_diff(self, artifact: Artifact, before: str, after: str) -> str:
        if artifact.format in ("csv", "xlsx"):
            return self._tabular_diff(before, after)
        elif artifact.format in ("md", "txt", "docx"):
            return self._text_diff(before, after)  # Existing unified_diff
        else:
            return self._text_diff(before, after)  # Fallback

    def _tabular_diff(self, before: str, after: str) -> str:
        """Show added/removed/changed rows and cells."""
        # Parse CSV/spreadsheet, compare row-by-row, highlight cell changes
```

#### 4.4 WorkspaceChanges Generalization

In `task_state.py`, generalize `WorkspaceChanges`:

```python
@dataclass
class DeliverableProgress:
    artifacts_created: int = 0
    artifacts_modified: int = 0
    artifacts_deleted: int = 0
    last_change: str = ""
    # Domain-specific counters (optional)
    custom_metrics: dict = field(default_factory=dict)

    # Backward compat
    @property
    def files_created(self) -> int: return self.artifacts_created
    @property
    def files_modified(self) -> int: return self.artifacts_modified
    @property
    def files_deleted(self) -> int: return self.artifacts_deleted
```

**Estimated scope:** ~200 lines for Artifact model, ~100 lines for changelog changes, ~100 lines for diff generalization, ~50 lines for task state changes.

---

### Phase 5: Memory & Learning Expansion

**Goal:** Extract and learn from domain-specific knowledge patterns.

#### 5.1 Extended Memory Entry Types

Add domain-specific memory types beyond the current six:

**Current:** `decision`, `error`, `tool_result`, `discovery`, `artifact`, `context`

**Added (cross-domain):**
- `assumption` — stated assumption that may need revisiting
- `risk` — identified risk or concern
- `metric` — quantitative finding (revenue figure, market size, ratio)
- `feedback` — stakeholder or verification feedback
- `constraint` — boundary condition or requirement
- `dependency` — inter-task or external dependency discovered

**Added (domain-specific, loaded from domain config):**
- Marketing: `competitor_insight`, `audience_preference`, `brand_guideline`, `channel_performance`
- Finance: `forecast_adjustment`, `management_signal`, `industry_trend`, `valuation_anchor`
- Consulting: `stakeholder_input`, `framework_decision`, `recommendation`, `implementation_risk`

#### 5.2 Memory Extraction Prompt Changes

The extractor prompt in `runner.py` currently focuses on tool execution context. Generalize:

**Current:** "Extract memory from the following tool calls and their results..."
**New:** "Extract key learnings from the following work. Focus on {domain_memory_types}. For each entry, specify the type, a concise content summary, and relevant tags."

The domain provides the memory type list and extraction guidance:

```python
class BaseDomain:
    def get_extraction_guidance(self) -> str:
        """Domain-specific guidance for memory extraction."""
        return "Extract decisions, discoveries, and errors from the work output."
```

Marketing override: "Extract market insights, competitor moves, audience preferences, and strategic decisions. Tag each with the relevant market segment and data source."

Finance override: "Extract assumptions used in modeling, risk factors identified, key metrics discovered, and any management signals from earnings transcripts. Tag each with ticker symbol and reporting period."

#### 5.3 Learning Manager Extension

In `learning/manager.py`, extend pattern extraction:

**Current patterns:** subtask_success, retry_pattern, task_template, model_failure

**Added patterns:**
- `verification_failure_pattern` — what types of verification failures recur (helps improve future planning)
- `domain_workflow_pattern` — successful task decomposition patterns per domain (e.g., "marketing strategy tasks that started with parallel market sizing + competitive analysis had higher success rates")
- `stakeholder_preference` — learned preferences from approval/rejection patterns

**Estimated scope:** ~100 lines for memory type extension, ~50 lines for extraction prompt changes, ~100 lines for learning manager extension.

---

### Phase 6: Enhanced Dependency Graph Support

**Goal:** Support the complex dependency patterns observed in white-collar workflows.

#### 6.1 Current State

The existing `Scheduler` in `engine/scheduler.py` supports:
- `depends_on: list[str]` per subtask
- Parallel dispatch of independent subtasks
- Blocking on unmet dependencies

This already supports fan-out/fan-in patterns. Enhancements needed:

#### 6.2 Conditional Dependencies (Gated Decision Points)

Add support for gate conditions on dependencies:

```python
@dataclass
class Subtask:
    # ... existing fields ...
    gate_condition: str | None = None  # "passed", "approved", "score > 0.8"
    on_gate_fail: str = "block"        # "block", "skip", "replan", "fail_task"
```

This enables:
- Due diligence: If Phase 2 (Exploratory DD) finds a deal-breaker, Phase 3 (Confirmatory DD) is skipped and the task terminates
- Investment analysis: If screening score < threshold, skip detailed analysis
- Marketing: If market sizing shows TAM < minimum, skip campaign planning

#### 6.3 Feedback Loop Support

Add support for iteration/feedback loops in the dependency graph:

```python
@dataclass
class Subtask:
    # ... existing fields ...
    feedback_target: str | None = None   # Subtask ID to send results back to
    max_iterations: int = 1              # How many feedback cycles before advancing
```

This enables:
- Marketing measurement → strategy revision cycles
- Financial model updates based on new earnings data
- Consulting hypothesis refinement based on data analysis

#### 6.4 Synthesis Bottleneck Handling

Add a subtask type for synthesis tasks that consume all prior outputs:

```python
@dataclass
class Subtask:
    # ... existing fields ...
    synthesis: bool = False  # If true, receives all prior subtask outputs as context
```

The orchestrator collects all completed subtask summaries and passes them as context to synthesis subtasks. This is critical for:
- Investment memos (synthesize all analysis)
- Marketing strategy decks (synthesize all research)
- Consulting recommendations (synthesize all workstream findings)

#### 6.5 Scheduler Changes

In `engine/scheduler.py`, extend `get_ready_subtasks()`:

```python
def get_ready_subtasks(self, completed: dict[str, SubtaskResult]) -> list[Subtask]:
    ready = []
    for subtask in self.pending:
        deps_met = all(d in completed for d in subtask.depends_on)
        if not deps_met:
            continue
        # NEW: Check gate conditions
        if subtask.gate_condition:
            gate_passed = self._evaluate_gate(subtask.gate_condition, completed)
            if not gate_passed:
                if subtask.on_gate_fail == "skip":
                    self._mark_skipped(subtask)
                    continue
                elif subtask.on_gate_fail == "fail_task":
                    raise TaskGateFailure(subtask)
                # "block" = don't add to ready, "replan" handled by orchestrator
                continue
        ready.append(subtask)
    return ready
```

**Estimated scope:** ~100 lines for gate conditions, ~50 lines for feedback loops, ~50 lines for synthesis handling, ~100 lines for scheduler changes.

---

### Phase 7: Database & Config Updates

**Goal:** Persist domain awareness through the stack.

#### 7.1 Schema Changes

```sql
-- Add to tasks table
ALTER TABLE tasks ADD COLUMN domain TEXT DEFAULT 'code';
ALTER TABLE tasks ADD COLUMN deadline_at TEXT;
ALTER TABLE tasks ADD COLUMN metadata TEXT DEFAULT '{}';  -- JSON: stakeholders, budget, etc.

-- Add to memory table
-- entry_type already TEXT, no schema change needed (just expand allowed values)
```

#### 7.2 Config Changes

```toml
# loom.toml additions

[domain]
default = "auto"  # auto-detect from workspace, or: code, marketing, finance, consulting, research, general

[domain.marketing]
default_tools = ["web_search", "web_fetch", "web_research", "spreadsheet", "document_write", "data_analysis", "comparison_matrix", "template_fill", "read_file", "glob_find", "task_tracker"]

[domain.finance]
default_tools = ["web_search", "web_fetch", "web_research", "spreadsheet", "document_write", "data_analysis", "calculator", "chart_data", "read_file", "glob_find", "task_tracker"]

[domain.consulting]
default_tools = ["web_search", "web_fetch", "web_research", "spreadsheet", "document_write", "data_analysis", "comparison_matrix", "template_fill", "read_file", "glob_find", "task_tracker"]

[data_sources]
# Optional API keys for external data
alpha_vantage_key = ""
sec_edgar_agent = ""
```

#### 7.3 CLI Changes

```bash
# Specify domain explicitly
loom cowork -w /path/to/project --domain marketing
loom run "Create a competitive analysis for Acme Corp" --domain marketing

# Auto-detect (default)
loom cowork -w /path/to/marketing-project  # detects from workspace contents
```

**Estimated scope:** ~30 lines for schema migration, ~50 lines for config additions, ~20 lines for CLI argument.

---

### Phase 8: Domain Packs (Stretch)

**Goal:** Package each domain as a self-contained "pack" that can be installed independently.

#### 8.1 Domain Pack Structure

```
loom-domain-marketing/
    __init__.py
    domain.py           # MarketingDomain class
    tools/
        spreadsheet.py
        comparison_matrix.py
    templates/
        planner.yaml
        executor.yaml
        verifier.yaml
        extractor.yaml
    verification/
        marketing_checks.py
    data/
        industry_benchmarks.json
        marketing_frameworks.json
```

#### 8.2 Pack Discovery

Similar to tool auto-discovery, domain packs register via entry points:

```toml
# In the domain pack's pyproject.toml
[project.entry-points."loom.domains"]
marketing = "loom_domain_marketing:MarketingDomain"
```

Loom discovers installed domain packs at startup and registers them.

#### 8.3 Bundled vs. External

- **Bundled (in core):** `code`, `general`
- **External packs:** `marketing`, `finance`, `consulting`, `research`

This keeps the core lean while allowing domain expertise to be maintained independently.

**Estimated scope:** ~200 lines for pack discovery infrastructure, domain packs are 500-1000 lines each.

---

## Part 4: Dependency Graph of Implementation Phases

```
Phase 0: Domain Abstraction Layer ─────────────────────────┐
    │                                                       │
    ├──► Phase 1: Domain-Aware Prompt Templates             │
    │         │                                             │
    ├──► Phase 2: General-Purpose Tools ────────────────────┤
    │         │                                             │
    └──► Phase 3: Domain-Aware Verification                 │
              │                                             │
              ▼                                             │
         Phase 4: Workspace Model Generalization            │
              │                                             │
              ▼                                             │
         Phase 5: Memory & Learning Expansion               │
              │                                             │
              ▼                                             │
         Phase 6: Enhanced Dependency Graph ────────────────┤
              │                                             │
              ▼                                             │
         Phase 7: Database & Config Updates                 │
              │                                             │
              ▼                                             │
         Phase 8: Domain Packs (Stretch) ──────────────────┘
```

**Phases 1, 2, 3 can run in parallel** after Phase 0 is complete — they are independent changes.
**Phase 4** depends on Phase 2 (tools create artifacts that need workspace tracking).
**Phase 5** depends on Phases 1 and 3 (memory extraction uses prompts, learning uses verification results).
**Phase 6** is independent but benefits from Phase 0's domain awareness.
**Phase 7** can start anytime after Phase 0.
**Phase 8** depends on everything.

---

## Part 5: Effort Estimates & Prioritization

| Phase | New Code | Modified Code | Test Impact | Priority |
|-------|----------|--------------|-------------|----------|
| 0. Domain Abstraction | ~400 LOC | ~50 LOC | No breakage | **Must-have** |
| 1. Prompt Templates | ~20 YAML files | ~30 LOC | No breakage | **Must-have** |
| 2. General Tools | ~2000 LOC | ~50 LOC | New tests needed | **Must-have** |
| 3. Verification Plugins | ~800 LOC | ~50 LOC | New tests needed | **Should-have** |
| 4. Workspace Generalization | ~450 LOC | ~100 LOC | Backward-compat | **Should-have** |
| 5. Memory & Learning | ~250 LOC | ~100 LOC | Backward-compat | **Nice-to-have** |
| 6. Dependency Graph | ~300 LOC | ~100 LOC | New tests needed | **Should-have** |
| 7. DB & Config | ~100 LOC | ~50 LOC | Migration needed | **Must-have** |
| 8. Domain Packs | ~3000 LOC | ~200 LOC | New tests needed | **Stretch** |
| **Total** | **~7300 LOC** | **~730 LOC** | | |

**Minimum viable expansion (Phases 0+1+2+7):** ~2550 LOC new, ~180 LOC modified. This gives domain selection, domain-appropriate prompts, new tools, and config support. The existing verification and workspace systems still work (just with code-centric defaults).

---

## Part 6: Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Domain detection is wrong | User gets wrong tools/prompts | Allow explicit `--domain` override; default to "general" not "code" when ambiguous |
| New tools have no tests | Quality regression | Require test coverage for each new tool before merge |
| Prompt templates diverge | Maintenance burden | Share common sections via YAML anchors or template inheritance |
| External API dependencies | Fragile, rate-limited | All API adapters optional; tools degrade gracefully to web search |
| Scope creep per domain | Never ships | MVP per domain: 1 planning prompt + 2-3 tools. Expand from real usage |
| Breaking changes to existing code workflow | User trust loss | Phase 0 specifically preserves `CodeDomain` as default; all backward-compat aliases |

---

## Part 7: Success Criteria

A successful expansion means a user can:

1. **Marketing:** `loom cowork -w ~/marketing-project --domain marketing` and have a productive session creating a competitive analysis, market sizing model, and positioning strategy — with the model using web research, spreadsheet, and document tools.

2. **Finance:** `loom run "Analyze AAPL as an investment opportunity" --domain finance` and get a multi-phase task that screens the stock, analyzes financials, builds a simple valuation model, assesses risks, and produces an investment memo — all verified for financial reasonableness.

3. **Consulting:** `loom cowork -w ~/client-project --domain consulting` and decompose a client problem using issue trees, gather evidence across workstreams, synthesize findings, and produce a recommendation document.

4. **Code (unchanged):** All existing code workflows continue to work identically. The 685 existing tests still pass. No user-facing behavior changes unless `--domain` is explicitly set.

---

## Refinement Round 1: Structural & Logical Gaps

Critical review of the plan above, identifying issues that would cause problems during implementation.

### R1.1 Cowork Mode Is Missing from the Plan

**Problem:** The entire plan focuses on task mode (plan → execute → verify) but the user's primary interaction mode is cowork mode — an interactive REPL where the model uses tools in a conversation loop with no planning phase. The plan never mentions how domains apply to cowork mode.

**Fix:** Domain awareness must apply to cowork mode too:
- `CoworkSession.__init__()` accepts a domain parameter
- The domain determines which tools are registered in the cowork session's tool registry
- The system prompt in cowork mode includes domain context ("You are a marketing strategist working interactively..." vs. "You are a software engineer...")
- The `ask_user` tool remains available in all domains
- Cowork mode is actually the *better* fit for non-code work, since marketing/finance tasks are inherently collaborative and iterative rather than fully autonomous

**Impact:** Add to Phase 0: cowork session domain wiring. Add to Phase 1: cowork system prompt templates per domain.

### R1.2 Tool Domain Filtering Is Too Restrictive

**Problem:** The plan says code-specific tools (`shell`, `git`, `code_analysis`) return `domains = ["code"]`, making them unavailable in other domains. This is wrong. A financial analyst might need `shell` to run `python analysis.py` or `curl` an API. A marketing strategist might use `shell` to process data with `csvtool` or `jq`.

**Fix:** Change the approach from "tools declare which domains they belong to" to "domains declare which tools they *recommend*." All tools remain *available* in all domains. The domain influences:
1. Which tools appear in the model's tool list by default (the domain's recommended set)
2. The prompt template, which guides the model toward domain-appropriate tools
3. An optional `strict_domain: bool = false` config that, when true, actually restricts tools

This is more pragmatic — the model can use `shell` in marketing mode if it needs to, but it won't be encouraged to.

**Impact:** Phase 2 tool filtering changes from whitelist to recommendation list.

### R1.3 Auto-Detection Is Fragile and Should Not Be Default

**Problem:** The plan proposes `domain: "auto"` as default, detecting from workspace contents. But many workspaces have mixed content (a marketing project with Python data analysis scripts; a code project with .xlsx test fixtures). Auto-detection will misclassify regularly.

**Fix:** Change defaults:
- Default domain is `"general"` (not `"auto"` and not `"code"`)
- `"general"` provides all tools and generic prompts — it works for everything, just not optimally
- `"auto"` is available but requires explicit opt-in: `domain = "auto"` in config
- Auto-detection heuristic: majority file type by count, with a confidence threshold. Below threshold → fall back to `"general"`
- When running from a workspace that already has a `.loom/` directory or `loom.toml` with a domain set, use that

**Impact:** Phase 0 and Phase 7 config changes.

### R1.4 Data Source Reality Check

**Problem:** The plan lists SimilarWeb, SEMrush, Bloomberg, Salesforce, etc. as data sources. But these are behind paywalls, complex authentication, and API rate limits. Loom cannot realistically integrate with Bloomberg Terminal or Salesforce CRM out of the box.

**Fix:** Tier the data sources by accessibility:

**Tier 1 — Free, no auth, ship in core:**
- SEC EDGAR (free, requires User-Agent header only)
- Web search via DuckDuckGo (already implemented)
- Web fetch (already implemented)
- Google Trends (via unofficial API or scraping — fragile)
- Wikipedia / public reference data

**Tier 2 — Free tier with API key, ship as optional:**
- Alpha Vantage (free: 25 calls/day)
- Financial Modeling Prep (free: 250 calls/day)
- News API (free: 100 calls/day)

**Tier 3 — Paid, external domain pack only:**
- SimilarWeb API, SEMrush API, Ahrefs API
- Bloomberg, FactSet, Capital IQ
- Salesforce, HubSpot CRM

The plan should not promise Tier 3 integrations. Phase 2 tools should use Tier 1 and Tier 2 sources. Tier 3 is Phase 8 (domain packs) territory.

**Impact:** Phase 2 data source section becomes more realistic. Remove specific adapter classes for Tier 3 sources.

### R1.5 Missing: User-Provided Context and Seed Data

**Problem:** In real workflows, users bring existing data — prior research, spreadsheets, brand guidelines, financial statements, existing models. The plan doesn't address how domain workspaces are seeded with initial context.

**Fix:** Add workspace context loading to Phase 0:
- Domain's `analyze_workspace()` scans for relevant existing files (the code domain already does this with `analyze_directory()`)
- Marketing domain: scans for existing brand guidelines, research docs, competitor lists
- Finance domain: scans for existing spreadsheets, financial data files, prior memos
- The planner prompt includes a summary of existing workspace contents as context
- In cowork mode, the model can `read_file` existing documents to incorporate them

This is actually already partially solved — `read_file` and `glob_find` are universal tools. The domain just needs to prompt the model to look for relevant existing data first.

**Impact:** Minor addition to domain's `analyze_workspace()` method descriptions.

### R1.6 Feedback Loops Are Over-Engineered

**Problem:** Phase 6.3 adds `feedback_target` and `max_iterations` to the Subtask dataclass. This is complex machinery that duplicates what replanning already provides. When a measurement subtask reveals that the strategy needs revision, the orchestrator's replan mechanism already handles this.

**Fix:** Remove the explicit feedback loop mechanism from Phase 6. Instead, enhance the replanning prompt (Phase 1) to be domain-aware about when replanning is appropriate:
- Marketing replanner: "If measurement results show campaign performance below target, revise the channel strategy and campaign plan"
- Finance replanner: "If new data changes key assumptions by >10%, update the model and re-run valuation"

Keep Phase 6 focused on gate conditions (6.2) and synthesis handling (6.4), which are genuinely new capabilities.

**Impact:** Phase 6 becomes simpler. Remove Section 6.3.

### R1.7 Spreadsheet Tool Needs Careful Parameter Design

**Problem:** The `spreadsheet` tool is described as "Create, read, edit spreadsheets" but LLMs are notoriously bad at precise cell-level operations. Saying "put CAGR formula in cell D15" requires exact parameter specification that models frequently get wrong.

**Fix:** Design the spreadsheet tool with operations that match LLM capabilities:
- `create_spreadsheet(name, headers, rows)` — create from structured data (the LLM provides JSON, the tool formats it)
- `read_spreadsheet(path, sheet, range)` — read a range and return as table
- `add_rows(path, sheet, rows)` — append data
- `add_column(path, sheet, column_name, formula_or_values)` — add a calculated column
- `create_pivot(path, sheet, group_by, aggregate)` — create a pivot summary
- `export_csv(path)` — export to CSV

Avoid operations that require cell-level addressing (A1, B2, etc.). Instead, use column names and row indices. This matches how LLMs naturally think about tabular data.

**Impact:** Phase 2 spreadsheet tool specification becomes more detailed and realistic.

### R1.8 Missing Migration Strategy for Schema Changes

**Problem:** Phase 7 proposes `ALTER TABLE tasks ADD COLUMN domain ...` but doesn't address how existing databases are migrated. Loom stores data in SQLite. Users with existing sessions will have old schemas.

**Fix:** Add a migration system:
- On startup, check schema version (store in a `schema_version` table or use SQLite `PRAGMA user_version`)
- Apply pending migrations sequentially
- Each migration is a Python function with up/down SQL
- Keep migrations in `src/loom/state/migrations/`

This is ~100 lines of infrastructure plus ~20 lines per migration. It's a one-time investment that pays off for all future schema changes.

**Impact:** Add migration subsystem to Phase 7.

### R1.9 Plan Dependency Graph Has Errors

**Problem:** The plan says "Phase 4 depends on Phase 2 (tools create artifacts that need workspace tracking)" and "Phase 5 depends on Phases 1 and 3." But Phase 4 (workspace generalization) is actually independent of Phase 2 (tools) — you can generalize the workspace model without the new tools existing. And Phase 5 (memory) could start as soon as Phase 0 is done, since it's primarily about prompt changes and memory type extension.

**Fix:** Corrected dependency graph:
```
Phase 0: Domain Abstraction ──────────────────┐
    │                                          │
    ├──► Phase 1: Prompt Templates ────────────┤
    │                                          │
    ├──► Phase 2: General-Purpose Tools ───────┤
    │                                          │
    ├──► Phase 3: Verification Plugins ────────┤
    │                                          │
    ├──► Phase 4: Workspace Generalization ────┤
    │                                          │
    ├──► Phase 5: Memory & Learning ───────────┤
    │                                          │
    ├──► Phase 6: Dependency Graph ────────────┤
    │                                          │
    └──► Phase 7: Database & Config ───────────┤
                                               │
         Phase 8: Domain Packs ────────────────┘
                 (depends on all above)
```

Phases 1-7 all depend only on Phase 0 and can run in parallel. Phase 8 depends on all of them.

### R1.10 Few-Shot Examples Missing from Planner Templates

**Problem:** The plan says marketing planner template should guide the model to create dependency graphs, but LLMs need examples. Without few-shot examples of marketing plans with correct dependency structures, the planner will produce suboptimal decompositions.

**Fix:** Each domain's planner template should include 1-2 example plans:

Marketing planner example:
```yaml
# Example: For goal "Create a go-to-market strategy for Product X"
# Subtasks:
#   1. market_sizing (depends_on: [], model_tier: 1) - Estimate TAM/SAM/SOM
#   2. competitive_analysis (depends_on: [], model_tier: 1) - Map competitor landscape
#   3. customer_segmentation (depends_on: [market_sizing], model_tier: 2) - Define target segments
#   4. positioning (depends_on: [market_sizing, competitive_analysis, customer_segmentation], model_tier: 2) - Craft positioning
#   5. channel_strategy (depends_on: [positioning], model_tier: 2) - Select channels and allocate budget
#   6. campaign_plan (depends_on: [channel_strategy], model_tier: 2) - Design campaigns
```

This is the single highest-leverage change for quality. Without it, the model will create flat, sequential plans instead of the parallel fan-out/fan-in structure that makes Loom's parallelism valuable.

**Impact:** Phase 1 templates become longer but dramatically more effective.

---

## Refinement Round 2: Feasibility, Prioritization & Practical Details

Deeper review after examining the actual template files and existing abstractions.

### R2.1 Template Changes Are Smaller Than Expected

**Discovery:** Reading the actual planner.yaml, executor.yaml, verifier.yaml, extractor.yaml, and replanner.yaml reveals they are already quite generic. The code-specific language is concentrated in just a few lines:

- **planner.yaml line 8:** "can be completed by a model with access to file tools and shell" — this is the *only* code-specific line in the role section
- **planner.yaml line 24:** `CODE STRUCTURE ANALYSIS: {code_analysis}` — domain-specific context variable
- **planner.yaml line 46:** "Set model_tier: 1 for simple file ops, 2 for code generation" — tier guidance is code-centric
- **executor.yaml line 32:** "Do NOT fabricate file contents or data. Read actual files before editing." — file-centric constraint
- **extractor.yaml line 17:** entry types `decision|error|tool_result|discovery|artifact|context` — code-centric type list

**Everything else is already domain-agnostic.** The verifier template has zero code-specific language. The replanner template has zero code-specific language.

**Revised approach:** Instead of creating 5 entirely new template files per domain (20 files total), create a *template override* system:
1. Load base templates (current ones)
2. Domain provides a small override dict that replaces specific sections
3. For example, the marketing domain overrides only: planner.role, planner.constraints, executor.constraints, extractor.instructions

This reduces 20 new YAML files to ~4 small override configs per domain. Dramatically less maintenance.

```python
class BaseDomain:
    def get_template_overrides(self) -> dict[str, dict[str, str]]:
        """Return template section overrides.

        Example: {"planner": {"role": "You are a marketing strategist..."}}
        Sections not overridden use the base template.
        """
        return {}
```

**Impact:** Phase 1 effort drops from ~20 YAML files to ~4 override dicts per domain. Much more maintainable.

### R2.2 The `{code_analysis}` Variable Generalizes Naturally

**Discovery:** The planner template has `{code_analysis}` as a variable that gets filled by `_analyze_workspace()`. The plan already proposes replacing this with `workspace_context = await self.domain.analyze_workspace(workspace_path)`. But the template variable name needs to change too.

**Fix:** Rename the template variable from `{code_analysis}` to `{workspace_analysis}`:

```yaml
# In planner.yaml (updated)
WORKSPACE ANALYSIS:
{workspace_analysis}
```

The `CodeDomain.analyze_workspace()` returns the same code structure analysis as before. `MarketingDomain.analyze_workspace()` returns a summary of existing documents, spreadsheets, and data files.

This is a one-line change in the base planner template + one-line change in the assembler.

### R2.3 The Existing Scheduler Is Simpler Than Expected

**Discovery:** Reading `scheduler.py`, it's only 58 lines. The `Scheduler` class has just 4 methods: `next_subtask()`, `runnable_subtasks()`, `has_pending()`, `is_blocked()`. There's no state — it reads from the plan's subtask statuses.

**Implication for Phase 6 (gate conditions):** The scheduler doesn't modify state; it just reads it. Gate evaluation would need the orchestrator to pass completed subtask results to the scheduler, which currently only reads subtask *status* (not result content). This requires changing the scheduler's interface.

**Revised approach:** Keep gate conditions simple — evaluate them in the orchestrator, not the scheduler. The orchestrator already has access to completed results. Before dispatching a subtask:

```python
# In orchestrator, not scheduler
for subtask in scheduler.runnable_subtasks(plan):
    if subtask.gate_condition:
        if not self._evaluate_gate(subtask, completed_results):
            subtask.status = SubtaskStatus.SKIPPED
            continue
    batch.append(subtask)
```

This avoids changing the scheduler's clean, stateless interface.

### R2.4 Synthesis Subtasks Need a Context Assembly Mechanism

**Problem:** Phase 6.4 proposes `synthesis: bool = False` on subtasks that should receive all prior outputs. But the current executor prompt template only passes `{memory_entries_formatted}` as context from prior work. Memory entries are extracted summaries, not full outputs.

For a synthesis task like "write the investment memo," the model needs access to the *full outputs* of prior subtasks, not just memory summaries. A 150-character memory entry saying "DCF valuation: $45-52/share" is insufficient to write a 10-page memo.

**Fix:** For synthesis subtasks, the executor prompt should include a `{prior_subtask_outputs}` variable containing the full summaries of all completed subtasks in the current task. The orchestrator already stores `SubtaskResult.summary` for each completed subtask. Concatenate these and inject into the prompt.

```yaml
# In executor.yaml, add conditional section:
prior_outputs: |
  COMPLETED WORK (use as input for synthesis):
  {prior_subtask_outputs}
```

The assembler includes this section only when the subtask has `synthesis: true`.

**Impact:** Small change to executor template and assembler, but critical for the quality of synthesis tasks.

### R2.5 The `document_write` Tool Should Generate Markdown, Not DOCX/PDF

**Problem:** The plan proposes `document_write` with "Markdown → PDF/DOCX" conversion. But converting to PDF/DOCX requires heavy dependencies (`python-docx`, `weasyprint` or `pdfkit`+`wkhtmltopdf`). These are complex to install and fragile.

**Fix:** The `document_write` tool should:
1. Generate Markdown files (no dependencies needed)
2. Support a structured parameter format: `sections: [{title, content, level}]`
3. Optionally export to HTML (trivial with `markdown` library, which is lightweight)
4. Leave PDF/DOCX conversion as a user-installed extension (Phase 8 domain pack territory)

Markdown is the right intermediate format because:
- LLMs naturally generate it
- It's readable as-is in any editor
- Git tracks changes to it
- Pandoc (if user has it) converts to any format
- The existing `read_file` tool can read it back

### R2.6 The `calculator` Tool Is Actually the Most Important Finance Tool

**Discovery:** The plan lists `calculator` as a simple "Financial calculations: NPV, IRR, CAGR" tool. But for financial analysis, computation is the core bottleneck — LLMs cannot reliably multiply 3-digit numbers.

**Revised specification:** The `calculator` tool should be more capable:
- **Arithmetic:** Basic operations on numbers (solves the LLM arithmetic problem)
- **Financial formulas:** NPV, IRR, XIRR, CAGR, WACC, DCF per-share value, amortization tables
- **Ratio calculations:** P/E, EV/EBITDA, D/E, ROE, ROA, current ratio, quick ratio — from raw financial data
- **Compound operations:** "Given revenue of $100M growing at 15% for 5 years, what's year-5 revenue?" — chain of calculations
- **Table operations:** "Calculate CAGR for each row in this dataset" — batch computation

The key insight: this tool turns the LLM from "bad at math" to "great at math by delegation." It should be available in ALL domains, not just finance.

### R2.7 Revised MVP (Minimum Viable Expansion)

After R1 and R2, the MVP shrinks to something that can ship quickly:

**MVP = Phase 0 (partial) + Phase 1 (overrides) + 3 key tools + Phase 7 (config only)**

1. **Domain config field** in `loom.toml` — just the `domain = "general"` default + override ability (10 lines)
2. **Domain class with template overrides** — `BaseDomain`, `CodeDomain`, `GeneralDomain`, `MarketingDomain`, `FinanceDomain` with override dicts (200 lines)
3. **Rename `{code_analysis}` → `{workspace_analysis}`** in planner template (2 lines changed)
4. **PromptAssembler domain-aware template loading** with override system (30 lines)
5. **Three new tools:**
   - `calculator` — arithmetic + financial formulas (150 lines)
   - `spreadsheet` — create/read/edit CSV+XLSX with column-level operations (250 lines)
   - `document_write` — structured Markdown document creation (100 lines)
6. **`--domain` CLI flag** for `loom cowork` and `loom run` (10 lines)

**Total MVP: ~750 lines new, ~50 lines modified.** Ships a functional marketing/finance experience.

Everything else (verification plugins, workspace generalization, memory expansion, dependency graph enhancements, domain packs) is incremental improvement that can come later based on actual usage feedback.

### R2.8 The `web_research` Tool Is Redundant

**Problem:** The plan proposes a `web_research` tool that "fetches multiple sources, extracts key facts, cites origins." But in cowork mode, the model already does this naturally — it calls `web_search`, reads the results, calls `web_fetch` on relevant URLs, and synthesizes. Adding a meta-tool that does this automatically is:
1. Duplicating behavior the model already exhibits
2. Harder to debug (what if it fetches the wrong sources?)
3. Less transparent to the user

**Fix:** Remove `web_research` from Phase 2. Instead, ensure the domain prompt templates encourage the model to use `web_search` and `web_fetch` together for research tasks. The model's natural research behavior is already good enough.

### R2.9 Approval Gates Need Domain Awareness

**Discovery:** The current approval system in `cowork/approval.py` auto-approves read-only tools and prompts for write/execute tools. But in non-code domains, the "dangerous" operations are different:
- In code: `shell`, `git push`, `delete_file` are dangerous
- In marketing: publishing content, sending emails, modifying CRM data would be dangerous (but these don't exist yet)
- In finance: there are no dangerous operations — it's all analysis and document creation

**Fix:** For the MVP, this is a non-issue because the new tools (calculator, spreadsheet, document_write) are all local file operations, which should be auto-approved like `read_file`. No approval system changes needed initially.

Long-term: when external API integrations exist (Tier 3 sources, CRM writes, email sends), the domain should define which tools require approval. Add `get_approval_required_tools() -> list[str]` to `BaseDomain`.

### R2.10 Test Strategy for New Tools

Each new tool needs:
1. **Unit tests** with mocked filesystem (no actual file I/O in tests)
2. **Parameter validation tests** (missing required params, wrong types)
3. **Edge case tests** (empty data, huge data, special characters)
4. **Integration test** with the tool registry (auto-discovery works)

For the spreadsheet tool specifically:
- Test CSV and XLSX creation with various data types
- Test reading back created files
- Test add_column with formulas
- Test with Unicode data

Estimate: ~50 test lines per tool, ~200 lines total for 3 MVP tools.

---

## Refinement Round 3: Consolidated Architecture & Execution Plan

Final consolidation pass. Reconciles all refinements into a coherent, implementable architecture.

### R3.1 Final Architecture After All Refinements

Incorporating R1 and R2 findings, the final architecture is:

```
loom/
├── src/loom/
│   ├── domains/                    # NEW: Phase 0
│   │   ├── __init__.py             # BaseDomain, DomainRegistry, get_domain()
│   │   ├── code.py                 # CodeDomain (preserves all current behavior)
│   │   ├── general.py              # GeneralDomain (all tools, generic prompts)
│   │   ├── marketing.py            # MarketingDomain (template overrides, workspace analysis)
│   │   ├── finance.py              # FinanceDomain (template overrides, workspace analysis)
│   │   └── consulting.py           # ConsultingDomain (template overrides, workspace analysis)
│   │
│   ├── tools/                      # EXTENDED: Phase 2
│   │   ├── calculator.py           # NEW: arithmetic + financial formulas
│   │   ├── spreadsheet.py          # NEW: CSV/XLSX create/read/edit
│   │   ├── document_write.py       # NEW: structured Markdown generation
│   │   └── ... (existing tools unchanged)
│   │
│   ├── prompts/
│   │   ├── assembler.py            # MODIFIED: accepts domain, applies overrides
│   │   └── templates/
│   │       ├── planner.yaml        # MODIFIED: {code_analysis} → {workspace_analysis}
│   │       └── ... (other templates unchanged)
│   │
│   ├── engine/
│   │   ├── orchestrator.py         # MODIFIED: domain.analyze_workspace() replaces hard-coded code analysis
│   │   └── ... (others unchanged)
│   │
│   ├── config.py                   # MODIFIED: add domain field
│   └── __main__.py                 # MODIFIED: --domain CLI flag
```

Key architectural decisions after refinement:
1. **Template overrides, not template copies** (R2.1) — domains provide small override dicts, not full template files
2. **Tools are recommended, not restricted** (R1.2) — all tools available in all domains; domain influences model behavior via prompts
3. **Default domain is "general", not "auto"** (R1.3) — explicit is better than magic
4. **Only Tier 1-2 data sources** (R1.4) — free/freemium APIs only; Tier 3 is Phase 8
5. **No web_research meta-tool** (R2.8) — model's natural web_search + web_fetch behavior is sufficient
6. **No feedback_target loops** (R1.6) — replanning already handles this
7. **Markdown documents, not PDF/DOCX** (R2.5) — lightweight, LLM-native format
8. **Column-level spreadsheet operations** (R1.7) — avoid cell-level addressing
9. **Calculator in all domains** (R2.6) — solves LLM math weakness universally

### R3.2 Final Implementation Order (Post-Refinement)

Based on all refinements, the implementation is reorganized into 4 batches:

**Batch A: MVP (Ship First)**
This is the minimum set to enable non-code use. Everything in this batch is required.

| Step | What | Files | Lines | Depends On |
|------|------|-------|-------|-----------|
| A1 | `BaseDomain`, `CodeDomain`, `GeneralDomain` | `domains/__init__.py`, `domains/code.py`, `domains/general.py` | ~200 | — |
| A2 | Config: add `domain` field | `config.py` | ~15 | — |
| A3 | CLI: add `--domain` flag | `__main__.py` | ~10 | A2 |
| A4 | Planner template: `{code_analysis}` → `{workspace_analysis}` | `prompts/templates/planner.yaml` | ~2 | — |
| A5 | PromptAssembler: domain-aware with override system | `prompts/assembler.py` | ~40 | A1 |
| A6 | Orchestrator: use `domain.analyze_workspace()` | `engine/orchestrator.py` | ~15 | A1 |
| A7 | `calculator` tool | `tools/calculator.py` | ~200 | — |
| A8 | `spreadsheet` tool | `tools/spreadsheet.py` | ~300 | — |
| A9 | `document_write` tool | `tools/document_write.py` | ~150 | — |
| A10 | Tests for A1-A9 | `tests/` | ~300 | A1-A9 |

**Total Batch A: ~1230 lines. All existing tests still pass.**

**Batch B: Domain Templates (Ship Second)**
Domain-specific template overrides and workspace analyzers.

| Step | What | Files | Lines | Depends On |
|------|------|-------|-------|-----------|
| B1 | `MarketingDomain` with template overrides + few-shot examples | `domains/marketing.py` | ~150 | Batch A |
| B2 | `FinanceDomain` with template overrides + few-shot examples | `domains/finance.py` | ~150 | Batch A |
| B3 | `ConsultingDomain` with template overrides + few-shot examples | `domains/consulting.py` | ~120 | Batch A |
| B4 | Executor template: add `{prior_subtask_outputs}` section | `prompts/templates/executor.yaml`, `prompts/assembler.py` | ~20 | Batch A |
| B5 | Cowork mode domain awareness (system prompt, tool context) | `cowork/session.py` | ~30 | Batch A |
| B6 | Tests for B1-B5 | `tests/` | ~150 | B1-B5 |

**Total Batch B: ~620 lines.**

**Batch C: Verification & Dependency Graph (Ship Third)**
Enhanced verification plugins and gate conditions.

| Step | What | Files | Lines | Depends On |
|------|------|-------|-------|-----------|
| C1 | `VerificationPlugin` interface | `engine/verification.py` | ~50 | — |
| C2 | Extract current code checks into `CodeVerificationPlugin` | `engine/verification.py` | ~80 (refactor, not new) | C1 |
| C3 | `FinanceVerificationPlugin` (balance checks, range checks) | `domains/finance_verification.py` | ~150 | C1 |
| C4 | `GeneralVerificationPlugin` (placeholder check, section check) | `domains/general_verification.py` | ~80 | C1 |
| C5 | Gate conditions on subtasks | `state/task_state.py`, `engine/orchestrator.py` | ~100 | — |
| C6 | Synthesis subtask support | `engine/orchestrator.py`, `prompts/assembler.py` | ~60 | B4 |
| C7 | Tests for C1-C6 | `tests/` | ~200 | C1-C6 |

**Total Batch C: ~720 lines.**

**Batch D: Polish & Extensions (Ship Last)**
Memory, learning, workspace, schema, data sources.

| Step | What | Files | Lines | Depends On |
|------|------|-------|-------|-----------|
| D1 | Extended memory entry types | `state/memory.py`, `engine/runner.py` | ~60 | Batch B |
| D2 | Schema migration system | `state/migrations/` | ~120 | — |
| D3 | Add `domain` column to tasks table | `state/schema.sql`, `state/migrations/001_add_domain.py` | ~30 | D2 |
| D4 | Workspace artifact abstraction (`ToolResult.artifacts_changed`) | `tools/registry.py` | ~20 | — |
| D5 | Data source adapter pattern + SEC EDGAR adapter | `data_sources/` | ~200 | — |
| D6 | Learning manager domain patterns | `learning/manager.py` | ~80 | Batch B |
| D7 | Tests for D1-D6 | `tests/` | ~200 | D1-D6 |

**Total Batch D: ~710 lines.**

**Grand total: ~3280 lines across all 4 batches.** This is significantly less than the original estimate of ~7300 lines because:
- Template overrides instead of full template copies (R2.1)
- Removed web_research tool (R2.8)
- Removed feedback loop machinery (R1.6)
- Removed PDF/DOCX conversion (R2.5)
- Simplified tool filtering (R1.2)

### R3.3 What "Done" Looks Like After Each Batch

**After Batch A (MVP):**
```bash
loom cowork -w ~/analysis --domain general
> Can you create a market sizing spreadsheet for the US electric vehicle charging market?
# Model uses web_search → calculator → spreadsheet to build a sizing model
# Works, but prompts are generic — model doesn't know marketing methodology
```

**After Batch B (Domain Templates):**
```bash
loom cowork -w ~/analysis --domain marketing
> Create a competitive analysis for Tesla Supercharger vs ChargePoint vs Electrify America
# Model knows to: research each competitor, build a comparison matrix, identify differentiation opportunities
# Planner creates parallel research subtasks with proper dependencies
```

```bash
loom run "Analyze AAPL as an investment" --domain finance -w ~/investments
# Planner creates: screening → business analysis || financial analysis → modeling → valuation → risk → memo
# Phases 2-3 run in parallel. Phase 7 (memo) is a synthesis subtask.
```

**After Batch C (Verification & Gates):**
```bash
# Same finance task, but now:
# - Spreadsheet models are verified (do the statements balance?)
# - DCF terminal value is checked (< 85% of total?)
# - Gate condition on screening: if screening fails filters, skip detailed analysis
```

**After Batch D (Polish):**
```bash
# Same tasks, but now:
# - Memory extracts domain-specific insights (assumptions, risks, metrics)
# - Learning system records successful marketing plan structures
# - SEC EDGAR data available for finance domain
# - Domain stored in database for task filtering
```

### R3.4 Risks That Remain After All Refinements

| Risk | Likelihood | Impact | Mitigation Already In Plan |
|------|-----------|--------|---------------------------|
| LLMs produce low-quality marketing/finance analysis | HIGH | Users lose trust | Template few-shot examples (R1.10); verification checks (Batch C); but ultimately limited by model capability |
| Calculator precision issues (floating point) | MEDIUM | Financial numbers slightly off | Use Python's `decimal.Decimal` for financial calculations |
| Spreadsheet tool can't handle complex Excel features | HIGH | Users expect full Excel | Tool description clearly states capabilities; complex Excel is out of scope |
| Domain auto-detection rarely used | LOW | Feature goes unused | Default to "general" (R1.3); CLI flag is the primary mechanism |
| Prompt template overrides become stale | MEDIUM | Domain prompts drift from base | Override system inherits base template improvements automatically |

### R3.5 What This Plan Does NOT Cover (Explicit Non-Goals)

1. **Real-time data feeds** — No Bloomberg Terminal integration, no live market data streaming
2. **Interactive visualizations** — Charts are data structures (JSON), not rendered images
3. **Email/Slack/CRM integration** — No sending or receiving external communications
4. **Multi-user collaboration** — Single-user tool; no shared sessions or permissions
5. **Regulatory compliance** — Tool provides no guarantees about financial advice accuracy
6. **PDF/DOCX/PPTX generation** — Markdown only; conversion is the user's responsibility
7. **Custom domain creation by users** — Users choose from built-in domains; custom domains are code changes

These are deliberately excluded to keep scope manageable. Each could be a future Phase 8 domain pack.

### R3.6 Final Effort Summary

| Batch | Lines | Calendar Effort | Cumulative Value |
|-------|-------|-----------------|-----------------|
| A (MVP) | ~1230 | First | Loom works for non-code tasks with generic prompts and 3 new tools |
| B (Domains) | ~620 | Second | Domain-specific intelligence makes marketing/finance tasks significantly better |
| C (Verification) | ~720 | Third | Quality gates catch common errors in financial models and documents |
| D (Polish) | ~710 | Fourth | Memory, learning, and data sources make repeated use more effective |
| **Total** | **~3280** | | **Full general-purpose expansion** |

The MVP (Batch A) is a clean, shippable increment that adds immediate value. Each subsequent batch adds incremental capability without blocking the prior batch from being useful.

---

## Refinement Round 4: Process Definition Plugin Architecture

**Supersedes Phases 0, 1, and 8 from the original plan and Refinement Rounds 1-3.**

Rounds 1-3 refined an architecture where domains are Python classes (`MarketingDomain`, `FinanceDomain`) with hardcoded template overrides. This round replaces that entire approach with something fundamentally better: **declarative process definition files** that turn Loom's engine into a generic orchestration machine that anyone can specialize without writing code.

### R4.1 Core Insight: Separate Engine from Intelligence

The current architecture tangles two concerns:

1. **The engine** — plan decomposition, dependency scheduling, tool dispatch, verification, retry, memory. This is domain-agnostic and should stay in Python.
2. **The intelligence** — *how* to decompose a marketing strategy, *what* makes a good financial model, *which* tools to use for competitive analysis, *what* to verify in a spreadsheet. This is domain knowledge and should NOT be in Python.

The process definition file is where domain intelligence lives. The engine reads it and adapts its behavior accordingly. No Python classes per domain. No template overrides in code. Just YAML files that describe how to work.

### R4.2 Process Definition File Format

A process definition is a single YAML file that lives in any of:

```
~/.loom/processes/marketing-strategy.yaml      # User-global processes
./loom-processes/investment-analysis.yaml       # Workspace-local processes
~/.loom/processes/my-custom-workflow.yaml       # User-created processes
```

Here's the complete format:

```yaml
# ─── PROCESS DEFINITION ───
# File: marketing-strategy.yaml

# ── Metadata ──
name: marketing-strategy
version: "1.0"
description: >
  Full marketing strategy development from market research through
  campaign planning. Follows standard STP (Segmentation, Targeting,
  Positioning) methodology with parallel research phases feeding into
  sequential strategy development.
author: "Loom Team"
tags: [marketing, strategy, research, go-to-market]

# ── Persona ──
# Injected as the role section of every prompt (planner, executor, verifier).
# This is the single most impactful section — it shapes all model behavior.
persona: |
  You are a senior marketing strategist with deep expertise in market
  research, competitive analysis, customer segmentation, and go-to-market
  strategy. You follow evidence-based marketing principles and always
  support claims with data and citations.

  Your approach:
  - Research before strategy. Never position without competitive context.
  - Triangulate data. Cross-reference multiple sources for any key metric.
  - Think in segments. One-size-fits-all strategies are lazy strategies.
  - Quantify everything. Market sizes, growth rates, CAC targets — no vague claims.
  - Cite sources. Every data point needs an origin.

# ── Tool Guidance ──
# Tells the model which tools to prefer and how to use them in this context.
# Does NOT restrict tools — all registered tools remain available.
tool_guidance: |
  For this type of work, prioritize these tools:
  - Use `web_search` and `web_fetch` for market research and competitive data.
  - Use `calculator` for market sizing math, growth projections, and ROI estimates.
  - Use `spreadsheet` for building comparison matrices, sizing models, and budgets.
  - Use `document_write` for strategy briefs, positioning documents, and campaign plans.
  - Use `read_file` to review any existing brand guidelines, prior research, or templates
    in the workspace before starting new work.

# ── Phase Templates ──
# Pre-defined subtask structures that the planner should use as its blueprint.
# The planner can adapt these (add/remove phases) based on the specific goal,
# but this gives it the right starting structure and dependency graph.
phases:
  - id: market-sizing
    description: >
      Estimate the Total Addressable Market (TAM), Serviceable Addressable
      Market (SAM), and Serviceable Obtainable Market (SOM). Use both
      top-down (industry reports, analyst estimates) and bottom-up
      (unit economics: customers × average revenue) approaches.
      Triangulate and reconcile if they differ by more than 15%.
    depends_on: []
    model_tier: 2
    verification_tier: 2
    acceptance_criteria: >
      TAM, SAM, and SOM are estimated with both methods. Sources are cited.
      Top-down and bottom-up estimates are within 15% or discrepancy is explained.
    deliverables:
      - "market-sizing-model.csv — TAM/SAM/SOM with assumptions"
      - "market-sizing-summary.md — narrative with methodology and sources"

  - id: competitive-analysis
    description: >
      Identify and analyze 3-8 direct and indirect competitors. For each:
      website traffic (if available via web search), pricing model, target
      customer, key differentiators, strengths, and weaknesses. Build a
      comparison matrix and identify white-space opportunities.
    depends_on: []
    model_tier: 2
    verification_tier: 2
    acceptance_criteria: >
      At least 3 competitors analyzed. Comparison matrix created as spreadsheet.
      Each competitor has pricing, positioning, and differentiator data.
      White-space opportunities identified.
    deliverables:
      - "competitive-matrix.csv — structured comparison"
      - "competitor-profiles.md — detailed profiles with SWOT"

  - id: customer-segmentation
    description: >
      Define 2-5 customer segments using demographic, behavioral, and
      psychographic variables. Size each segment using the market sizing
      data. Create a persona document for each primary segment including
      needs, pain points, media consumption, and buying triggers.
    depends_on: [market-sizing]
    model_tier: 2
    verification_tier: 2
    acceptance_criteria: >
      2-5 segments defined with sizing. Primary persona document created.
      Segments are mutually exclusive and collectively exhaustive (MECE).
    deliverables:
      - "customer-segments.csv — segment definitions with sizing"
      - "personas.md — detailed persona documents"

  - id: positioning
    description: >
      Craft the positioning statement, value proposition, and messaging
      framework. The positioning must differentiate from identified
      competitors and resonate with target segments. Include: positioning
      statement (For [target], [brand] is the [category] that [benefit]
      because [reason to believe]), messaging hierarchy, and 3-5 proof points.
    depends_on: [market-sizing, competitive-analysis, customer-segmentation]
    model_tier: 3
    verification_tier: 2
    is_synthesis: true
    acceptance_criteria: >
      Positioning statement follows standard format. Differentiates from
      top 3 competitors. Maps to at least one target segment's needs.
      Messaging framework has hierarchy (primary, secondary, tertiary).
    deliverables:
      - "positioning-strategy.md — positioning statement, messaging, proof points"

  - id: channel-strategy
    description: >
      Recommend optimal marketing channels based on where target segments
      are reachable and what the competitive landscape shows. Allocate
      a hypothetical budget across channels with expected ROI ranges.
      Include: channel selection rationale, budget split, KPI targets per channel.
    depends_on: [positioning]
    model_tier: 2
    verification_tier: 1
    acceptance_criteria: >
      At least 3 channels recommended with rationale. Budget allocation
      spreadsheet created. KPI targets set per channel.
    deliverables:
      - "channel-strategy.csv — channel budget allocation and KPIs"
      - "channel-strategy.md — rationale and implementation notes"

  - id: campaign-plan
    description: >
      Create a campaign plan for the first 90 days. Include campaign
      themes, content calendar, key messages per channel, and measurement
      plan. Tie each campaign element back to the positioning and target segments.
    depends_on: [channel-strategy]
    model_tier: 2
    verification_tier: 1
    acceptance_criteria: >
      90-day campaign plan with at least 3 campaign themes. Each tied
      to positioning and segments. Measurement plan with specific KPIs.
    deliverables:
      - "campaign-plan.md — full 90-day plan"
      - "content-calendar.csv — week-by-week content plan"

# ── Verification Rules ──
# Domain-specific deterministic checks applied during Tier 1 verification.
# These supplement (not replace) the engine's built-in checks.
verification:
  rules:
    - name: sources-cited
      description: "All data claims must cite a source"
      check: "output contains at least one URL or source reference per data point"
      severity: warning  # warning | error

    - name: no-placeholders
      description: "No placeholder text in deliverables"
      check: "deliverables do not contain [TBD], [TODO], [INSERT], or [PLACEHOLDER]"
      severity: error

    - name: segments-mece
      description: "Customer segments should be mutually exclusive and collectively exhaustive"
      check: "segmentation covers the full market without overlap"
      severity: warning

# ── Memory Guidance ──
# Tells the memory extractor what kinds of knowledge to extract and retain
# between subtasks in this process.
memory:
  extract_types:
    - type: market_insight
      description: "Quantitative market data (sizes, growth rates, shares)"
    - type: competitor_move
      description: "Competitor actions, strategies, or positioning"
    - type: audience_preference
      description: "Target audience behaviors, preferences, or pain points"
    - type: strategic_decision
      description: "Strategic choices made and their rationale"
    - type: assumption
      description: "Assumptions that may need revisiting with new data"

  extraction_guidance: |
    Focus on extracting data points with their sources, strategic
    decisions with rationale, and assumptions that downstream phases
    depend on. Tag each entry with the relevant market segment when applicable.

# ── Workspace Initialization ──
# Guidance for analyzing an existing workspace before planning.
workspace_analysis:
  scan_for:
    - "*.md — existing strategy documents, brand guidelines, briefs"
    - "*.csv, *.xlsx — existing data, prior research, competitor data"
    - "*.pdf — industry reports, analyst presentations"
  guidance: |
    Before planning, scan the workspace for existing materials. If brand
    guidelines exist, the positioning phase must respect them. If prior
    research exists, reference it rather than re-doing it.

# ── Planner Examples ──
# Few-shot examples showing the planner how to decompose goals in this domain.
# These are injected into the planner prompt as examples.
planner_examples:
  - goal: "Create a go-to-market strategy for a B2B SaaS product"
    subtasks:
      - id: market-sizing
        description: "Estimate TAM/SAM/SOM for the B2B SaaS vertical"
        depends_on: []
        model_tier: 2
      - id: competitive-analysis
        description: "Map 5-7 competitors: pricing, features, positioning"
        depends_on: []
        model_tier: 2
      - id: customer-segmentation
        description: "Define 3-4 ICP segments with firmographic and behavioral criteria"
        depends_on: [market-sizing]
        model_tier: 2
      - id: positioning
        description: "Craft positioning statement and messaging framework"
        depends_on: [market-sizing, competitive-analysis, customer-segmentation]
        model_tier: 3
      - id: channel-strategy
        description: "Select channels and allocate budget"
        depends_on: [positioning]
        model_tier: 2
      - id: launch-plan
        description: "Create 90-day launch plan with campaigns and KPIs"
        depends_on: [channel-strategy]
        model_tier: 2

# ── Replanning Guidance ──
# Domain-specific guidance for when and how to revise the plan.
replanning:
  triggers: |
    Consider replanning if:
    - Market sizing reveals the market is too small (TAM < stated minimum)
    - Competitive analysis reveals an unexpected dominant player
    - Customer research contradicts initial segment assumptions
    - Stakeholder feedback changes the target market or positioning constraints
  guidance: |
    When replanning, preserve completed research phases. Revise strategy
    phases (positioning, channels) based on new data. Do not re-do research
    that already produced valid results.
```

### R4.3 How the Engine Consumes Process Definitions

The engine doesn't need domain-specific Python code. It reads the process definition and wires it into existing extension points:

```
┌──────────────────────────────────────────────────┐
│                 Process Definition                │
│              (marketing-strategy.yaml)            │
└───────┬──────┬──────┬──────┬──────┬──────┬───────┘
        │      │      │      │      │      │
        ▼      ▼      ▼      ▼      ▼      ▼
   ┌────────┬──────┬───────┬──────┬──────┬────────┐
   │Persona │Phases│Tool   │Verify│Memory│Planner │
   │        │      │Guide  │Rules │Guide │Examples│
   └───┬────┴──┬───┴───┬───┴──┬───┴──┬───┴───┬────┘
       │       │       │      │      │       │
       ▼       ▼       ▼      ▼      ▼       ▼
   ┌────────────────────────────────────────────┐
   │              Prompt Assembler               │
   │  (injects sections into existing templates) │
   └──────────────┬─────────────────────────────┘
                  │
                  ▼
   ┌────────────────────────────────────────────┐
   │          Orchestrator (unchanged)           │
   │  plan → dispatch → verify → learn cycle    │
   └────────────────────────────────────────────┘
```

**Key: the engine code doesn't change.** The process definition is loaded, parsed into a `ProcessDefinition` dataclass, and passed to the `PromptAssembler` which injects the relevant sections into prompts.

### R4.4 ProcessDefinition Dataclass

```python
@dataclass
class PhaseTemplate:
    id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    model_tier: int = 2
    verification_tier: int = 1
    is_critical_path: bool = False
    is_synthesis: bool = False
    acceptance_criteria: str = ""
    deliverables: list[str] = field(default_factory=list)

@dataclass
class VerificationRule:
    name: str
    description: str
    check: str             # Natural-language check description (LLM-evaluated)
    severity: str = "warning"  # "warning" or "error"

@dataclass
class MemoryType:
    type: str
    description: str

@dataclass
class PlannerExample:
    goal: str
    subtasks: list[dict]   # Simplified subtask dicts for few-shot

@dataclass
class ProcessDefinition:
    # Metadata
    name: str
    version: str = "1.0"
    description: str = ""
    author: str = ""
    tags: list[str] = field(default_factory=list)

    # Behavior
    persona: str = ""                                    # Injected as role
    tool_guidance: str = ""                              # Injected into executor
    phases: list[PhaseTemplate] = field(default_factory=list)  # Blueprint for planner
    verification: list[VerificationRule] = field(default_factory=list)
    memory_types: list[MemoryType] = field(default_factory=list)
    extraction_guidance: str = ""
    workspace_scan: list[str] = field(default_factory=list)
    workspace_guidance: str = ""
    planner_examples: list[PlannerExample] = field(default_factory=list)
    replanning_triggers: str = ""
    replanning_guidance: str = ""
```

### R4.5 Loading & Discovery

```python
class ProcessLoader:
    """Discovers and loads process definition files."""

    SEARCH_PATHS = [
        Path.cwd() / "loom-processes",      # Workspace-local
        Path.cwd() / ".loom" / "processes",  # Workspace .loom dir
        Path.home() / ".loom" / "processes", # User-global
    ]

    def discover(self) -> dict[str, Path]:
        """Find all available process definitions. Returns {name: path}."""
        processes = {}
        # Later paths take precedence (user-global < workspace-local)
        for search_dir in reversed(self.SEARCH_PATHS):
            if search_dir.exists():
                for yaml_file in search_dir.glob("*.yaml"):
                    defn = self._quick_parse_name(yaml_file)
                    if defn:
                        processes[defn] = yaml_file
        return processes

    def load(self, name_or_path: str) -> ProcessDefinition:
        """Load a process definition by name or file path."""
        path = Path(name_or_path)
        if not path.exists():
            # Search by name
            available = self.discover()
            if name_or_path not in available:
                raise ProcessNotFound(name_or_path, list(available.keys()))
            path = available[name_or_path]
        return self._parse(path)

    def _parse(self, path: Path) -> ProcessDefinition:
        """Parse YAML into ProcessDefinition."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        # ... validate and construct ProcessDefinition
```

### R4.6 PromptAssembler Integration

The assembler gains one new parameter — an optional `ProcessDefinition`:

```python
class PromptAssembler:
    def __init__(self, templates_dir=None, process: ProcessDefinition | None = None):
        self._process = process
        # ... existing init

    def build_planner_prompt(self, task, workspace_listing="", workspace_analysis=""):
        template = self.get_template("planner")
        role = template.get("role", "").strip()

        # INJECT: process persona overrides the generic role
        if self._process and self._process.persona:
            role = self._process.persona.strip()

        # INJECT: phase templates as blueprint for the planner
        phase_blueprint = ""
        if self._process and self._process.phases:
            phase_blueprint = self._format_phase_blueprint(self._process.phases)

        # INJECT: few-shot examples
        examples = ""
        if self._process and self._process.planner_examples:
            examples = self._format_planner_examples(self._process.planner_examples)

        # Existing variable substitution + new sections
        replacements = {
            "goal": task.goal,
            "workspace_path": task.workspace,
            "workspace_analysis": workspace_analysis or "Not analyzed.",
            "phase_blueprint": phase_blueprint,
            "planner_examples": examples,
            # ... existing vars
        }
        # ... assemble as before

    def build_executor_prompt(self, task, subtask, state_manager, ...):
        template = self.get_template("executor")
        role = template.get("role", "").strip()

        # INJECT: process persona
        if self._process and self._process.persona:
            role = self._process.persona.strip()

        # INJECT: tool guidance as additional constraints
        extra_constraints = ""
        if self._process and self._process.tool_guidance:
            extra_constraints = self._process.tool_guidance.strip()

        # ... rest of existing assembly, append extra_constraints to constraints section
```

**The base templates (planner.yaml, executor.yaml, etc.) stay generic.** The process definition injects domain intelligence *into* them. If no process is loaded, everything works exactly as it does today.

### R4.7 What Changes in the Planner Template

Only one structural addition — placeholders for process-injected content:

```yaml
# planner.yaml (updated)
role: |
  You are a task planning assistant. Your job is to decompose a complex goal
  into a sequence of concrete, independently executable subtasks.

  Each subtask must be:
  - Atomic: accomplishes one specific thing
  - Verifiable: has clear success criteria
  - Scoped: can be completed by a model with the available tools

instructions: |
  GOAL:
  {goal}

  WORKSPACE:
  {workspace_path}

  CONTEXT PROVIDED BY USER:
  {user_context}

  WORKSPACE CONTENTS:
  {workspace_listing}

  WORKSPACE ANALYSIS:
  {workspace_analysis}

  RECOMMENDED PHASE STRUCTURE:
  {phase_blueprint}

  EXAMPLE DECOMPOSITIONS:
  {planner_examples}

  Decompose this goal into subtasks. You may use the recommended phase
  structure as a starting point, adapting it to the specific goal. Add,
  remove, or modify phases as needed.

  Respond with ONLY a JSON object:
  ...
```

When no process is loaded, `{phase_blueprint}` and `{planner_examples}` resolve to empty strings, and the planner works exactly as before — fully generic decomposition.

### R4.8 Verification Rule Execution

Process verification rules are natural-language descriptions, not code. They're evaluated by the existing Tier 2 LLM verifier:

```python
# In verification.py, extend the verifier prompt:
def _build_verification_prompt(self, subtask, result, process_rules):
    base_prompt = self._prompts.build_verifier_prompt(subtask, result_summary, tool_calls)

    if process_rules:
        rules_text = "\n".join(
            f"- [{r.severity.upper()}] {r.name}: {r.check}"
            for r in process_rules
        )
        base_prompt += f"\n\nADDITIONAL DOMAIN-SPECIFIC CHECKS:\n{rules_text}"

    return base_prompt
```

The LLM verifier already evaluates acceptance criteria. Process verification rules are just additional criteria injected into the same prompt. No new verification infrastructure needed.

### R4.9 CLI & Config Integration

```bash
# Specify a process by name (discovered from search paths)
loom run "Create a GTM strategy for our new API product" --process marketing-strategy

# Specify a process by file path
loom run "Analyze AAPL" --process ./my-processes/equity-analysis.yaml

# Interactive mode with a process
loom cowork -w ~/projects/acme --process marketing-strategy

# List available processes
loom processes
#   marketing-strategy    v1.0  Full marketing strategy development
#   investment-analysis   v1.0  Investment analysis and memo generation
#   competitive-intel     v1.0  Quick competitive intelligence report

# No process = current behavior (generic code-focused)
loom cowork -w ~/code-project
```

Config in `loom.toml`:
```toml
[process]
# Default process (used when --process not specified)
# Empty = no process (current behavior)
default = ""

# Additional process search paths
search_paths = ["~/my-processes"]
```

### R4.10 Example Process Definitions

Beyond marketing strategy, here are other processes that would ship as built-in examples:

**investment-analysis.yaml** — Screens a stock, analyzes fundamentals, builds a valuation model, assesses risks, produces an investment memo. Phases are heavily sequential. Verification checks financial statement linkage, terminal value reasonableness, ratio ranges.

**competitive-intel.yaml** — Quick 3-phase process: identify competitors, research each, build comparison matrix. Lightweight version of the marketing strategy's competitive analysis phase. Good for ad-hoc requests.

**research-report.yaml** — General research workflow: frame question, gather evidence from multiple sources, synthesize findings, write report with citations. Applies to any knowledge domain.

**consulting-engagement.yaml** — McKinsey-style problem decomposition: define the issue tree (MECE), assign workstreams, gather evidence per workstream, synthesize recommendations. Includes red-team verification.

**due-diligence.yaml** — Gated process with explicit go/no-go decisions after each phase. If early screening finds deal-breakers, later phases are skipped.

Users create their own by copying and editing these examples.

### R4.11 Why This Is Better Than Hardcoded Domains

| Aspect | Hardcoded Domains (Old Plan) | Process Definitions (New Plan) |
|--------|------------------------------|-------------------------------|
| Adding a new domain | Write a Python class, edit registry, redeploy | Write a YAML file, drop it in a folder |
| Customizing behavior | Fork the code or wait for upstream changes | Edit the YAML file |
| Sharing expertise | Package as a Python module | Share a YAML file |
| User learning curve | None (domains are hidden) | Low (read the YAML to understand the process) |
| Engine complexity | N domain classes, override resolution, registry | 1 ProcessLoader, 1 ProcessDefinition dataclass |
| Versioning | Tied to Loom releases | Independent — processes version separately |
| Testing | Need Python tests per domain | Process definitions are data — validate schema only |
| Community contribution | PR to main repo, Python knowledge required | Share YAML files, no programming needed |

### R4.12 Revised Implementation Plan (Process-Based)

This replaces the 4-batch plan from R3.2 with a cleaner approach:

**Batch A: Process Engine (Ship First)**

| Step | What | Lines |
|------|------|-------|
| A1 | `ProcessDefinition` dataclass + `ProcessLoader` | ~200 |
| A2 | `PromptAssembler` gains `process` parameter, injects persona/phases/examples/tool_guidance | ~80 |
| A3 | Planner template: add `{phase_blueprint}`, `{planner_examples}`, `{workspace_analysis}` | ~10 |
| A4 | Orchestrator: load process, pass to assembler, use `workspace_analysis` | ~30 |
| A5 | Verifier: inject process verification rules into LLM verifier prompt | ~20 |
| A6 | Memory extractor: inject process memory guidance into extractor prompt | ~15 |
| A7 | CLI: `--process` flag, `loom processes` command | ~40 |
| A8 | Config: `[process]` section in loom.toml | ~15 |
| A9 | Tests for A1-A8 | ~200 |
| **Total** | | **~610** |

**Batch B: New Tools (Ship Second)**

| Step | What | Lines |
|------|------|-------|
| B1 | `calculator` tool — arithmetic + financial formulas | ~200 |
| B2 | `spreadsheet` tool — CSV/XLSX create/read/edit (column-level) | ~300 |
| B3 | `document_write` tool — structured Markdown generation | ~150 |
| B4 | Tests for B1-B3 | ~200 |
| **Total** | | **~850** |

**Batch C: Built-In Process Definitions (Ship Third)**

| Step | What | Lines |
|------|------|-------|
| C1 | `marketing-strategy.yaml` — full process definition as shown above | ~150 |
| C2 | `investment-analysis.yaml` — finance process with sequential phases | ~140 |
| C3 | `research-report.yaml` — general research workflow | ~80 |
| C4 | `competitive-intel.yaml` — lightweight competitive analysis | ~70 |
| C5 | `consulting-engagement.yaml` — McKinsey-style problem solving | ~100 |
| C6 | Validation tests for all process definitions | ~100 |
| **Total** | | **~640** |

**Batch D: Polish (Ship Last)**

| Step | What | Lines |
|------|------|-------|
| D1 | Schema migration system | ~120 |
| D2 | Synthesis subtask support (`is_synthesis` flag in executor prompt) | ~40 |
| D3 | Gate conditions for gated processes like due-diligence | ~80 |
| D4 | Cowork mode process awareness (system prompt from persona) | ~30 |
| D5 | `ToolResult.artifacts_changed` generalization | ~20 |
| D6 | Tests for D1-D5 | ~150 |
| **Total** | | **~440** |

**Grand total: ~2540 lines.** Down from 3280 in R3, down from 7300 in the original plan.

### R4.13 Final Architecture

```
~/.loom/
├── processes/                      # User-global process definitions
│   ├── marketing-strategy.yaml
│   ├── investment-analysis.yaml
│   ├── my-custom-process.yaml      # User-created
│   └── ...
├── loom.toml                       # Global config
└── loom.db                         # SQLite database

~/workspace/
├── loom-processes/                  # Workspace-local processes (override global)
│   └── our-team-workflow.yaml
├── loom.toml                       # Workspace config (overrides global)
└── ... (project files)

src/loom/
├── processes/
│   ├── __init__.py                 # ProcessDefinition, ProcessLoader
│   ├── schema.py                   # YAML schema validation
│   └── builtin/                    # Ships with Loom
│       ├── marketing-strategy.yaml
│       ├── investment-analysis.yaml
│       ├── research-report.yaml
│       ├── competitive-intel.yaml
│       └── consulting-engagement.yaml
├── tools/
│   ├── calculator.py               # NEW
│   ├── spreadsheet.py              # NEW
│   ├── document_write.py           # NEW
│   └── ... (existing tools)
├── prompts/
│   ├── assembler.py                # MODIFIED: accepts ProcessDefinition
│   └── templates/
│       ├── planner.yaml            # MODIFIED: new placeholders
│       └── ... (unchanged)
├── engine/
│   ├── orchestrator.py             # MODIFIED: loads process, passes to assembler
│   └── verification.py             # MODIFIED: injects process verification rules
└── config.py                       # MODIFIED: [process] section
```

The engine is fully domain-agnostic. All domain intelligence lives in YAML files that anyone can create, edit, share, and version independently of the Loom codebase.

### R4.14 Future: Process Marketplace

Once the process definition format stabilizes, a natural evolution is a community repository of process definitions:

```bash
# Future concept (not in scope for this plan)
loom process install marketing-strategy
loom process install financial-due-diligence
loom process search "real estate"
```

This is just file discovery from a remote repository — no plugin system, no packaging, no dependencies. Just YAML files downloaded to `~/.loom/processes/`.

### R4.15 Summary: What Changed from R3

| R3 Architecture | R4 Architecture |
|----------------|----------------|
| `MarketingDomain` Python class | `marketing-strategy.yaml` process file |
| Template override dicts in code | `persona` and `tool_guidance` in YAML |
| `domains/` package with 5 Python files | `processes/` package with 1 loader + N YAML files |
| Tools restricted by domain | All tools available; process guides model via `tool_guidance` |
| Verification plugins in Python | Verification rules as natural-language checks in YAML |
| Domain detected from workspace files | Process selected by user via `--process` flag |
| 3280 LOC across 4 batches | **2540 LOC across 4 batches** |
| New domain = new Python class | **New process = new YAML file** |

---

## Criticism Round 1: Architectural Blind Spots and Runtime Gaps

### C1.1 The Tool Gap: Processes Can't Deliver Capability

R4's central claim is "new process = new YAML file, no code needed." But this only holds when the existing tools are sufficient. The moment a domain needs a capability the base toolset doesn't provide, the promise breaks:

- A financial analysis process needs a DCF model builder, a WACC calculator, or a financial statement normalizer. The generic `calculator` tool does arithmetic, but it doesn't know about cost-of-equity formulas or how to link income statement → balance sheet → cash flow.
- A legal review process needs a clause extraction tool, a contract comparison tool, or a regulatory citation validator.
- A data analysis process needs a `run_python` sandbox or a `sql_query` tool.

In R4, the user who needs a new tool must: (1) write a Python class extending `Tool`, (2) get it registered in the `ToolRegistry`, and (3) then reference it from their YAML. This makes the "no code" claim misleading — it's "no code *if the tools already exist*."

**The programming language analogy is precise here.** The base tools are the standard library. Process definitions are user programs. But there's no `import` mechanism — no way for a process definition to deliver the tools it depends on. Imagine Python where you can write scripts but can't install packages.

**Opportunity:** Process definitions should be able to bundle tool definitions. This doesn't mean arbitrary Python execution from YAML — it means a process *package* (a directory, not a single file) that can include:

```
marketing-strategy/
├── process.yaml          # The process definition (what R4 already defines)
├── tools/
│   ├── market_sizer.py   # Tool: Tool subclass, auto-registered when process loads
│   └── brand_scorer.py   # Tool: scoring rubric evaluator
└── templates/            # Optional: custom prompt fragments
    └── brand-voice.txt   # Included via {brand_voice} in persona
```

The `ProcessLoader` would scan the `tools/` directory, import the modules, and register the tools into the `ToolRegistry` before orchestration begins. Tools bundled with a process are only available when that process is active — they don't pollute the global tool namespace.

This makes process definitions a true extension unit: intelligence + capability in one package.

### C1.2 Verification Rules Are Under-Specified

The verification rules in R4 are natural-language strings evaluated by the LLM verifier:

```yaml
- name: sources-cited
  check: "output contains at least one URL or source reference per data point"
  severity: warning
```

This has two problems:

1. **No deterministic path.** Some checks are trivially deterministic — "deliverables do not contain [TBD]" is a regex. Running it through an LLM is wasteful and non-deterministic. The plan puts *all* process rules into Tier 2 (LLM) verification, but some belong in Tier 1 (deterministic).

2. **No check type taxonomy.** The `check` field is a free-form string. The engine has no way to know whether a check is regex-matchable, requires file inspection, or needs LLM judgment. This means you can't optimize, can't give useful error messages when a check is malformed, and can't route checks to the right tier.

**Opportunity:** Add an optional `type` field:

```yaml
- name: no-placeholders
  type: regex                    # Tier 1: deterministic
  check: "\\[TBD\\]|\\[TODO\\]|\\[INSERT\\]|\\[PLACEHOLDER\\]"
  target: deliverables           # What to check against
  severity: error

- name: sources-cited
  type: llm                     # Tier 2: needs judgment
  check: "Every quantitative claim cites a source"
  severity: warning
```

When `type: regex`, the `DeterministicVerifier` runs it. When `type: llm` (or omitted), it goes to the LLM verifier as currently planned. This keeps the "just write natural language" simplicity as the default while enabling fast, free, deterministic checks for the common cases.

### C1.3 The Planner's Relationship to Phase Templates Is Ambiguous

R4.7 says:

> "You may use the recommended phase structure as a starting point, adapting it to the specific goal. Add, remove, or modify phases as needed."

This creates an identity crisis. Are phase templates:

- **Prescriptive blueprints** that the planner should follow unless there's a strong reason not to?
- **Suggestive examples** that the planner can freely ignore?

The answer matters because it determines how much the process author can rely on their phases actually executing. A marketing strategist who carefully designs a 6-phase process with specific dependency ordering will be frustrated if the planner decides to skip competitive analysis because it didn't seem relevant.

**Opportunity:** Add a `strictness` field to the process definition:

```yaml
phase_mode: guided     # "strict" | "guided" | "suggestive"
# strict:     planner must use these phases (can adjust descriptions, not structure)
# guided:     planner should follow this structure, may add/modify, explain deviations
# suggestive: phases are just hints, planner decomposes freely
```

Default to `guided` for backward compatibility with R4's current semantics. `strict` is important for regulated workflows (compliance, audit) where the process must be followed exactly. `suggestive` is the current no-process behavior.

### C1.4 No Process Composition or Inheritance

Every process definition is a standalone island. There's no way to:

- **Compose:** "Run competitive-intel as phase 2 of marketing-strategy" (process-calls-process)
- **Inherit:** "This is like research-report but with financial verification rules" (extend a base)
- **Mixin:** "Add the citation-checking rules to any process" (reusable fragments)

This means the `consulting-engagement.yaml` and `research-report.yaml` processes will duplicate persona text, verification rules, and memory guidance — because they share a "research-then-synthesize" pattern.

**Not proposing full inheritance** (that way lies YAML Turing-completeness). But a simple `includes` mechanism would help:

```yaml
includes:
  - common/citation-rules.yaml    # Merges verification rules
  - common/research-persona.yaml  # Merges persona text
```

The merge semantics should be trivial: later values override earlier ones, lists are concatenated. This keeps it simple while preventing copy-paste rot across process definitions.

### C1.5 Workspace Analysis Runs Code Analysis for Non-Code Workspaces

The orchestrator's `_analyze_workspace` method (line 451-466 in orchestrator.py) calls `analyze_directory` from `loom.tools.code_analysis`, which parses Python ASTs, extracts classes/functions/imports. For a marketing workspace full of CSVs, PDFs, and Markdown files, this returns nothing useful.

The plan's `workspace_analysis` section in process definitions describes *what to scan for*, but the engine's actual implementation always runs code analysis. The process definition's `scan_for` guidance is never wired to anything — it's documentation masquerading as configuration.

**Opportunity:** The `_analyze_workspace` method should be process-aware. When a process defines `workspace_analysis.scan_for`, the engine should use that to decide what to analyze, not blindly run `code_analysis.analyze_directory`. For non-code workspaces, this means: list file types present, read headers of CSV files, extract titles from Markdown files, note PDF filenames. This is a different analysis than AST parsing.

---

## Criticism Round 2: Real-World Usage Scenarios and Edge Cases

### C2.1 Process Selection Is Manual — But Should Sometimes Be Automatic

R4 requires `--process marketing-strategy` on the command line. The R4.15 comparison table even lists this as an advantage: "Domain detected from workspace files → Process selected by user via --process flag."

But this creates friction for the most common case: the user has a workspace with existing process artifacts (market sizing spreadsheets, competitor analyses) and types `loom cowork`. They now need to remember which process they used last time. Worse, if they share a workspace with a colleague, the colleague needs to know which process to specify.

**Opportunity:** Allow an optional `loom-process.yaml` symlink or `[process]` setting in workspace-local `loom.toml`:

```toml
# workspace loom.toml
[process]
default = "marketing-strategy"
```

This already exists in R4.9's config section but only at the global level. Make it workspace-scoped too. When present, `--process` becomes optional. The workspace remembers its own process.

Additionally, process *auto-detection* isn't inherently bad — it was bad in R3 because it was based on fragile workspace file heuristics picking Python classes. A simpler version works: if there's exactly one `.yaml` file in `./loom-processes/`, use it. If there are multiple, list them and ask. This is discoverable and predictable.

### C2.2 Process Definitions Will Outgrow Single Files

The marketing-strategy.yaml example is already ~150 lines. A serious financial analysis process with detailed phase descriptions, 10+ verification rules, 5 planner examples, and extensive memory guidance will hit 400-500 lines. That's still manageable, but add the tool bundling from C1.1, custom prompt fragments, and example deliverables, and you're looking at a package.

R4 assumes process = single YAML file. This works for simple processes but doesn't scale.

**Opportunity:** Support both:

```
# Simple: single file (current R4 behavior)
~/.loom/processes/quick-research.yaml

# Complex: directory with manifest
~/.loom/processes/financial-analysis/
├── process.yaml              # Main definition
├── tools/                    # Bundled tools (C1.1)
│   └── dcf_builder.py
├── examples/                 # Planner few-shot examples (externalized)
│   ├── equity-analysis.yaml
│   └── fixed-income.yaml
├── verification/             # Verification rule sets (externalized)
│   └── financial-checks.yaml
└── templates/                # Prompt fragments
    └── risk-framework.txt
```

The `ProcessLoader` already discovers by scanning directories. Make it handle both: if the path is a `.yaml` file, load it directly (current behavior). If it's a directory, look for `process.yaml` inside it.

### C2.3 Version Field Has No Teeth

The `version: "1.0"` field exists but nothing consumes it. There's no:

- **Compatibility check:** "This process requires Loom engine ≥ 2.0"
- **Migration path:** "This process uses schema v2 features not in v1"
- **Upgrade warning:** "Your saved process is v1, the installed version is v2"

In practice, when the process definition format evolves (and it will — adding `type` to verification rules, adding `phase_mode`, adding `includes`), old YAML files will silently break or produce unexpected behavior.

**Opportunity:** Define what `version` means. The simplest approach: it's the *schema version*, not the content version. The engine validates against the declared schema version and gives clear errors:

```
Error: Process 'financial-analysis' uses schema v2 (phase_mode field)
but this version of Loom only supports schema v1. Upgrade Loom or
downgrade the process definition.
```

Content versioning (v1.0 → v1.1 of the marketing strategy) is the user's responsibility and doesn't need engine support.

### C2.4 Memory Guidance Has No Feedback Loop

The `memory.extract_types` section tells the extractor what *kinds* of knowledge to look for. But there's no mechanism for:

- **The planner to know what memory types exist** when building the plan. If market_insight memories are available from a prior run, the planner should know to check for them before scheduling a market-sizing phase.
- **Cross-process memory sharing.** If a user runs `competitive-intel` and then `marketing-strategy`, the competitive intelligence memories are tagged with `competitive-intel`'s memory types (if any), not `marketing-strategy`'s. The marketing process can't find them because memory retrieval is type-filtered.
- **Memory type validation.** Nothing prevents a process from defining `extract_types: [{type: "foo"}]` and the extractor producing entries tagged `bar`. The types are suggestions, not constraints.

**Opportunity:** Memory types in the process definition should feed into the memory *retrieval* prompt, not just the extraction prompt. When the planner or executor retrieves memories, the process's `extract_types` should guide what to search for. And cross-process memory sharing should work by defining a shared type namespace — or by making memory retrieval ignore types entirely and rely on semantic similarity (which the current system may already do).

### C2.5 The Deliverables Field Is Unverified

Phase templates specify `deliverables`:

```yaml
deliverables:
  - "market-sizing-model.csv — TAM/SAM/SOM with assumptions"
  - "market-sizing-summary.md — narrative with methodology and sources"
```

But nothing in the verification pipeline checks whether these specific files were actually created. The `DeterministicVerifier` checks `tool_calls.result.files_changed` for non-emptiness, but it doesn't cross-reference against the expected deliverables list.

This is a missed opportunity for a free Tier 1 check: "Phase declared deliverables X, Y, Z. Files actually created: X, Z. Missing: Y." This is deterministic, zero-cost, and immediately useful.

### C2.6 No Process-Level Error Budget or Retry Policy

Individual subtask retries exist in the engine. But processes have no way to express:

- "If any research phase fails, it's OK — skip it and note the gap in synthesis"
- "If the market-sizing phase fails, stop the entire process (it's a hard dependency)"
- "Allow up to 2 retries on web_search-heavy phases (they're flaky)"

The `is_critical_path` field exists on `PhaseTemplate` but it's not wired to anything. The engine treats all phase failures identically.

**Opportunity:** Wire `is_critical_path` to the replanning logic. When a critical-path phase fails after retries, halt the process. When a non-critical phase fails, mark it as skipped and adjust downstream phases. This is already implied by the dependency graph but never made explicit.

---

## Criticism Round 3: Extensibility, Evolution, and the Ecosystem Play

### C3.1 The Missing Abstraction: Process Packages as the Extension Unit

Pulling together C1.1 and C2.2: R4 introduces two independent extension mechanisms — tools (Python classes) and processes (YAML files). But the user's insight is correct: **these should be one extension unit.**

The programming language analogy:

| Concept | Language Analog | Current Loom | Proposed Loom |
|---------|----------------|-------------|---------------|
| Runtime | Python interpreter | Orchestrator engine | Same |
| Standard library | builtins, os, json | web_search, read_file, calculator | Same |
| User script | .py file | Process definition YAML | Same |
| Package | pip package (code + data + deps) | **Nothing** | **Process package** |
| Package manager | pip/PyPI | **Nothing** | `loom process install` (R4.14) |

Right now, R4.14's "marketplace" concept is "just YAML files." But if processes can bundle tools, templates, and verification rules, the marketplace becomes a real package ecosystem. The download unit is a directory, not a file. The install process is "copy directory + register tools."

This is the highest-leverage design decision in the whole plan. Getting the package format right enables an ecosystem. Getting it wrong means every serious process requires users to also manually install Python tool packages.

**Recommendation:** Define the process package format now, even if Batch A only implements single-file loading. The format should be:

```
<process-name>/
├── process.yaml          # Required: the process definition
├── tools/                # Optional: Python Tool subclasses
├── templates/            # Optional: prompt fragments
├── examples/             # Optional: few-shot examples
└── README.md             # Optional: human documentation
```

The `ProcessLoader` in Batch A can initially only support `process.yaml` (single file). Batch D adds directory support and tool auto-registration. But the format is defined upfront so early adopters structure their processes correctly.

### C3.2 Tool Guidance vs. Tool Requirements

R4 is explicit: "Does NOT restrict tools — all registered tools remain available." The process's `tool_guidance` is advisory, not prescriptive.

This is the right default, but there are cases where restriction matters:

- **Security:** A process running in a sandboxed environment shouldn't have access to `shell_execute`.
- **Cost control:** A "quick research" process shouldn't use the `spreadsheet` tool (which might trigger expensive model calls for layout decisions).
- **Correctness:** A financial analysis process where the user wants all math in the `calculator` tool, not freeform text arithmetic.

**Opportunity:** Add optional `tools.required` and `tools.excluded` fields:

```yaml
tools:
  guidance: |
    Prefer web_search for market data...
  required: [web_search, calculator, spreadsheet]  # Must be available or error
  excluded: [shell_execute]                         # Remove from tool schemas
```

`required` is a pre-flight check: if a required tool isn't registered, fail fast with a clear message. `excluded` removes tools from the schema sent to the model. `guidance` remains advisory. This is a clean separation of concerns.

### C3.3 Process Lifecycle Hooks

R4 describes processes as static definitions loaded at startup. But real workflows have lifecycle needs:

- **Before planning:** "Check that API keys for data sources are configured" (pre-flight)
- **After planning:** "Show the plan to the user and get approval before executing" (gate)
- **Between phases:** "Export intermediate deliverables to a shared drive" (side-effect)
- **After completion:** "Generate a summary email with all deliverables attached" (post-processing)

The engine's event system (`_emit(TASK_REPLANNING, ...)`) already has the hooks infrastructure. Process definitions should be able to declare lifecycle handlers:

```yaml
hooks:
  before_plan:
    - check: "Environment variable OPENAI_API_KEY is set"
      action: warn
  after_complete:
    - action: "Generate executive summary from all deliverables"
      model_tier: 2
```

This is lower priority than the core architecture but worth designing the extension point now.

### C3.4 The Cowork Gap Is Bigger Than D4 Suggests

Batch D4 says "Cowork mode process awareness (system prompt from persona) — ~30 lines." This undersells the integration challenge.

In cowork mode, the user has a conversation. They might say "analyze AAPL for me." If a process is loaded, the cowork session should:

1. Recognize this as a process-appropriate task
2. Use the process's persona in the system prompt (D4 covers this)
3. When the user says "delegate this" or the model decides to use `delegate_task`, pass the process to the orchestrator
4. Make process-specific memory types available for conversation recall
5. Let the user ask "what phase are we on?" and get a meaningful answer

Items 3-5 aren't covered. In particular, the handoff from cowork → orchestrator needs to carry the process definition, and the orchestrator's results need to flow back with process-aware context.

**Opportunity:** Define the cowork-process contract explicitly:

- `CoworkSession.__init__` accepts an optional `ProcessDefinition`
- The system prompt includes the persona (D4) AND tool guidance AND workspace analysis guidance
- `delegate_task` passes the process to the orchestrator
- Process memory types are included in the conversation recall tool's system context

### C3.5 No Telemetry or Process Performance Tracking

When a user runs the same process multiple times, there's no way to know:

- Which phases consistently fail or need retries?
- Which phases take the most tokens (cost)?
- Which verification rules fire most often?
- How does the process perform across different model tiers?

This data is essential for process authors to improve their definitions. It's also essential for Loom to recommend model tier adjustments.

**Opportunity:** The existing task/subtask result storage in SQLite should tag results with the process name and phase ID. A `loom process stats marketing-strategy` command could then show:

```
Phase               Avg Tokens  Fail Rate  Avg Retries
market-sizing           4,200      5%         0.1
competitive-analysis    6,800     15%         0.3    ← consider higher model tier
positioning             8,100      8%         0.2
channel-strategy        3,200      2%         0.0
campaign-plan           5,500     12%         0.2
```

This is just a query over existing data — minimal new code, high value.

### C3.6 Process Definition Testing Is Under-Specified

R4.12 says "Process definitions are data — validate schema only." But schema validation only catches structural errors (missing required fields, wrong types). It doesn't catch:

- **Dependency cycles:** Phase A depends on B, B depends on A
- **Dangling dependencies:** Phase A depends on "market-sizing" but no phase has that ID
- **Model tier conflicts:** A Tier 1 model assigned to a synthesis task that clearly needs Tier 3
- **Deliverable conflicts:** Two phases declare the same deliverable filename
- **Empty phases:** A phase with no description (the planner will hallucinate)

**Opportunity:** `ProcessLoader._parse()` should run validation beyond schema:

```python
def _validate(self, defn: ProcessDefinition) -> list[str]:
    errors = []
    # Check dependency graph is a DAG
    # Check all depends_on references exist
    # Check no duplicate phase IDs
    # Check no duplicate deliverable filenames
    # Warn if synthesis phases aren't model_tier >= 2
    return errors
```

This is cheap to implement and catches errors at load time rather than mid-execution.

### C3.7 The Biggest Missed Opportunity: Process Definitions as Composable Pipelines

R4 treats a process as a flat list of phases with dependencies. But the most powerful real-world workflows are *recursive* — a phase in one process is itself a sub-process:

- Marketing strategy's "competitive analysis" phase is actually the entire `competitive-intel.yaml` process.
- Due diligence's "financial analysis" phase is the entire `investment-analysis.yaml` process.
- A consulting engagement's workstreams are each mini-processes.

This is the process-calls-process composition from C1.4, but elevated to a core design principle. If process definitions can reference other processes as phases, you get:

1. **Reusability without duplication** — write `competitive-intel` once, use it in 5 processes
2. **Depth without complexity** — each process stays under 100 lines
3. **Independent versioning** — update `competitive-intel` and all parent processes benefit
4. **The marketplace becomes modular** — install building-block processes, compose them

This is how real programming languages work: you don't write everything in `main()`, you compose functions.

**Recommendation:** Add a `process` field to phase templates:

```yaml
phases:
  - id: competitive-analysis
    process: competitive-intel    # Delegate to another process definition
    depends_on: []
    # Description, verification, etc. come from the referenced process
```

The engine sees this phase, loads the referenced process, runs it as a sub-orchestration, and collects its deliverables as the phase output. The parent process's dependency graph treats the sub-process as an atomic phase.

This is the single feature that would transform Loom from "a tool that runs workflows" to "a platform that composes intelligence." It's ambitious — but the engine already supports nested task orchestration via `delegate_task`. This just makes it declarative.

### C3.8 Summary of Recommendations by Priority

| Priority | Issue | Fix | Batch |
|----------|-------|-----|-------|
| **P0** | Process package format (C3.1) | Define format now, implement single-file in A, directory in D | A + D |
| **P0** | Deliverables verification (C2.5) | Add to DeterministicVerifier | A |
| **P0** | Dependency graph validation (C3.6) | Add to ProcessLoader._parse() | A |
| **P1** | Verification rule types (C1.2) | Add `type: regex\|llm` field | A |
| **P1** | Phase strictness (C1.3) | Add `phase_mode` field | A |
| **P1** | Cowork integration (C3.4) | Expand D4 scope | D |
| **P1** | Workspace-scoped default process (C2.1) | Workspace loom.toml `[process]` | A |
| **P2** | Tool bundling (C1.1) | Directory-based processes with tools/ | D |
| **P2** | Process composition (C3.7) | `process:` field on phase templates | D |
| **P2** | Tool requirements/exclusions (C3.2) | `tools.required` / `tools.excluded` | A |
| **P3** | Process includes (C1.4) | `includes:` for shared fragments | Future |
| **P3** | Lifecycle hooks (C3.3) | `hooks:` section | Future |
| **P3** | Performance telemetry (C3.5) | Tag results with process/phase | D |
| **P3** | Schema versioning (C2.3) | Engine-aware `version` validation | D |
| **P3** | Memory cross-process (C2.4) | Shared memory type namespace | Future |
