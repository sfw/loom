# Output Subsystem Plan (2026-02-26)

## Goal
Design a modular output subsystem so Loom can reliably create, format, and deliver tuned artifacts such as slide decks, polished documents, and report packages, while fitting current process contracts, tool execution, and verification flows.

## Scope and Non-Goals
1. Scope: research + architecture + rollout plan only.
2. Scope: integrate with current process definitions, tools, verification gates, TUI, and API.
3. Non-goal: immediate code implementation in this plan.
4. Non-goal: replacing existing `write_file`/`document_write` workflows on day one.

## Product Research Summary (Current State)

### What already works well
1. Loom has strong orchestration and contract enforcement for deliverables.
   - `src/loom/prompts/assembler.py` injects exact deliverable filenames into executor prompts.
   - `src/loom/engine/runner.py` has canonical deliverable write-path policy guards.
   - `src/loom/engine/verification.py` enforces expected deliverable existence.
2. Loom supports multiple output-adjacent tools today.
   - `document_write` for structured Markdown (`src/loom/tools/document_write.py`).
   - `spreadsheet` for CSV artifacts (`src/loom/tools/spreadsheet.py`).
   - `timeline_visualizer` for markdown/mermaid/csv/json timeline artifacts.
   - `humanize_writing` for style-quality scoring and rewrite planning.
3. Loom can ingest and preview Office/PDF artifacts.
   - DOCX/PPTX text extraction (`src/loom/content_utils.py`, `src/loom/tools/file_ops.py`).
   - TUI file viewer supports `.docx`, `.pptx`, `.pdf` previews (`src/loom/tui/screens/file_viewer.py`).
4. Process packaging already preserves template assets.
   - Installer copies `templates/` and extra assets (`tests/test_installer.py`).

### Gaps that block a true output subsystem
1. No first-class output architecture.
   - Generation is fragmented across tools writing final files directly.
   - No shared intermediate representation for docs/decks.
2. Process templates are passive assets.
   - Package `templates/` are copied but not used by runtime rendering.
3. No renderer/plugin layer for stylized outputs.
   - No deterministic DOCX/PPTX writer path in core tools.
   - No reusable brand/theme/profile abstraction.
4. Verification is weak for stylized artifacts.
   - Deterministic syntax checks are limited to `.py/.json/.yaml`.
   - Artifact excerpting for semantic verification currently excludes Office formats.
5. Delivery surfaces are minimal.
   - API has no artifact listing/download endpoints.
   - No packaging/export flow for "deliverable bundles".

## Design Principles
1. Keep current workflows working; output subsystem is additive.
2. Separate composition from rendering.
3. Make style deterministic through profiles/templates, not only prompt wording.
4. Keep process contract as the control plane (not hardcoded domain behavior).
5. Verify both content and format before marking synthesis phases complete.
6. Prefer plugin interfaces so new output types do not require core rewrites.

## Proposed Architecture

### 1) Output Contract Layer (Process-Driven)
Add optional output contract blocks to process definitions.

Proposed contract extension (backward-compatible in schema v2):
1. `output.profiles`
   - Named style/render profiles (for example `board_deck`, `client_report`).
   - Includes `kind`, `format`, `template`, `style_tokens`, `quality_checks`.
2. `output.phase_mappings`
   - Maps phase IDs to output targets and profile names.
   - Keeps existing `deliverables` list as canonical filename enforcement.
3. `output.defaults`
   - Global defaults for tone, citation style, chart style, and file naming.

Example direction:
- `build-presentation` can declare `presentation.pptx` + profile `board_deck`.
- `quality-assurance` can require an executive brief profile for `.docx` or `.md`.

### 2) Composition Layer (Semantic Output Model)
Introduce an intermediate output model before final file rendering.

New module family (proposed):
1. `src/loom/output/models.py`
   - `OutputBundle`, `OutputSection`, `OutputSlide`, `OutputTable`, `OutputChartSpec`.
2. `src/loom/output/composer.py`
   - Builds `OutputBundle` from phase artifacts + process output profile.
3. `src/loom/output/normalizer.py`
   - Validates required sections, heading flow, references, and metadata.

Result: tools and models produce one normalized structure that multiple renderers can consume.

### 3) Renderer Plugin Layer
Create a pluggable rendering registry similar to tool discovery.

Proposed interface:
1. `OutputRenderer` base with `name`, `kinds`, `formats`, `render(bundle, profile, ctx)`.
2. `RendererRegistry` with auto-discovery.
3. Built-in renderers (phased):
   - `markdown_renderer` (MVP)
   - `html_renderer` (MVP)
   - `docx_renderer` (Phase 2)
   - `pptx_renderer` (Phase 2)
   - `pdf_renderer` (Phase 3, optional dependency)

Key behavior:
- Renderers are deterministic and template-driven.
- Output style tokens map to concrete typography/layout rules.
- Renderers emit structured metadata (section count, slide count, warnings).

### 4) Template and Theme Resolution
Use process package assets as runtime templates.

Resolution order:
1. `process.package_dir/templates/` when process is package-based.
2. Workspace-local overrides in `./.loom/templates/`.
3. Built-in defaults in `src/loom/output/templates/`.

This turns existing copied package templates into active render inputs.

### 5) Delivery Layer
Add delivery adapters for where output artifacts go.

Proposed adapters:
1. Workspace delivery (default): writes canonical files into task/run workspace.
2. Bundle delivery: creates zip bundles with manifest.
3. API delivery: task-scoped artifact listing/download surface.

Proposed API additions:
1. `GET /tasks/{task_id}/artifacts`
2. `GET /tasks/{task_id}/artifacts/{artifact_id}`
3. `GET /tasks/{task_id}/artifacts/{artifact_id}/manifest`

### 6) Output Verification Layer
Extend verification for stylized outputs.

Deterministic checks:
1. Render success and file existence for all declared output targets.
2. Office integrity checks for generated `.docx/.pptx` (open/read smoke checks).
3. Structural checks by kind.
   - Deck: minimum slide count, required slide titles, non-empty speaker notes when required.
   - Document: required sections/headings, citation marker policy.

LLM verification context upgrades:
1. Include extracted Office excerpts in artifact snapshots.
2. Include renderer metadata (slide count, warnings, template used).

### 7) Orchestration and Tooling Integration
Add one orchestration-friendly tool that fronts the subsystem.

Proposed tool:
1. `output_pipeline`
   - `compose`: build normalized `OutputBundle`.
   - `render`: materialize target format(s).
   - `package`: produce final bundle + manifest.
   - `deliver`: publish via delivery adapter.

Existing tools remain available for low-level edits, but synthesis phases should prefer `output_pipeline` when output profiles are declared.

## Integration Map (Planned Touchpoints)
1. `src/loom/processes/schema.py`
   - Parse and validate optional output contract blocks.
2. `src/loom/prompts/assembler.py`
   - Inject active output profile requirements into executor prompts.
3. `src/loom/engine/runner.py`
   - Enforce output-contract path policy similarly to current deliverable policy.
4. `src/loom/engine/verification.py`
   - Add output-format deterministic checks and Office excerpt support.
5. `src/loom/tools/__init__.py`
   - Register `output_pipeline` and output-related helpers.
6. `src/loom/tui/screens/file_viewer.py`
   - Improve deck/doc preview fidelity (for generated artifacts).
7. `src/loom/api/routes.py` + `src/loom/api/schemas.py`
   - Add artifact discovery/download endpoints.

## Rollout Plan

### R1: Contracts and MVP Renderer Path
1. Add process output contract parsing + validation.
2. Introduce `OutputBundle` model and Markdown/HTML renderers.
3. Add `output_pipeline` tool with `compose` and `render` operations.
4. Keep existing deliverables checks intact.

Exit criteria:
1. Process can declare output profile for synthesis phases.
2. Same content can render to at least two formats deterministically.
3. Existing processes run unchanged when no output contract is declared.

### R2: Office Output Generation (DOCX/PPTX)
1. Add DOCX and PPTX renderers.
2. Add template/theme resolution from process package `templates/`.
3. Add deterministic Office integrity verification checks.

Exit criteria:
1. `consulting-engagement`-style phase can produce real `.pptx` artifact.
2. Deck/doc generation uses selected profile/template.
3. Verification fails when Office outputs are malformed or structurally incomplete.

### R3: Delivery and API Surface
1. Add artifact manifest and bundle packaging.
2. Add artifact list/download API endpoints.
3. Surface output artifacts in TUI run views.

Exit criteria:
1. Completed tasks expose downloadable deliverables with metadata.
2. Artifact manifest links each output to phase/profile/template/version.

### R4: Quality and Personalization Loop
1. Integrate `humanize_writing` optionally into narrative-output QA.
2. Add style profile scoring for output bundles.
3. Add telemetry events for output rendering and profile usage.

Exit criteria:
1. Narrative outputs can be quality-gated against target score/profile.
2. Output telemetry supports auditing style/render decisions.

## Testing Strategy
1. Unit tests for output contract parsing and validation.
2. Unit tests for each renderer with fixture bundles.
3. Golden tests for deterministic render output structure.
4. Verification tests for new deterministic output checks.
5. Integration tests on built-in processes with declared output profiles.
6. API tests for artifact listing and download.

## Migration Strategy
1. Default behavior remains current tool-based file writes.
2. Output subsystem activates only when process output contract is declared or `output_pipeline` is called.
3. Existing `deliverables` remain canonical filename enforcement source.
4. Add migration docs and examples for process authors.

## Risks and Mitigations
1. Risk: added complexity in process schema.
   - Mitigation: optional blocks, strict validation, clear defaults.
2. Risk: renderer drift across formats.
   - Mitigation: normalize through one `OutputBundle` and golden tests per renderer.
3. Risk: Office generation fragility.
   - Mitigation: deterministic smoke checks and explicit fallback to Markdown outputs.
4. Risk: template incompatibility across packages.
   - Mitigation: template manifest versioning and renderer capability checks.
5. Risk: verification cost inflation.
   - Mitigation: keep deterministic checks first and bounded artifact excerpts.

## Success Metrics
1. At least one built-in process produces a true stylized deck artifact (`.pptx`) with profile-driven structure.
2. At least one built-in process produces a stylized document artifact (`.docx` or enhanced `.md`) from the same composition path.
3. Verification catches malformed or incomplete stylized outputs before phase completion.
4. Process package templates are actively consumed at runtime (not just copied as assets).
5. API and TUI expose produced artifacts as first-class deliverables.

## Recommended First Pilot
1. Pilot process: `consulting-engagement`.
2. Pilot target: replace `presentation-outline.md`-only endpoint with optional profile-driven `presentation.pptx` plus existing outline.
3. Reason: this process already encodes slide-ready constraints and is the clearest path to immediate user-visible value.
