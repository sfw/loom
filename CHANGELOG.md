# Changelog

All notable changes to Loom are documented in this file.

This changelog is generated directly from git commit history (non-merge commits) to keep entries accurate and auditable.

## [Unreleased]

- 2026-03-07 `7d66e79` Fix regressions from sealed artifact policy refactor

## [0.2.0] - 2026-03-08

- Normalize package version authority to `pyproject.toml` and bump to `0.2.0`
- Replace hardcoded runtime version literals with shared `loom.__version__` usage
- Add version consistency guardrails (`scripts/check_version_consistency.py`, CI check, and tests)
- Update release-facing docs/examples for `0.2.0` (`README.md`, `INSTALL.md`, `docs/tutorial.html`)

## [History]

### 2026-02-13
- `675c4fe` Initial commit
- `ae23f7c` Add project overview spec (00-PROJECT-OVERVIEW.md)
- `17acd74` Add project structure spec (01-PROJECT-STRUCTURE.md)
- `5f12e64` Add orchestrator loop spec (02-ORCHESTRATOR-LOOP.md)
- `3286a68` Add task state/memory and model router specs (03, 04)
- `a051fb0` Add tool system spec (05-TOOL-SYSTEM.md)
- `03bb276` Add verification gates spec (06-VERIFICATION-GATES.md)
- `bb5a7df` Add API server spec (07-API-SERVER.md)
- `cb6ed66` Add event system spec (08-EVENT-SYSTEM.md)
- `89b363d` Add TUI client and agent integration specs (09, 10)
- `f9a0cf3` Add workspace and file management spec (11-WORKSPACE-FILES.md)
- `ee7d15a` Add prompt architecture and error recovery specs (12, 13)
- `0fd04f3` Add human-in-the-loop and learning system specs (14, 15)
- `c7f4cc2` Scaffold project: pyproject.toml, CLI, config, CI, test infra
- `b562cf1` Implement task state and memory layer (Spec 03)
- `e433dd3` Implement prompt template system (Spec 12)
- `53115f1` Implement model router with Ollama and OpenAI providers (Spec 04)
- `90b2a95` Implement tool system with registry, file ops, shell, search (Spec 05)
- `1741fae` Implement workspace and file management (Spec 11)
- `f21b625` Implement orchestrator loop with event bus and scheduler (Spec 02)
- `b1a670f` Implement API server with FastAPI endpoints and SSE streaming (Spec 07)
- `b1f6e11` Add README, installation guide, and web-based tutorial
- `39b008c` Fix runtime bugs: /config tier attribute and engine scratch_dir path

### 2026-02-14
- `41845e8` Wire Phase 2 integrations: verification gates, changelog, memory extraction, event persistence, re-planning
- `24da424` Update docs for Phase 2: verification gates, event persistence, re-planning
- `1b4b3b0` Implement confidence scoring, approval gates, retry escalation, and webhook delivery (Specs 08, 13, 14)
- `f97a5a5` Implement TUI client, MCP server, and learning system (Specs 09, 10, 15)
- `09794d3` Fix CI: use --extra dev for optional dependency installation
- `70ae27f` Update README and INSTALL docs for all implemented features
- `f269e31` Update tutorial with TUI, approval gates, MCP, and learning sections
- `5d1f29c` Add full end-to-end integration test and agent connection documentation
- `5d8bc94` Add SubtaskRunner class for subtask execution encapsulation
- `0d61a93` Refactor orchestrator: extract SubtaskRunner, add parallel dispatch
- `396f524` Update docs and add CHANGELOG for SubtaskRunner and parallel dispatch
- `c89ce04` Add implementation plan for bringing Loom up to snuff
- `8630e8d` Implement phases 1, 2, and 5: new tools, streaming, error intelligence
- `0597406` Implement phases 3, 4, and 6: TUI enhancements, smarter planning, conversation mode
- `b15311d` Refactor tool registration to plugin auto-discovery model
- `d7049c0` Update CHANGELOG with plugin auto-discovery entry
- `f4dc27d` Add gap analysis: Loom vs Claude Code cowork model
- `84b59fa` Implement cowork mode: conversation-first interactive execution
- `620437d` Update CHANGELOG and README with cowork mode, new tools, and Anthropic provider
- `926be5e` Add web_search tool using DuckDuckGo (no API key required)
- `19cf277` Add per-tool-call approval system for cowork mode
- `c0d9c9d` Add streaming, task tracker, PDF/image support; close all gaps
- `cf9cb55` Rewrite TUI to use CoworkSession directly (no server required)
- `112c73f` Update all docs and tutorials to reflect cowork features
- `4a14ba3` Fix all ruff lint errors: unused imports, line length, naming conventions
- `83c6343` Move PLUGIN_PLAN.md to planning/refactors/
- `2b018d7` Add TUI refactor plan: multi-panel command center design
- `55f99fb` Refactor TUI into multi-panel command center with dark theme
- `ba37f64` Add comprehensive error handling across all system layers
- `6982e1c` Clarify that cowork mode works with any model provider
- `d5df627` Add conversation archive refactor plan for non-lossy cowork history
- `2a152be` Add three-layer context awareness strategy to conversation archive plan
- `64427b9` Add cowork-to-task delegation as core component of conversation archive plan
- `ba64a59` Implement non-lossy conversation history + task delegation for cowork mode
- `71bf5df` Add mid-session switching between cowork sessions
- `0bbfb49` Fix 52 bugs, security flaws, and edge cases from comprehensive code review
- `d3cf583` Fix 23 bugs from second comprehensive code review
- `77d7f5b` Fix 11 bugs from rounds 3-5 code review + changelog + audit doc

### 2026-02-15
- `8b6d5e4` Add comprehensive plan for general-purpose Loom expansion
- `5a64836` Redesign expansion plan around declarative process definitions
- `0d2c4d5` Add three rounds of critical review to process definition architecture
- `5ca225d` Fix 15 lint errors: line length, unused imports, f-string and alias issues
- `d34e4fd` Implement process definition plugin architecture with extensible tool system
- `7f17236` Clean up built-in process definition YAML formatting
- `6ada59a` Fix correctness bugs: type safety, double stat, unused --process flag
- `6caa980` Harden safety: exponent limits, file size checks, error logging
- `28cae9b` Fix type: ignore comments in spreadsheet tool with assertions
- `9bfaa5e` Add tests for safety guards and edge cases
- `70ecb71` Remove type: ignore in cycle detection with proper type narrowing
- `70ed852` Minor YAML formatting cleanup in builtin process definitions
- `cd40f19` Update all documentation for process definition plugin architecture
- `1a2d639` Add process package installer with loom install/uninstall commands
- `a3dad14` Add security review gate to process package installer
- `1dc26b5` Round 1: Fix correctness bugs across verification, installer, schema, CLI
- `d04a7cf` Round 2: Safety and edge case hardening
- `20217aa` Round 3: API consistency fixes
- `93b8742` Round 4: Improve test coverage for verification, schema, and tool fixes
- `43403fb` Add Google Analytics process package
- `e1de19b` Fix lint errors: import ordering and collections.abc
- `ba2f17f` Add fuzzy matching, batch edits, and diff display to edit_file
- `a36c757` Code review fixes: bug fixes, TUI diff support, expanded tests
- `ea59164` Fix lint: line length violations in file_ops and tests
- `7e0eda0` Rewrite README for clarity, accuracy, and the human reader
- `bd5b344` Add multimodal content support: images, PDFs, thinking blocks
- `b144a87` Wire multimodal content through TUI, API, and terminal display
- `e66c3ff` Fix multimodal integration gaps: OpenAI provider, deserialization, events
- `2e07188` Harden multimodal content: fix data loss, injection, leaks, validation
- `0690100` Fix TUI rendering bugs: center modals, escape markup, style diffs
- `47f4353` Unify TUI as the single interactive interface; `loom` launches it by default
- `b7e4587` Update all docs and tutorial for unified TUI as default interface
- `0be941d` Harden TUI: graceful DB fallback, broader error handling, ask_user in followups
- `a7821ea` Add sad-path tests for TUI cowork integration
- `376b051` Add first-run setup wizard and `loom setup` command
- `fc59cd1` Move setup wizard into TUI as a modal screen
- `e1d92e5` Fix TUI setup bugs, harden session logic, update docs
- `3fdf077` Lint cleanup: remove unused imports and variables
- `2064c25` Fix CI lint failures: remove f-string prefix and sort imports
- `2be0121` Add process package authoring guide for AI and human authors
- `eeb27a7` Broaden README to reflect Loom as a general-purpose agent engine
- `9410b17` Expand default workspace analysis to scan for non-code documents
- `20ee5b9` Add Word (.docx) and PowerPoint (.pptx) document support
- `f866705` Fix lint error and update docs for office document support
- `98afef2` Update model names in README tagline to Kimi, Minimax, GLM
- `6e8b5d8` Replace remaining Qwen references with MiniMax in README
- `8e42288` Fix MiniMax model identifiers in README config examples
- `22a1c83` Use kimi-k2.5 as primary model in README config example

### 2026-02-16
- `da45203` Move PLAN.md into planning/refactors directory
- `6af21c4` Add automatic behavioral reflection to ALM (Spec 16)
- `13502e2` Add spec to drop regex from codebase in favor of structural alternatives
- `1830758` Rewrite DROP-REGEX spec to scope only to learning/reflection.py
- `8d3f0c6` Redesign ALM spec around task completion gap analysis
- `83ec647` Replace regex reflection engine with task completion gap analysis
- `832bf93` Add /learned command for reviewing and deleting learned patterns
- `f50de9e` Update docs and specs for ALM and /learned command
- `ab46d78` Update tutorial.html for ALM and /learned command
- `faef639` Add help and quit to command palette; update README intro
- `fc38c86` Redraft README intro sections: Claude cowork framing and MCP mention
- `489e694` Add foundation refactor plan from full codebase audit
- `f7be8ce` Add tree-sitter integration plan from design discussion
- `03490ce` Implement 6-phase foundation refactor across 22 files
- `71cca12` Fix linter errors: import ordering and line lengths
- `ceeaa59` Fix duplicate tool registration crash on TUI startup
- `e62c125` Rename loom.toml to loom.toml.example, gitignore loom.toml
- `2316bd6` Fix setup wizard not accepting keystrokes
- `edc7d2d` adding code review
- `9fbe157` Add code review plan from main branch
- `026b94f` tui fixes and review
- `4d488ad` Fix 14 code review issues (P0-P2) with tests
- `0894a4c` Add TUI review document from main branch
- `9e6a8cf` Fix 8 TUI review findings (P0-P2) from 2026-02-16-TUI-REVIEW
- `e1c694d` Update docs to reflect current codebase state
- `0e727be` Add tree-sitter integration for code analysis and structural editing
- `b80f32a` Fix review findings: byte/char offsets, nested defs, Rust impl, naming

### 2026-02-17
- `1f0f2ff` Update docs for tree-sitter integration and current stats

### 2026-02-16
- `a44cdcf` Fix API process isolation, reset TUI file panel, and add tree-sitter CI gate
- `770152d` Harden weak tests to reduce false positives
- `6404dd6` code-review docs and uv.lock
- `ab0de15` Harden TUI setup flow and add process/package refactor plan
- `39750b4` Propagate active process through TUI sessions and delegation
- `fdacb0b` Enforce process required tools in TUI and orchestrator
- `35eb106` Honor process.default for run and TUI launch
- `2256c7f` Add in-session /process controls and process status display
- `c72f542` Copy full process package assets with safe installer filtering
- `670f5b7` Enforce strict process phase blueprints in planning
- `d37618d` Fix package install docs for Google Analytics process
- `7c4894d` Add process controls to the TUI command palette
- `d4377f4` Reject process tasks with missing required tools at API create
- `3e1d2db` Document in-session process controls in README
- `3c5e99b` Apply process.default in API task creation path
- `4abb24c` Enforce process runtime semantics for critical and synthesis phases
- `8eb1ae4` Close process/package plan: isolated deps, collisions, docs
- `3226783` Harden web tools and scope process deliverable verification

### 2026-02-17
- `f174ac0` feat: add process testing framework and MCP tooling improvements
- `559ee89` fix: apply configured api keys in openai-compatible and ollama providers
- `334f03e` refactor: broaden cowork system prompt beyond coding-only framing
- `d7da827` Improve process orchestration UX, resiliency, and docs parity
- `37e968e` Add rich TUI file viewer renderers and learned-pattern cleanup
- `e273c04` Close file viewer modal on backdrop click
- `663ebb3` Implement MCP config management across CLI, TUI, and runtime
- `3fd3d8e` Refine MCP OAuth plan and fix TUI input/hint layout
- `c0dd66b` Improve process run live feedback and process-use UX
- `a66447e` Expand Ctrl+P palette coverage and command prefills
- `85d1efb` Harden process runs, TUI UX, and provider message compatibility
- `8d71f5d` Harden verifier JSON parsing for critical-path retries
- `e858f11` Harden OpenAI payloads against empty assistant content
- `92e4d4e` Add market-research process and harden TUI process UX
- `c815333` Replace hard truncation with semantic compaction

### 2026-02-18
- `1a9841e` Harden process verification stack and tighten delegate logging

### 2026-02-19
- `e1dc880` massive processing refactor. v2 of process definition.
- `84c5798` TUI hierarchy fixes
- `ba3c7a4` Fixing some process issues, TUI issues, and more.
- `bf26548` fixing a small bug in testing and reworked the process progress box in the TUI
- `028b4e5` small ui bug
- `21cfe1d` doc updates

### 2026-02-20
- `28b4caa` Fix TUI mouse interaction startup flag
- `ec0c240` Simplify MCP/auth UX and prune routing complexity

### 2026-02-22
- `4f42efe` Tune compactor budgeting and docs defaults
- `f1d80d2` Align CLI run process resolution with TUI flow
- `c883f64` Stabilize event persistence assertions in integration tests
- `2d28ae3` Harden replanning with version fencing and strict ID continuity

### 2026-02-23
- `1fe5202` docs: align model role split examples and revalidate dead-code plan
- `ccb07fb` feat: add artifact ingest handling and telemetry transparency controls
- `b92d1b6` Implement telemetry transparency events and default-on rollout
- `d88f5f2` Harden contradiction scan coverage gating
- `4fba048` Avoid hard import dependency for contradiction event type
- `d082fdc` Implement keyless research tool suite
- `a2d66d2` doc updates, process updates, retry architecutre changes
- `c177ba6` adding potential web frontend
- `1f7f7e9` Export evidence ledger CSV during wrap-up

### 2026-02-24
- `79295b3` Harden economic provider JSON payload parsing
- `9aa8371` Honor read_roots for spreadsheet read operations
- `63eaa9c` Add /run goal-file input support and docs
- `291437f` Add humanize_writing tool with deterministic scoring and tests
- `6e43efe` Make map-competition non-blocking for unverified competitor entries

### 2026-02-25
- `3318acd` Add keyless investment tool suite and FRED economic provider support
- `1d5524a` Add TUI input history navigation with resume restore
- `3b24715` Fix portfolio optimizer event-loop freeze in TUI runs
- `a0e495f` Add keyless market-signal tool suite
- `6914233` Fix synthesis deadlocks and add stalled-run safeguards
- `d2dd913` Fix deterministic synthesis contract responses
- `84e95f5` docs: align tutorial intro with harness positioning
- `0e22f88` Make missing-data remediation non-fatal in marketing strategy

### 2026-02-26
- `c5880f7` docs: use uv run loom in README commands
- `3ebb559` adding technical design doc
- `e088aa9` Implement reliability hardening and refresh technical design

### 2026-02-27
- `8fb9b59` docs: update README.md
- `551670d` docs: add README screenshot
- `0de3623` Implement resource-first auth lifecycle, migration tooling, and UX hardening
- `d133ef2` Harden process prompts and shell safety for segmentation runs
- `f3747a4` Improve cowork turn telemetry and render chat as Markdown
- `e98d3d5` Reduce startup and interaction latency via async MCP warmup
- `d1dfa4f` Add hybrid tool entrypoints and follow-up config/auth updates
- `338f427` Harden auth profile selection and tool inventory coverage

### 2026-02-28
- `dc3afe9` cowork: compact context via recall index and add turn telemetry
- `55156ee` Improve hybrid list_tools flow and enforce scoped schema lookup
- `b94f013` Fix cowork prompt JSON examples in f-string
- `c2e61ff` Repair dangling tool-call chains in cowork context
- `4536fb3` Add model-planned failure remediation with bounded metadata context
- `8ef408a` Add cowork tool-loop convergence guard and context-window fix
- `f1b3064` Add structured match and file counts to ripgrep_search
- `f23d72d` Implement chat history replay refactor for TUI session resume
- `3a743cf` Harden slash command hints with full scrollable catalog
- `e994e59` Fix duplicate chat panel rows during live turns
- `0883857` Implement cowork delegate progress UX and realtime TUI refresh
- `44db5b9` Fix lint line-length in delegate progress widget
- `1184a00` Harden /run launch liveness and in-place stage heartbeat UX
- `2edf67e` Harden process-run cancel flow and fix queued Ctrl+W hangs
- `b5df6ce` Move technical design doc to docs with updated date
- `5cda3c2` changing image
- `1a139bf` Add live GA4 API retrieval with auth to google-analytics package
- `1e59c87` Fix ga_live_api tests leaking global tool registrations
- `ab377f5` Move GA live API tests into package-local test directory

### 2026-03-01
- `c4630d8` Improve auth discovery scope and preserve selected mode in /auth
- `d6e05ef` tui: remove /process mode and simplify /run palette flows
- `38f57c9` mcp: add oauth token runtime and manager support
- `486a6b6` Harden MCP/Auth management and tab UX
- `c876a05` Add header activity indicator and refine active animation
- `0ad619a` Add rich /model details and new /models TUI command
- `1738235` Refine header activity indicator rendering behavior
- `17d9b20` Align outputs with replans via phase IDs; remove deprecated auth-path noise
- `48bda35` tui: align sidebar progress icons in dedicated column
- `9d1a0f4` Fix MCP stale alias reload and auth-check explicit scoping
- `e0d72f6` Implement hybrid cowork chat stop control and hardening
- `227c1e7` Implement cowork steering queue controls and execution fixes
- `fa6035e` Fix remote MCP OAuth and streamable MCP tool discovery
- `9c20a61` Enable clickable markdown links with URL tooltip in cowork chat
- `7c236df` auth: harden mcp oauth boundary and reuse compatible drafts
- `72bb236` tui: make auth manager sync explicit and refresh read-only
- `951db4c` Retheme cowork markdown to replace magenta defaults
- `e3e2ad4` auth: classify audit history separately from active orphan findings
- `c9fbfc1` auth: add /auth oauth lifecycle commands and tui actions

### 2026-03-02
- `241e47c` Complete auth cleanup phases: OAuth parity, TUI mode UX, and token handling
- `e8a2d6b` docs: sync endpoint count and current project stats
- `5dac6ba` changing image
- `e2dba38` tests: remove credential-style URI fixture
- `ae69a68` ci: keep live canary config out of uploaded artifacts
- `7416521` Improve tooling UX and provider-specific coding agent integrations
- `385c66a` Guard on_key screen lookup when no screen stack is mounted

### 2026-03-03
- `ed92c32` Add phase-level iteration loop runtime, gates, persistence, and docs
- `42b5e08` Improve process run controls, auth preflight, and workspace selection
- `a30398d` Fix chat scroll lock and tighten activity pane spacing
- `5c36ed6` feat(cowork): harden long-session context recall and indexing

### 2026-03-04
- `3456125` Harden ask_user runtime flow and execution-surface gating
- `ed2a6bc` Align delegate_task TUI approval default with /run
- `f3c68ca` docs: remove stale LOC remark from README
- `1ce7cca` adding trufflehog precommit hook

### 2026-03-05
- `0813474` Improve command palette, landing flow, and close-tab safety
- `031a631` Add loom2 hero image to README
- `3dd0b07` feat(validity): enforce claim-level synthesis gates and evidence lineage
- `6b72edf` Enforce evidence-backed edits for sealed verified artifacts
- `bfdd915` Enable ask_user interactive defaults
- `935e21d` Default process test execution to temp workspace

### 2026-03-06
- `ca97094` Refine ask_user UX and raise per-subtask cap default
- `b75d87a` Split placeholder prepass from confirm-or-prune resolution
- `fecb07a` Harden SQLite migrations and startup persistence semantics
- `8aaefc4` Implement telemetry coverage and auditability hardening
- `03a16be` adding refactor plans and updates to technical doc
- `c42ebbd` Relink migration diagnostics to event type constants
- `0d77f06` Fix verification guard wording and token redaction
- `d8df362` Harden fan-in output coordination and transactional publish flows
- `0d70627` Implement runtime telemetry verbosity modes and controls

### 2026-03-07
- `a7fef18` Refactor core engine modules into package-based subsystems
- `48ad5ec` refactor(tui): split app into package modules with css file
- `cc90b6a` refactor(cli): split __main__ into modular cli package
- `8ac0558` docs(plan): update main entrypoint split plan
- `0c86629` Split monolithic TUI tests into tests/tui modules
- `ab1647b` Split orchestrator tests into package and add legacy path guard
- `d8ab27d` Unify sealed artifact mutation policy and sanitize docs paths
