# Workspace-First Desktop App Plan (2026-03-23)

## Objective
Build a modern desktop application for Loom that makes its strongest capabilities legible and usable through a workspace-first UI.

The app should make four user needs first-class:
1. Multiple workspaces exposed directly in the app, with fast switching.
2. Agent runs bundled under a workspace instead of floating as disconnected tasks.
3. Conversation and history presented as a beautiful, navigable activity journal.
4. Settings that expose main controls clearly, with advanced tuning hidden behind progressive disclosure.

## Executive Summary
Loom's core engine already appears stronger than its current product surface in several areas: orchestration, verification, persistence, replayability, process execution, and API/MCP integration. The main bottleneck is not the absence of capability, but the lack of a client experience that organizes those capabilities around the way users actually work.

This plan proposes a desktop app built as:
1. `Tauri 2` shell.
2. `React + TypeScript + Vite` frontend.
3. Bundled local `loomd` sidecar process as the default runtime.
4. Existing Python engine kept as the source of truth for execution, persistence, and workspace state.
5. A workspace-first information architecture that groups conversations, runs, approvals, artifacts, and settings under each workspace.

Docker should **not** be the primary runtime for the desktop product. It should remain an optional deployment and future sandbox target. The default should be an embedded `loomd` for the best local UX.

## Why This Plan
The current TUI is strong for power users, but it keeps Loom too close to an operator tool. Relative to products like OpenCode and OpenWork, the gap is primarily:
1. Discoverability of features.
2. UI legibility for long-running agent work.
3. Workspace-centric navigation.
4. Reuse/share/configuration surfaces.
5. Distribution and packaging confidence.

The goal is not to replace Loom's engine. The goal is to wrap it in a client architecture that exposes its strengths cleanly.

## Locked Product Decisions
1. The long-term primary GUI should be a desktop app with a web-rendered frontend, not a TUI evolution alone.
2. The engine remains Python and SQLite-backed; the frontend does not access SQLite directly.
3. `loomd` is embedded as the default desktop sidecar runtime.
4. Docker is an optional future execution or deployment mode, not the default desktop dependency.
5. Workspaces are the top-level navigation object in the app.
6. Conversations and runs are distinct objects, but both belong to a workspace.
7. Runs launched from conversation context should backlink to the originating conversation when possible.
8. Settings must use progressive disclosure: main controls visible by default, advanced controls hidden behind an explicit toggle.
9. The API boundary between UI and engine must be explicit and stable enough to support future non-desktop clients.

## Non-Goals (Initial Program)
1. Replacing the existing TUI in one step.
2. Rewriting orchestration, verification, or tool execution semantics.
3. Supporting full multi-user hosted collaboration in v1.
4. Building a browser-first SaaS product before local desktop is solid.
5. Exposing every internal tuning knob in the first UI release.
6. Frontend-driven state mutations that bypass engine validation/persistence.

## Current Baseline (Repo-Accurate)
1. Loom already ships an API server with task create/status/streaming and telemetry/config surfaces ([docs/agent-integration.md](/Users/sfw/Development/loom/docs/agent-integration.md#L42), [docs/2026-02-28-SYSTEM-TECHNICAL-DESIGN.md](/Users/sfw/Development/loom/docs/2026-02-28-SYSTEM-TECHNICAL-DESIGN.md#L598)).
2. Cowork sessions, turn history, transcript replay, and UI state persistence already exist in SQLite-backed systems ([README.md](/Users/sfw/Development/loom/README.md#L59), [docs/2026-02-28-SYSTEM-TECHNICAL-DESIGN.md](/Users/sfw/Development/loom/docs/2026-02-28-SYSTEM-TECHNICAL-DESIGN.md#L595)).
3. The current primary interactive UX is TUI-first, with panels, slash commands, approvals, and process-run tabs ([README.md](/Users/sfw/Development/loom/README.md#L37), [docs/2026-02-28-SYSTEM-TECHNICAL-DESIGN.md](/Users/sfw/Development/loom/docs/2026-02-28-SYSTEM-TECHNICAL-DESIGN.md#L641)).
4. There is already a prior web cowork plan, but not a full workspace-first desktop product plan ([planning/refactors/2026-02-24-COWORK-WEB-FRONTEND-IMPLEMENTATION-PLAN.md](/Users/sfw/Development/loom/planning/refactors/2026-02-24-COWORK-WEB-FRONTEND-IMPLEMENTATION-PLAN.md)).
5. There is no dedicated desktop shell, packaged `loomd`, or workspace-centric app surface in the repo today.

## Product UX Contract

### 1) Workspace-First Navigation
1. The left rail lists workspaces, not sessions.
2. Switching workspaces is fast, searchable, and remembers the last-open surface within that workspace.
3. Each workspace has a home view showing recent conversations, active runs, approvals, artifacts, and configuration state.
4. Local and remote workspaces should be representable with consistent UI affordances, even if remote support expands later.

### 2) Runs Bundled Under Workspace
1. Runs are visible under the selected workspace.
2. Active runs are pinned above historical runs.
3. Runs show concise status metadata: title, phase, elapsed time, changed files count, verification outcome.
4. A run detail view is richer than a transcript and includes progress, outputs, evidence, approvals, and artifacts.

### 3) Conversation as Activity Journal
1. Conversation view is not a flat chat log.
2. Tool activity, diffs, milestones, approvals, and run launches appear as structured cards or sections.
3. History is searchable and scannable.
4. Long threads remain performant through virtualized rendering and staged hydration.
5. Users can jump from messages to runs, files, artifacts, and changed outputs.

### 4) Settings with Progressive Disclosure
1. Main settings are simple and plain-English.
2. Advanced tuning is hidden behind an explicit toggle.
3. Settings support global defaults and workspace overrides.
4. Risky or highly technical settings are not shown in the default view.

## Recommended Tech Stack

### Desktop Shell
1. `Tauri 2`

Why:
1. Native desktop packaging and menus.
2. Lower memory footprint than Electron.
3. Good fit for launching and supervising a local sidecar process.
4. Supports file dialogs, notifications, deep links, updater flows, and OS integration.

### Frontend
1. `React`
2. `TypeScript`
3. `Vite`
4. `TanStack Router`
5. `TanStack Query`
6. `Zustand`
7. `TanStack Virtual`
8. `CodeMirror` for text/code viewers and editor-like surfaces where needed

Why:
1. React has the broadest ecosystem for sophisticated desktop-grade information architecture.
2. Query + Router provide strong data and navigation patterns for a multi-pane app.
3. Zustand is sufficient for app-local state without over-architecting.
4. Virtualization will be essential for long conversations, histories, and run timelines.

### Styling and Design System
1. CSS variables as the token layer.
2. Utility-first layout tooling is acceptable, but only on top of a real component system.
3. Shared design primitives should be extracted early: rails, panes, cards, timelines, inspectors, status badges, activity rows, permission modals, settings forms.

Recommendation:
1. Use `Tailwind` for layout speed if desired.
2. Do **not** let the app become a pile of ad hoc utility classes without tokens or reusable primitives.

### Backend / Engine
1. Existing Python engine remains the source of truth.
2. Existing FastAPI/SSE approach remains the base transport.
3. `loomd` becomes a stable local daemon entrypoint oriented to client apps.

### Transport
1. `HTTP + SSE` by default.
2. Add WebSocket only where bidirectional streaming materially simplifies UX.

Rationale:
1. Loom already fits SSE/event-streaming well.
2. SSE is operationally simpler and sufficient for most run/conversation updates.

## Monorepo Strategy
This program should stay in the existing Loom monorepo.

Why:
1. The desktop app is a new client surface for the same engine, not a separate product.
2. API, schema, event, and UX changes will often need to land together.
3. Packaging `loomd` with the desktop shell is easier when the engine and client live in one repository.
4. A monorepo keeps docs, plans, tests, and release work aligned.

### Recommended Repo Layout
Proposed structure:
1. `<repo-root>/src/loom/`
2. `<repo-root>/apps/desktop/`
3. `<repo-root>/packages/ts-sdk/` (optional, if a generated or hand-maintained TS client is created)
4. `<repo-root>/planning/refactors/`
5. Existing `tests/` retained for Python/backend coverage

Responsibilities:
1. `src/loom/`
2. Python engine, API, persistence, migrations, tooling, `loomd`, CLI/TUI
3. `apps/desktop/`
4. Tauri shell plus web frontend for the desktop app
5. `packages/ts-sdk/`
6. Shared typed client for the desktop UI and any future web/remote clients

### Frontend Placement Decision
Default recommendation:
1. Put the Tauri shell and React app under the same app tree: `apps/desktop/`

Suggested shape:
1. `<repo-root>/apps/desktop/src-tauri/`
2. `<repo-root>/apps/desktop/src/`
3. `<repo-root>/apps/desktop/package.json`

Why:
1. It keeps the desktop product cohesive.
2. It reduces overhead versus splitting shell and frontend too early.
3. It still leaves room for extracting shared TS packages later if a second client appears.

Alternative:
1. Split shell and UI into `apps/desktop/` and `apps/desktop-ui/`

Use that only if:
1. We know a browser-hosted or separate remote client is coming very soon.
2. We need the desktop web frontend to run independently from the shell as a first-order development target.

### JS Tooling Boundary
Recommended JS workspace tooling:
1. `pnpm` workspace at repo root
2. Root `package.json` for JS orchestration only
3. App-local package manifests under `apps/desktop/` and `packages/ts-sdk/`

Recommended root-level additions:
1. `pnpm-workspace.yaml`
2. Root `package.json`
3. Shared TS config/base config files if needed

The Python toolchain should remain authoritative for the engine:
1. `uv` / existing Python workflow for backend development
2. `pnpm` for desktop/frontend development

Do not attempt to force the whole repo under one universal build tool.

### Build and Dev Command Strategy
Recommended command model:
1. Python engine commands stay under `uv run ...`
2. Desktop/frontend commands live under `pnpm --filter desktop ...`
3. Root convenience commands can orchestrate common workflows, but should call the native tool for each layer

Examples:
1. `uv run loom serve`
2. `pnpm --filter desktop dev`
3. `pnpm --filter desktop build`
4. Root helper commands like `pnpm dev:desktop` may wrap the above for convenience

### Release Boundary
Monorepo release responsibilities:
1. Python package/versioning remains tied to Loom engine release semantics.
2. Desktop app gets its own packaging/build pipeline.
3. If `packages/ts-sdk/` exists, it should version in lockstep with compatible engine API changes unless there is a strong reason to publish it independently.

## Runtime Architecture

### Default Runtime: Embedded `loomd`
The desktop app should launch a bundled `loomd` sidecar on startup and supervise its lifecycle.

Responsibilities of the desktop shell:
1. Locate and launch `loomd`.
2. Wait for health readiness.
3. Pass workspace/runtime/config paths.
4. Surface health/version/startup failures to the user.
5. Restart or prompt on crash.
6. Shut down cleanly when the app exits.

Responsibilities of `loomd`:
1. Own database lifecycle and migrations.
2. Own task, cowork, run, and workspace persistence.
3. Expose client-safe HTTP/SSE endpoints.
4. Perform all filesystem and tool mutations.
5. Remain the single source of truth for settings and runtime status.

### Docker Positioning
Docker should remain:
1. A headless deployment target.
2. A future sandbox or isolated execution option.
3. A possible remote-host runtime for teams.

Docker should **not** be required for:
1. App install.
2. App startup.
3. Default local workspace use.

## Information Architecture

### Global App Shell
1. Left rail: workspaces.
2. Secondary column: workspace-local navigation.
3. Main pane: selected conversation, run, or workspace overview.
4. Right inspector: approvals, artifacts, context, metadata, or settings details.
5. Top bar: workspace identity, model/runtime status, quick actions, search, and command entry.

### Workspace Navigation Model
For each workspace:
1. Overview
2. Conversations
3. Runs
4. Files/Artifacts
5. Extensions
6. Settings

### Conversation Model
Conversation surfaces should support:
1. Message stream
2. Structured tool activity
3. Inline run cards
4. Search
5. History jump
6. Attachments/artifacts
7. Follow-up and approvals

### Run Model
Run surfaces should support:
1. Status summary
2. DAG/stage/timeline visualization
3. Subtask progress
4. Evidence and verification outcomes
5. Generated outputs
6. Changed files
7. Remediation/retry history
8. Jump back to parent conversation if applicable

## API and Engine Work Required
The existing API is task-oriented. A workspace-first client will need a broader application API.

### New or Expanded API Domains
1. Workspace registry and metadata
2. Workspace-local conversations index
3. Workspace-local runs index
4. Conversation detail and history paging
5. Run detail and event timeline
6. Unified approvals feed
7. Artifact/file previews
8. Settings read/write
9. Runtime health/version/diagnostics
10. Extension/process/MCP inventory for workspace surfaces

### Recommended API Shape

#### Workspaces
1. `GET /workspaces`
2. `POST /workspaces`
3. `PATCH /workspaces/{id}`
4. `DELETE /workspaces/{id}`
5. `GET /workspaces/{id}/overview`

#### Conversations
1. `GET /workspaces/{id}/conversations`
2. `POST /workspaces/{id}/conversations`
3. `GET /conversations/{id}`
4. `GET /conversations/{id}/messages`
5. `POST /conversations/{id}/message`
6. `GET /conversations/{id}/stream`

#### Runs
1. `GET /workspaces/{id}/runs`
2. `POST /workspaces/{id}/runs`
3. `GET /runs/{id}`
4. `GET /runs/{id}/timeline`
5. `GET /runs/{id}/artifacts`
6. `POST /runs/{id}/cancel`
7. `POST /runs/{id}/message`
8. `GET /runs/{id}/stream`

#### Settings
1. `GET /settings`
2. `PATCH /settings`
3. `GET /workspaces/{id}/settings`
4. `PATCH /workspaces/{id}/settings`

#### Approvals and Notifications
1. `GET /approvals`
2. `POST /approvals/{id}/reply`
3. `GET /notifications/stream`

### Data Model Implications
Likely schema changes:
1. Workspace registry table and/or richer workspace metadata.
2. UI preference state for workspace ordering, recents, and panel preferences.
3. Conversation-to-run linkage metadata.
4. Possibly denormalized overview/index tables for fast workspace home hydration.

If schema changes are required, follow the repo's migration-first policy:
1. Update canonical schema files.
2. Add migration steps and registry entries.
3. Add migration tests.
4. Update docs/changelog.

## Frontend State Model

### Server State
Use `TanStack Query` for:
1. Workspaces list
2. Workspace overview
3. Conversation lists and detail
4. Runs list and detail
5. Settings
6. Extensions/process/MCP inventories

### Local UI State
Use `Zustand` for:
1. Selected workspace
2. Active pane/tab
3. Right-sidebar mode
4. Local draft state
5. Expanded/collapsed timeline nodes
6. UI-only preferences not worth round-tripping immediately

### Streaming State
Use dedicated streaming adapters for:
1. Active conversation text/tool updates
2. Active run progress/timeline events
3. Global notification and approval updates

## Design and Performance Requirements
1. Virtualize long message and run lists.
2. Hydrate history incrementally.
3. Keep the main reading column calm and highly legible.
4. Use structured cards and timelines instead of dumping JSON or logs into the main feed.
5. Preserve the ability to inspect raw detail in secondary panels.
6. Support desktop widths first, but keep layouts responsive enough for smaller windows.

## Workstreams

### W1: `loomd` Daemonization and Client Runtime Contract
Files:
1. `<repo-root>/src/loom/cli/...`
2. `<repo-root>/src/loom/api/server.py`
3. `<repo-root>/src/loom/api/engine.py`
4. New daemon-oriented entrypoint(s)

Deliverables:
1. Stable `loomd` process entrypoint.
2. Health/version/readiness endpoints suitable for a desktop shell.
3. Clean startup/shutdown lifecycle.
4. Clear runtime directories and config path rules.

Acceptance:
1. Desktop shell can launch `loomd`, wait for readiness, and recover from failures cleanly.

### W2: Workspace Registry and Overview API
Files:
1. `<repo-root>/src/loom/api/routes.py` or split route modules
2. `<repo-root>/src/loom/state/...`
3. Potential schema/migration files

Deliverables:
1. Workspace listing/create/update/remove flows.
2. Workspace overview endpoint.
3. Recent activity and active run summaries per workspace.

Acceptance:
1. UI can load a workspace-centric home screen without scraping conversation/task tables ad hoc.

### W3: Conversation Index and Journal API
Files:
1. `<repo-root>/src/loom/cowork/...`
2. `<repo-root>/src/loom/api/...`
3. `<repo-root>/src/loom/state/...`

Deliverables:
1. Workspace-scoped conversation listing.
2. History paging/search primitives.
3. Streaming API suitable for journal-style rendering.
4. Rich event typing for tool cards, milestones, and run-launch linkage.

Acceptance:
1. UI can render conversations as structured activity journals with fast initial load and paged backfill.

### W4: Run Index and Run Detail API
Files:
1. `<repo-root>/src/loom/api/...`
2. `<repo-root>/src/loom/engine/...`
3. `<repo-root>/src/loom/events/...`

Deliverables:
1. Workspace-scoped run listing.
2. Rich run detail payloads.
3. Timeline/status/evidence/artifact summaries.
4. Run-to-conversation linkage where applicable.

Acceptance:
1. UI can show a dedicated run detail view that is more useful than raw event logs.

### W5: Settings API Unification
Files:
1. `<repo-root>/src/loom/config.py`
2. `<repo-root>/src/loom/api/...`
3. Potential workspace setting persistence files

Deliverables:
1. Read/write API for main settings.
2. Explicit grouping of basic vs advanced settings.
3. Global defaults plus workspace override model.

Acceptance:
1. UI can expose a polished settings surface without writing config files directly.

### W6: Desktop Shell Foundation
Files (new, proposed):
1. `<repo-root>/apps/desktop/`

Deliverables:
1. `Tauri 2` app scaffold.
2. Sidecar lifecycle management.
3. Auto-update strategy stub.
4. Native dialogs and OS integration.

Acceptance:
1. App boots locally, launches `loomd`, and renders a minimal shell.

### W7: Frontend Foundation
Files (new, proposed):
1. `<repo-root>/apps/desktop-ui/` or colocated frontend under desktop app

Deliverables:
1. React/Vite app shell.
2. Routing, query client, local store, design tokens.
3. Global command/search surface.
4. Empty, loading, and error system for desktop-grade resilience.

Acceptance:
1. App framework is stable enough for feature work without repeated re-architecture.

### W8: Workspace Shell UX
Deliverables:
1. Left workspace rail.
2. Workspace overview.
3. Secondary navigation for conversations/runs/settings.
4. Workspace switching and recents.

Acceptance:
1. Users can move cleanly between multiple workspaces without losing context.

### W9: Conversation Journal UX
Deliverables:
1. Structured message stream.
2. Tool cards, milestone blocks, and inline run cards.
3. Search and history jump.
4. Artifact/file jump surfaces.

Acceptance:
1. A long conversation remains understandable and visually calm.

### W10: Run Detail UX
Deliverables:
1. Active run timeline.
2. Subtask detail and verification summary.
3. Outputs/artifacts/changed files.
4. Approval and remediation surfaces.

Acceptance:
1. Users can understand exactly what an autonomous run did and where to intervene.

### W11: Settings UX
Deliverables:
1. Basic settings panels for common tasks.
2. Hidden advanced mode.
3. Workspace overrides.
4. Clear save/apply/reload feedback.

Acceptance:
1. Main setup tasks are simple, while power-user tuning remains available without cluttering default flows.

### W12: Packaging and Distribution
Deliverables:
1. Bundled `loomd` inside desktop package.
2. Cross-platform packaging strategy.
3. Local dev workflow for frontend + sidecar.
4. Release validation checklist.

Acceptance:
1. A user can install and run the desktop app without separately setting up Python tooling for normal use.

## Suggested Delivery Phases

### Phase 0: Foundation and Contracts
1. Define `loomd` contract.
2. Define app IA and screen contracts.
3. Add missing API surfaces for workspaces, conversations, runs, settings.

### Phase 1: App Shell and Workspace Navigation
1. Desktop shell launches `loomd`.
2. Workspace rail and overview ship.
3. Basic conversations and runs indexes appear under workspace.

### Phase 2: Conversation and Run Surfaces
1. Journal-style conversation rendering.
2. Dedicated run detail/timeline UI.
3. Approvals and artifact inspector.

### Phase 3: Settings and Extension Surfaces
1. Basic/advanced settings split.
2. Workspace override UX.
3. Extension/process/MCP inventory surfaces.

### Phase 4: Polish, Packaging, and Distribution
1. Packaging hardening.
2. Crash recovery and update flows.
3. Performance tuning for long histories.
4. Design system cleanup.

## Testing Strategy

### Backend
1. API contract tests for workspaces, conversation indexes, runs, and settings.
2. Stream ordering tests for conversation and run event feeds.
3. Migration tests for any new workspace/UI metadata tables.
4. Sidecar startup/readiness and failure-mode tests.

### Frontend
1. Component tests for key structured surfaces.
2. Integration tests for workspace switching, conversation streaming, and run detail loading.
3. E2E desktop tests for launch, workspace creation, run viewing, and settings save flows.

### Packaging
1. Startup on clean machines.
2. DB migration during app upgrade.
3. Sidecar crash and restart scenarios.
4. Missing/config-invalid model setup flows.

## Risks and Mitigations

### Risk 1: Frontend outruns API contracts
Mitigation:
1. Freeze route contracts early.
2. Add typed client generation or shared schema definitions where practical.

### Risk 2: Bundled Python sidecar packaging complexity
Mitigation:
1. Prototype packaging early.
2. Treat sidecar bundling as a first-order engineering track, not release-end glue work.

### Risk 3: Long-thread performance degrades
Mitigation:
1. Virtualization from day one.
2. Incremental hydration and explicit event summarization.

### Risk 4: Settings UX becomes a mirror of raw config
Mitigation:
1. Design around user jobs, not config keys.
2. Keep advanced mode truly separate.

### Risk 5: App becomes a second product with divergent semantics
Mitigation:
1. Keep engine as source of truth.
2. Avoid frontend-only mutations or shadow logic.

## Open Questions
1. How much of current TUI cowork event shape can be reused directly versus normalized for the app?
2. Do we want generated TS clients from the API schema as part of phase 0?
3. Should workspace registry live in Loom's main DB or in a separate app metadata layer owned by `loomd`?
4. What minimum installer footprint is acceptable for bundling Python and model-adjacent dependencies?
5. Do we need `packages/ts-sdk/` immediately, or can phase 0 start with typed app-local API clients and extract later?

## Immediate Next Steps
1. Approve the stack and runtime decisions in this document.
2. Write a follow-on technical design for `loomd` packaging and lifecycle.
3. Write an API contract plan for workspaces, conversations, runs, and settings.
4. Produce wireframes for the workspace shell, workspace overview, conversation journal, run detail view, settings basic view, and settings advanced view.
5. Stand up a minimal `Tauri + React` shell that can launch a placeholder local sidecar and render a static workspace layout.

## Success Criteria
This program is successful when:
1. Loom feels like a workspace operating environment rather than a raw agent console.
2. Users can move between multiple workspaces without losing context.
3. Agent runs feel native to workspace history rather than bolted-on background tasks.
4. Long conversations become easier to read, search, and trust.
5. Main configuration tasks become easy without taking power away from advanced users.
6. Desktop installation and startup feel product-grade rather than source-code-grade.
