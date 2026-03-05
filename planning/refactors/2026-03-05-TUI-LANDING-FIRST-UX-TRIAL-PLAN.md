# TUI Landing-First UX Trial Plan (2026-03-05)

## Objective
Ship a trial UX where first entry is a branded, task-first landing composer when startup does **not** resume a prior session, while also delivering the other agreed UX refinements:
1. Conversation-first presentation (diagnostics moved out of primary chat flow).
2. Better empty-state onboarding.
3. Layout density improvements.
4. Stronger visual hierarchy/focus states.
5. Shortcut legend readability (`ctrl +` style labels).
6. Action feedback for key operations.
7. Less metadata noise and duplication.

This trial must be easy to revert.

## Locked Product Decisions
1. Landing appears only when startup is **not** resuming a session.
2. Add a user preference to always skip landing and open chat directly.
3. Use lowercase `ctrl +` labels instead of caret shorthand in user-facing shortcut surfaces.
4. When landing is shown, do not create a new session until first prompt submit.
5. `/new` opens directly to chat/workspace (no landing interstitial).
6. Footer remains visible on the landing surface.
7. Landing includes a stylized `loom` text logo above the main entry composer.
8. Landing displays current workspace path as persistent context (screenshot-style placement).

## Baseline (Repo-Accurate)
1. `src/loom/tui/app.py` always composes the 3-pane workspace layout (`Sidebar + Tabs + Input/Footer`) and always mounts `#user-input`.
2. Startup always injects diagnostic info blocks into chat via `_render_startup_summary()` and resume info messaging in `_initialize_session()`.
3. Sidebar width is fixed at `32` cols in `src/loom/tui/widgets/sidebar.py`, regardless of sparse content.
4. Progress panel always renders with `No tasks tracked` empty text.
5. Status bar exists but is hidden by CSS (`#status-bar { display: none; }`), so user feedback is primarily chat lines + occasional `notify()`.

## Scope
In scope:
1. New startup landing state + transition into normal workspace view.
2. Conditional landing gating by resume detection and bypass preference.
3. Empty-state call-to-action content in landing.
4. Stylized text-logo branding for the landing surface.
5. Persistent workspace path context on landing.
6. De-noise startup/resume diagnostics in chat stream.
7. Sidebar and chat-density refinements.
8. Contrast/focus styling improvements for active controls.
9. Shortcut legend copy update from caret notation to lowercase `ctrl +` notation.
10. Better feedback for reload/auth/MCP actions.
11. Tests + docs/config updates for new behavior.

Out of scope:
1. Process orchestration semantics.
2. Auth/MCP workflow logic changes beyond UX messaging.

## UX Contract
1. Startup behavior:
2. If a session is resumed at startup, open directly into chat/workspace layout.
3. If no session is resumed, show a branded landing composer.
4. If bypass preference is enabled, always open directly into chat/workspace layout.
5. If landing is shown, session creation is deferred until first prompt submission.
6. `/new` always opens directly into chat/workspace (never via landing).
7. Landing content:
8. Centered brand/title.
9. Primary prompt box copy: `Tell me what you need me to solve`.
10. Minimal context line (workspace/model/auth status).
11. Optional 2-3 starter prompts (single-line, low visual weight).
12. A stylized `loom` text logo is rendered above the composer (text-based, no image assets).
13. Workspace path is visible while landing is active.
14. Transition:
15. Submitting landing prompt transitions immediately to normal chat/workspace layout and starts the turn.
16. No duplicate first message rendering between landing and chat.
17. Diagnostics:
18. Remove startup metadata wall from default chat entry path.
19. Keep full details accessible via `/session` and command palette.
20. Shortcut labels:
21. Use lowercase `ctrl +` labels in visible shortcut legend and auxiliary shortcut buttons.
22. Keep actual key bindings unchanged (`ctrl+*` remains the binding source of truth).
23. Footer:
24. Footer remains mounted and visible on landing and workspace surfaces.

## Trial and Revert Strategy
1. Gate new startup behavior behind TUI config toggles:
2. `tui.startup_landing_enabled` (default `true` for trial branch).
3. `tui.always_open_chat_directly` (default `false`).
4. Precedence: `always_open_chat_directly=true` always bypasses landing.
5. `/new` behavior is independent of startup landing toggles and always goes direct chat.
6. Keep existing chat-first startup path available as fallback branch in code (single guard).
7. Revert path: set `startup_landing_enabled=false` (or remove guarded path) without touching session/persistence internals.

## Architecture Changes

### 1) Startup Surface State
Files:
1. `src/loom/tui/app.py`

Plan:
1. Add an explicit startup surface mode state (landing vs workspace).
2. Resolve mode after `_resolve_startup_resume_target()` and before final focus decisions.
3. Centralize mode switching in one method with explicit branch policy:
4. startup may show landing (per resume + bypass rules)
5. `/new` always targets workspace/chat
6. Defer session creation when in landing mode until first submit.

Acceptance:
1. Resume path bypasses landing.
2. Non-resume path shows landing unless bypass is enabled.
3. `/new` bypasses landing and opens chat/workspace.
4. No empty sessions are created during landing idle.

### 2) Landing Widget
Files:
1. `src/loom/tui/widgets/landing.py` (new)
2. `src/loom/tui/widgets/__init__.py`
3. `src/loom/tui/app.py`

Plan:
1. Add a focused landing widget with:
2. Stylized `loom` text logo block (ASCII/text-art style tuned for terminal readability).
3. Prompt input (`#landing-input`) and placeholder text.
4. Context/status line.
5. Optional starter prompt buttons.
6. Footer row remains visible while landing is active.
7. Persistent workspace path text (screenshot-style placement) while landing is active.
8. Refactor submit logic so landing and chat input share a single text-dispatch path.

Acceptance:
1. Landing submit runs the same turn flow as `#user-input`.
2. Transition to workspace layout occurs before/with first streamed output.
3. Footer is visible and stable on landing.
4. Stylized `loom` logo is visible above the entry input at common terminal sizes.
5. Workspace path is visible on landing without opening sidebar/chat panels.

### 3) Layout and Density
Files:
1. `src/loom/tui/app.py`
2. `src/loom/tui/widgets/sidebar.py`
3. `src/loom/tui/widgets/chat_log.py`

Plan:
1. Sidebar:
2. Add compact mode class when workspace tree is sparse and no active progress rows.
3. Reduce width in compact mode (e.g., 24 from 32).
4. Hide progress block entirely when no tasks (instead of permanent empty panel).
5. Chat:
6. Add a readable content-width cap for info/model text (keep tool widgets unconstrained as needed).
7. Keep full-width behavior on very narrow terminals.

Acceptance:
1. Sparse workspace no longer wastes left-column space.
2. Main reading column is less fatiguing on wide terminals.

### 4) Conversation-First and Metadata De-Noise
Files:
1. `src/loom/tui/app.py`

Plan:
1. Stop auto-injecting verbose startup summary into chat on every initialization.
2. Replace with concise, single-line startup/resume notice (or notification).
3. Keep deep details in `/session` output and existing command palette actions.
4. Ensure session IDs are not duplicated in startup surfaces.

Acceptance:
1. First chat screen is task-oriented, not diagnostics-heavy.
2. No repeated session metadata blocks.

### 5) Visual Hierarchy and Focus
Files:
1. `src/loom/tui/app.py` (CSS section)
2. `src/loom/tui/widgets/sidebar.py` (CSS)
3. `src/loom/tui/theme.py` (if token tweaks needed)

Plan:
1. Increase active tab contrast and selected tree-node visibility.
2. Strengthen focused input border treatment vs unfocused surfaces.
3. Keep palette consistent with existing Loom dark theme.

Acceptance:
1. Active/focused elements are immediately identifiable at a glance.

### 6) Action Feedback Hardening
Files:
1. `src/loom/tui/app.py`

Plan:
1. Standardize user-visible success/error feedback helper for:
2. workspace reload
3. auth manager save/close changed
4. MCP manager save/close changed
5. Prefer short notifications plus optional chat info line for durable audit.

Acceptance:
1. User receives immediate confirmation/error for UX-affecting actions.

### 7) Shortcut Legend Readability
Files:
1. `src/loom/tui/app.py`
2. `src/loom/tui/commands.py` (if command labels are mirrored there)

Plan:
1. Replace footer-visible caret labels with explicit lowercase `ctrl +` strings where Loom controls label text.
2. Ensure custom footer shortcut buttons (`auth`, `mcp`) also use lowercase `ctrl +` style labels.
3. Keep one canonical visible shortcut surface to avoid conflicting notation.

Acceptance:
1. Primary visible shortcuts read as lowercase `ctrl + x` style, not caret shorthand.
2. Key behavior is unchanged.

### 8) Config Surface
Files:
1. `src/loom/config.py`
2. `loom.toml.example`
3. `README.md` (TUI behavior section)

Plan:
1. Add new `TUIConfig` booleans:
2. `startup_landing_enabled`
3. `always_open_chat_directly`
4. Parse defaults and document behavior.

Acceptance:
1. Trial behavior can be toggled without code edits.

## Implementation Sequence
1. Add config toggles and helper methods in `LoomApp`.
2. Implement landing widget and mode gating on startup.
3. Refactor submit path shared by landing/chat inputs.
4. Add stylized `loom` logo and persistent workspace-path rendering on landing.
5. Remove verbose startup summary injection and unify concise notices.
6. Implement sidebar compact/hide-empty-progress behavior.
7. Add chat width cap and focus/contrast style tweaks.
8. Update visible shortcut labels to lowercase `ctrl +` format.
9. Add standardized action feedback helper for reload/auth/MCP.
10. Add/update tests and docs.

## Test Plan
Files:
1. `tests/test_tui.py`
2. `tests/test_config.py`

Cases:
1. Startup routing:
2. resume target present -> workspace/chat directly
3. no resume + landing enabled -> landing shown
4. no resume + bypass enabled -> workspace/chat directly
5. landing idle does not create a session
6. `/new` opens workspace/chat directly (no landing)
7. Landing submit:
8. prompt transitions to workspace mode
9. first submit creates session and dispatches through shared turn path
10. landing footer remains visible
11. landing shows stylized `loom` text logo above composer
12. landing shows workspace path persistently
13. Sidebar density:
14. empty tasks hide progress section
15. sparse tree enables compact sidebar class
16. Metadata noise:
17. startup does not render old verbose summary block
18. `/session` still returns full diagnostic details
19. Action feedback:
20. reload/auth/MCP changed paths emit notify/info once each
21. Shortcut labels:
22. visible footer/legend labels show lowercase `ctrl +` format
23. key bindings still trigger expected actions
24. Config parsing:
25. new TUI config fields load defaults and explicit values

## Risks and Mitigations
1. Risk: landing introduces friction for frequent users.
2. Mitigation: bypass preference + automatic skip on resume.
3. Risk: duplicate submit behavior between landing and chat.
4. Mitigation: single shared dispatch function, dedicated tests.
5. Risk: layout regressions on small terminals.
6. Mitigation: width-cap only when viewport allows; snapshot/pilot tests.
7. Risk: trial rollback cost.
8. Mitigation: explicit config guard and isolated widget/state changes.

## Rollout Checklist
1. Merge behind config toggles.
2. Enable landing trial in local/dev configs.
3. Capture quick qualitative feedback (first-prompt latency, confusion points, usage of bypass).
4. Decide keep/adjust/revert after trial window.
