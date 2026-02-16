# TUI Review (2026-02-16)

## Scope
- Reviewed: `src/loom/tui/*`, setup flow integration in `src/loom/__main__.py`, and related app wiring.
- Validation method:
  - Static code review.
  - Headless Textual probes against `SetupScreen` and `LoomApp`.

## Executive Summary
The first-run setup issue is reproducible and rooted in focus/event routing: hidden `Input` widgets are still focused, so numeric keypresses are written into hidden inputs instead of being handled by setup step key handlers. There are additional high-impact TUI issues around modal input event leakage, setup-state duplication, and model/session consistency after re-setup.

---

## P0 Findings

### 1) Setup wizard numeric choices do not advance because hidden input keeps focus
- Symptoms: On provider step, pressing `1/2/3` does not move to details.
- Evidence:
  - Step-switch logic tries `self.set_focus(None)` for non-detail steps: `src/loom/tui/screens/setup.py:252-257`.
  - Key handling expects screen-level `on_key` events: `src/loom/tui/screens/setup.py:307-357`.
  - Probe showed focus remains on hidden `#input-url`; pressing `1` sets `input-url` value and `_step` stays `0`.
- Root cause:
  - Hidden inputs in non-active steps remain focusable and keep focus, swallowing keypresses.
  - Clearing focus via `set_focus(None)` is not sufficient here.
- Refactor plan:
  - Disable (`disabled=True`) and/or `can_focus=False` all setup inputs for non-details steps.
  - Enable/focus only currently active detail inputs when in details step.
  - Add explicit key bindings (`Binding("1", ...)`, etc.) for provider/roles/utility to avoid dependence on bubbled character events.

### 2) Modal setup input submissions leak into main chat submit handler
- Symptoms: Pressing Enter in setup input can trigger regular chat submission.
- Evidence:
  - App-level handler listens to all `Input.Submitted`: `src/loom/tui/app.py:534-552`.
  - No `event.input.id` guard for `"user-input"`.
  - Probe: entering value in setup URL field produced a chat turn payload (`"http://localhost:11434abc"`).
- Root cause:
  - Setup modal input events bubble to app-level `on_user_submit`.
- Refactor plan:
  - In `on_user_submit`, return early unless `event.input.id == "user-input"`.
  - Add defensive checks for active modal screens before handling chat submission.

### 3) Setup back-navigation can duplicate primary model entries and generate invalid TOML
- Symptoms: Going back from utility prompt to roles and re-selecting roles appends another `primary` model.
- Evidence:
  - Appends model unconditionally: `src/loom/tui/screens/setup.py:424-437`.
  - Back from utility does not remove prior primary draft: `src/loom/tui/screens/setup.py:490-492`.
  - `_generate_toml` with duplicate `[models.primary]` produces TOML parse error (`Cannot declare ('models', 'primary') twice`).
- Root cause:
  - Wizard state machine stores models as append-only history instead of a mutable primary/utility draft.
- Refactor plan:
  - Represent wizard state as explicit `primary_model` and `utility_model` drafts.
  - On re-selection, replace draft instead of append.
  - Add final validation that model names are unique before saving.

### 4) Setup allows saving config with no executor-capable model
- Symptoms: User can choose `Utility` roles for primary and skip utility model, then setup finishes but app fails to select executor.
- Evidence:
  - Role choice allows utility preset at primary step: `src/loom/tui/screens/setup.py:333-336`.
  - No final role coverage validation before save: `src/loom/tui/screens/setup.py:469-480`.
- Root cause:
  - Wizard validates fields, not runtime-required role coverage.
- Refactor plan:
  - Enforce coverage: at least one model with `executor` (and ideally `planner`).
  - In primary flow, either disallow utility-only preset or require utility-add path before confirm.

---

## P1 Findings

### 5) Re-running `/setup` in an existing session can leave session/model mismatch
- Symptoms: Config and status bar may show a new model while active `CoworkSession` remains the old one.
- Evidence:
  - `_finalize_setup` updates `self._model` then calls `_initialize_session`: `src/loom/tui/app.py:183-201`.
  - `_initialize_session` keeps existing session when `self._session is not None`: `src/loom/tui/app.py:288-290`.
- Root cause:
  - Re-setup path reuses existing session object rather than rebuilding with new model/system prompt.
- Refactor plan:
  - On setup completion during active app, explicitly create a new session (or rebind model in session safely) and mark old persisted session inactive.

### 6) Files panel shows only current turn changes, not session history
- Symptoms: Earlier changes disappear on subsequent turns.
- Evidence:
  - Per-turn file entries are rebuilt in `_update_files_panel`: `src/loom/tui/app.py:885-934`.
  - Panel clears table on each update: `src/loom/tui/widgets/file_panel.py:39-47`.
- Root cause:
  - UI state is treated as stateless per turn.
- Refactor plan:
  - Maintain cumulative file-change timeline in app state and append to table.
  - Optional: dedupe by path with latest operation or provide both “latest” and “all changes” modes.

### 7) TUI API client includes dead endpoint method
- Symptoms: `stream_all_events()` calls `/events/stream`, which is not implemented.
- Evidence:
  - Client method: `src/loom/tui/api_client.py:152-156`.
- Root cause:
  - Client/server contract drift.
- Refactor plan:
  - Remove method or implement route; add client-server contract tests.

### 8) TUI persistence path diverges from configured DB path
- Symptoms: TUI startup persistence uses `workspace.scratch_dir/loom.db` rather than `memory.database_path`.
- Evidence:
  - `src/loom/__main__.py:146-151`.
- Root cause:
  - Separate persistence path convention in TUI bootstrap.
- Refactor plan:
  - Align TUI persistence with `config.memory.database_path` or make split DBs explicit and configurable.

---

## P2 Findings

### 9) Setup key handler depends on `event.character` only
- Symptoms: Less resilient to non-character key variants (e.g., keypad behavior).
- Evidence:
  - `key = event.character`: `src/loom/tui/screens/setup.py:307-308`.
- Refactor plan:
  - Prefer normalized `event.key` checks and explicit bindings for supported keys.

### 10) Missing end-to-end tests for setup keyboard flow and event isolation
- Evidence gaps:
  - Existing setup tests cover utility methods but not full Textual key-flow behavior (`tests/test_setup.py`).
- Refactor plan:
  - Add Textual pilot tests for:
    - provider step `1/2/3` progression,
    - roles step key handling,
    - utility prompt `y/n`,
    - ensuring setup modal input submission does not hit chat `on_user_submit`.

---

## Repro Notes (from probes)
1. Provider step key ignored:
- Initial: `step=0`, focused widget `input-url`.
- After pressing `1`: `input-url.value == '1'`, `step` unchanged.

2. Modal input leak:
- During `/setup`, submitting setup URL field triggered app chat submit path with setup text payload.

3. Duplicate primary generation:
- Utility prompt -> back to roles -> choose roles again produced two `primary` entries.
- TOML output then fails to parse due duplicate `[models.primary]` table.

---

## Suggested Fix Order
1. Fix setup focus/event routing (P0-1) and app-level input event guard (P0-2).
2. Fix setup state model and validation (P0-3, P0-4).
3. Fix `/setup` session/model consistency (P1-5).
4. Resolve files-panel semantics and API client contract drift (P1-6/7).
5. Add missing end-to-end setup tests to lock behavior.
