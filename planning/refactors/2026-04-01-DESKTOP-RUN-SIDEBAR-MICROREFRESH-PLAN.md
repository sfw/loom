# Desktop Run Sidebar Microrefresh Plan (2026-04-01)

## Objective
Make newly launched and status-changing runs appear in the desktop sidebar through targeted state updates, not broad workspace refreshes.

## Executive Summary
The missing sidebar run is not primarily a rendering bug. It is a state propagation gap caused by inconsistent workspace identity for auto-subfolder runs.

Today:
1. run creation intentionally scopes execution into a subfolder
2. workspace overview correctly groups those runs under the parent workspace
3. run detail resolves workspace identity from the scoped subfolder path instead of the parent workspace root
4. the frontend patch path drops the update when `workspace_id` is blank
5. the desktop notification stream does not carry general run lifecycle events, so there is no automatic repair path short of a manual or heavyweight refresh

This is exactly the kind of issue the frontend refactor should fix with local-first state transitions:
1. backend responses must carry stable parent-workspace identity
2. launch/control/detail/stream flows must patch the selected workspace's run collection directly
3. full workspace refreshes should remain recovery paths only

## Observed Runtime Evidence
From the desktop runtime database for the reported run:
1. task id: `0cd532ca`
2. created at: `2026-04-01T20:00:50.143155`
3. updated at: `2026-04-01T20:03:43.912195`
4. status: `failed`
5. stored `workspace_path`: `/Users/sfw/Documents/loom-root/AI-Native/can-you-research-who-is-going`
6. stored metadata:
   - `source_workspace_root = /Users/sfw/Documents/loom-root/AI-Native`
   - `run_workspace_relative = can-you-research-who-is-going`
   - `run_workspace_mode = scoped_subfolder`
7. registered workspace row:
   - `ws-027ce6429feb -> /Users/sfw/Documents/loom-root/AI-Native`

This means the task is correctly grouped under `AI-Native`, but only if consumers use `source_workspace_root` fallback logic.

## Repo-Accurate Diagnosis

### 1. Auto-subfolder launch is intentional
`POST /tasks` rewrites the task workspace into a scoped subfolder and stores the parent workspace in metadata.

Primary files:
1. `src/loom/api/routes.py`

Relevant behavior:
1. `task_workspace` becomes the scoped run folder
2. `metadata["source_workspace_root"]` stores the parent workspace root
3. desktop run launch always sets `auto_subfolder: true`

### 2. Workspace overview already does the right grouping
Workspace-scoped task listing uses:
1. `metadata.source_workspace_root`
2. falling back to `workspace_path`

So overview and workspace run lists are already conceptually correct.

Primary files:
1. `src/loom/state/memory.py`
2. `src/loom/api/routes.py`

### 3. Run detail uses the wrong workspace identity source
`GET /runs/{run_id}` currently resolves workspace identity from `task_row.workspace_path` directly. For auto-subfolder runs, that path is the scoped child folder, not the registered parent workspace root.

Primary files:
1. `src/loom/api/routes.py`

Consequence:
1. `workspace_id` can be blank on run detail responses for valid parent-grouped runs
2. the frontend cannot safely patch `overview.recent_runs`

### 4. Frontend microrefresh path correctly refuses ambiguous updates
The shared workspace patcher exits early when `detail.workspace_id` is blank.

Primary files:
1. `apps/desktop/src/hooks/useWorkspace.ts`

This guard is reasonable. The problem is upstream identity inconsistency, not the guard itself.

### 5. Launch path intentionally avoids broad workspace refresh
The desktop run launcher:
1. creates the run
2. selects the run
3. relies on follow-up run detail/state patching
4. does not call `refreshWorkspaceSurface()`

Primary files:
1. `apps/desktop/src/hooks/useRuns.ts`

That matches the desired architecture.

### 6. Workspace notification stream does not cover run lifecycle changes
Desktop workspace notifications currently stream only:
1. approvals
2. ask-user events

They do not stream:
1. task created
2. task planning
3. task executing
4. task paused
5. task completed
6. task failed

Primary files:
1. `src/loom/api/routes.py`
2. `apps/desktop/src/hooks/useWorkspace.ts`

Consequence:
1. if the local run patch path misses, the sidebar remains stale
2. only a later overview refresh repairs it

## Root Cause
The state architecture has one broken edge:
1. launch produces a parent-grouped run
2. run detail returns child-workspace identity
3. workspace patchers reject the update
4. no workspace-scoped run lifecycle stream exists to compensate

So the sidebar is not missing the run because React failed. It is missing the run because the state graph never received a usable parent-workspace run patch.

## Goals
1. Keep routine run visibility fully local-first and patch-driven.
2. Preserve `refreshWorkspaceSurface()` as a recovery path, not a standard launch/control dependency.
3. Guarantee that auto-subfolder runs resolve to the parent workspace consistently across overview, run detail, approvals, and notifications.
4. Make run launch and pause/resume state changes immediately visible in the selected workspace sidebar.
5. Add regression coverage for auto-subfolder identity and sidebar patch behavior.

## Non-Goals
1. Reverting auto-subfolder run workspaces.
2. Reintroducing broad workspace refreshes after every run action.
3. Replacing the desktop state model with a new library.
4. Redesigning the run UI.

## Design Principles
1. The parent workspace root is the stable grouping identity for desktop workspace surfaces.
2. Run detail, approvals, overview, and notifications must all agree on that identity.
3. Local mutations should patch collections immediately when the affected workspace is already known.
4. Stream and detail payloads should reconcile optimistic local rows, not replace the architecture with refetches.
5. Full overview refresh remains appropriate only for bootstrap, reconnect, manual refresh, and explicit stale-repair flows.

## Proposed Fix

### Workstream 1: Unify Task Workspace Grouping Semantics

#### Problem
The backend already knows how to group task-backed runs under a parent workspace for overview queries, but it does not reuse that logic for run detail and task-context-derived payloads.

#### Plan
1. Introduce a single helper for "task workspace group root":
   - prefer `metadata.source_workspace_root`
   - otherwise use `workspace_path`
2. Use that helper anywhere workspace identity is resolved for task-backed objects.
3. Keep `workspace_path` in run detail as the actual run workspace path for artifacts/file routing, but compute `workspace_id` and `workspace` summary from the group root.

#### Primary files
1. `src/loom/api/routes.py`
2. `src/loom/state/workspaces.py` if shared canonicalization helpers need reuse

#### Target call sites
1. `get_run()`
2. `_task_context()`
3. any notification/approval payload builders that use `_task_context()`

#### Acceptance
1. `GET /runs/{id}` returns the parent workspace id for auto-subfolder runs.
2. task approvals/questions for auto-subfolder runs also resolve to the parent workspace id.

### Workstream 2: Make Launch a True Local Upsert

#### Problem
The current launch path selects the new run and waits for later run detail to patch workspace state. That is fine in principle, but it leaves a visibility hole when the first authoritative detail payload has ambiguous workspace identity.

#### Plan
1. On successful `createTask()`, immediately upsert a provisional run summary into the selected workspace overview using already-known local information:
   - `task_id`
   - selected workspace id
   - selected workspace path
   - goal
   - process
   - optimistic status such as `planning` or `executing`
2. Keep this upsert in the run hook or workspace patch layer, not in a broad refresh helper.
3. When `fetchRunDetail()` returns, reconcile the provisional row with authoritative values.
4. Keep this as an additive safety net even after backend identity is fixed, because it improves perceived liveness and removes launch flicker.

#### Primary files
1. `apps/desktop/src/hooks/useRuns.ts`
2. `apps/desktop/src/hooks/useWorkspace.ts`

#### Acceptance
1. A newly launched run appears in the sidebar immediately after creation.
2. The row later reconciles cleanly with the authoritative run detail.
3. No workspace refresh is needed for normal launch visibility.

### Workstream 3: Harden Run-State Microrefreshes

#### Problem
Pause/resume/cancel and terminal transitions should update shared workspace state entirely through local patching.

#### Plan
1. Keep optimistic local run detail transitions for pause/resume/cancel.
2. Ensure those transitions always patch the current workspace run row.
3. Ensure selected-run stream events continue to patch the shared workspace row on status changes.
4. Add a small defensive fallback:
   - if a run detail payload arrives with blank `workspace_id`
   - but the current selected run already exists in the selected workspace recent-runs list
   - reconcile into that existing row rather than dropping the patch
5. Treat this fallback as defensive hardening, not the primary fix.

#### Primary files
1. `apps/desktop/src/hooks/useRuns.ts`
2. `apps/desktop/src/hooks/useWorkspace.ts`

#### Acceptance
1. Pause/resume/cancel transitions update the sidebar row without overview refresh.
2. Terminal status updates update the sidebar row without overview refresh.
3. Auto-subfolder runs no longer disappear from the patch path.

### Workstream 4: Optional Workspace-Scoped Run Lifecycle Stream

#### Problem
The current notification stream is approval/question only. That is acceptable for the short-term fix, but it leaves the workspace shell blind to run creation or run status changes unless the run is actively selected or a recovery refresh happens.

#### Plan
1. Do not block the main fix on this.
2. Consider a later additive workspace-scoped stream for:
   - `task_created`
   - `task_planning`
   - `task_executing`
   - `task_paused`
   - `task_completed`
   - `task_failed`
3. If added, route those events into the same run patch layer rather than another broad refresh path.

#### Primary files
1. `src/loom/api/routes.py`
2. `apps/desktop/src/hooks/useWorkspace.ts`
3. `apps/desktop/src/api.ts`

#### Acceptance
1. Unselected runs can still patch workspace recent-run state in real time.
2. No broad overview refresh is introduced.

## Implementation Order
1. Fix backend workspace identity consistency for task-backed run detail and task context.
2. Add immediate optimistic run upsert on launch.
3. Add defensive frontend reconciliation for ambiguous run detail payloads.
4. Add tests for launch, pause/resume, and terminal-state sidebar patching.
5. Evaluate whether the optional workspace-scoped run lifecycle stream is still worth doing after the first four steps.

## Test Strategy

### API Tests
1. Add a test that `GET /runs/{id}` for an auto-subfolder run returns the parent workspace id.
2. Add a test that `_task_context()`-driven task approval/question payloads for auto-subfolder runs resolve to the parent workspace id.
3. Keep the existing grouped-overview test as the baseline proof that overview behavior remains correct.

Primary files:
1. `tests/test_api.py`

### Frontend Hook Tests
1. `useRuns` launch success should immediately upsert a recent-run row without calling `refreshWorkspaceSurface()`.
2. `useWorkspace.syncRunDetail` should accept authoritative run detail for auto-subfolder runs after backend fix.
3. Add a defensive test for reconciling an existing selected-workspace run row when detail arrives with blank `workspace_id`.
4. Pause/resume/cancel optimistic state changes should update shared workspace run state locally.
5. Terminal stream events should update shared workspace run state locally and still avoid broad refreshes.

Primary files:
1. `apps/desktop/src/hooks/useRuns.test.tsx`
2. `apps/desktop/src/hooks/useWorkspace.test.tsx`

### Sidebar/Screen Tests
1. Launch a run and assert the sidebar shows the run row immediately.
2. Pause a selected run and assert the sidebar badge updates to `Paused`.
3. Resume and complete/fail flows should patch the badge/status text in place.

Primary files:
1. `apps/desktop/src/components/Sidebar.test.tsx`
2. optionally `apps/desktop/src/components/RunsTab.test.tsx`

## Rollout Notes
1. This fix aligns with the current React-state refactor direction.
2. It reduces dependency on broad workspace refreshes instead of increasing it.
3. The backend identity unification is the highest-leverage step because it repairs multiple surfaces at once:
   - run detail
   - task approvals
   - task questions
   - any future task-backed notification payloads

## Risks and Mitigations
1. Risk: changing task-context workspace resolution could affect approval/inbox grouping.
   - Mitigation: add explicit auto-subfolder approval/question tests.
2. Risk: optimistic launch upserts could duplicate rows.
   - Mitigation: always key by task id and reconcile with authoritative detail.
3. Risk: preserving both run workspace path and parent workspace identity may confuse future callers.
   - Mitigation: document that `workspace_id` is grouping identity while `workspace_path` remains the actual run workspace path.

## Acceptance Criteria
1. Launching an auto-subfolder run makes it appear in the sidebar immediately.
2. Pausing, resuming, completing, and failing a run update the sidebar through local state transitions.
3. `refreshWorkspaceSurface()` is not required for routine run visibility.
4. `GET /runs/{id}` and task-context-derived payloads consistently identify the parent workspace for auto-subfolder runs.
5. The existing workspace-overview grouping behavior remains unchanged.
