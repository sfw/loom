# Desktop React State Architecture Refactor Plan (2026-04-01)

## Objective
Refactor the desktop frontend so it behaves like a well-structured React application:
1. state changes should propagate through targeted React updates
2. broad refreshes should become recovery paths, not the default update mechanism
3. local UI drafts and interaction state should stay local
4. large screens should not rerender because unrelated microstate changed elsewhere

## Executive Summary
The desktop app already uses React hooks and local state, but it still behaves too much like a refresh-driven client:
1. one giant app context causes wide rerender fan-out
2. workspace, thread, run, approval, and activity updates still rely on broad refetches
3. local draft state is coupled to global collection updates
4. polling is still filling gaps that should be covered by event-driven state transitions

The correct direction is not "more caching" or "fewer requests" in isolation. The correct direction is:
1. partition shared state by domain
2. introduce explicit patch/update functions for shared collections
3. treat SSE, optimistic actions, and response payloads as first-class state inputs
4. reserve heavyweight refreshes for initial load, manual refresh, reconnect recovery, and corruption repair

## Current Repo-Accurate Problems
1. `apps/desktop/src/context/AppContext.tsx` exposes a single provider value containing the entire app state and action surface.
2. `apps/desktop/src/hooks/useAppState.ts` returns one giant object, so almost every state change invalidates every `useApp()` consumer.
3. `apps/desktop/src/hooks/useWorkspace.ts` still treats `refreshWorkspaceSurface()` as a catch-all sync primitive for overview, approvals, and tab-visible data.
4. `apps/desktop/src/hooks/useConversation.ts` still escalates many thread-local transitions into workspace-surface refreshes:
   - auto-title persistence
   - turn completion/interruption
   - retry load
   - conversation creation
5. `apps/desktop/src/hooks/useInbox.ts` fans one approval action into several overlapping refreshes.
6. `apps/desktop/src/hooks/useWorkspace.ts` notification handling appends the event locally but still refreshes overview and approvals shortly after.
7. `apps/desktop/src/hooks/useConnection.ts` reloads the whole shell snapshot when archived-workspace visibility changes.
8. `apps/desktop/src/hooks/useDesktopActivity.ts` still depends on backend polling for activity that is already mostly known locally.
9. `apps/desktop/src/hooks/useWorkspace.ts` resets workspace draft state and file-tree mode whenever the `workspaces` array changes, even for unrelated workspace updates.
10. Large screen components such as `AppShell`, `Sidebar`, `ThreadsTab`, `RunsTab`, and `FilesTab` all subscribe to the same giant app context instead of narrow slices.

## Goals
1. Make shared state updates narrow, predictable, and domain-specific.
2. Eliminate routine use of broad workspace refreshes after known local events.
3. Preserve local draft state across unrelated collection updates.
4. Reduce unnecessary rerenders caused by the monolithic app context.
5. Reduce overlapping polling where SSE or optimistic updates already provide the needed signal.
6. Improve test coverage around state patching so refactors stay safe.

## Non-Goals
1. Replacing React with a different state library just to paper over architectural issues.
2. Rewriting the desktop app in one large cut-over.
3. Removing all server refresh paths; some recovery and bootstrap refreshes are still appropriate.
4. Changing backend transport away from SSE in this refactor wave.
5. Performing a broad visual redesign.

## Design Principles
1. Keep server data normalized by domain, not by screen.
2. Prefer explicit patch functions over "re-fetch everything".
3. Treat responses and stream events as authoritative incremental updates whenever possible.
4. Keep ephemeral UI microstate inside the component or feature hook that owns it.
5. Let React rerender from state changes, not from imperative reload choreography.
6. Make recovery refreshes explicit and rare.
7. Improve render isolation before introducing new memoization complexity.
8. Prefer multiple small contexts or selector-backed stores over one giant provider value.

## Target State Model

### 1. Shell State
Owns:
1. runtime status
2. connection state
3. workspace list summaries
4. global toasts
5. active tab and workspace/thread/run selection

Rules:
1. archived-workspace toggles only reload workspace summaries
2. runtime health polling only updates runtime state
3. shell bootstrap is separate from workspace-list refresh

### 2. Workspace State
Owns:
1. selected workspace overview
2. approval inbox
3. notifications
4. workspace inventory
5. workspace settings
6. workspace artifacts

Rules:
1. expose patchers such as:
   - `applyWorkspaceOverview`
   - `patchWorkspaceSummary`
   - `patchConversationSummary`
   - `patchRunSummary`
   - `removeApprovalItem`
   - `applyApprovalItem`
   - `appendNotification`
2. `refreshWorkspaceSurface()` becomes a recovery helper, not a routine post-action step

### 3. Conversation State
Owns:
1. conversation detail
2. messages
3. events
4. optimistic events
5. turn status
6. queued follow-up messages
7. conversation-local draft state

Rules:
1. thread events patch conversation state first
2. thread events patch workspace recent-thread rows and counters directly when needed
3. completion/interrupt paths should not require workspace overview reloads if the affected metadata is already known

### 4. Run State
Owns:
1. run detail
2. timeline
3. artifacts
4. instruction history
5. run-local controls and optimistic status

Rules:
1. continue the pattern now started for run status synchronization
2. patch workspace recent-run rows and counters directly from run events and run detail responses
3. do not re-fetch the whole workspace surface for routine pause/resume/complete flows

### 5. Files State
Owns:
1. file tree by directory
2. preview
3. editor draft
4. pending file open queue
5. file filtering microstate

Rules:
1. keep file explorer mode and local editor drafts independent from unrelated workspace-summary churn
2. only reload affected directories or file previews when file actions complete

### 6. UI Microstate
Owns only component-local concerns:
1. popovers
2. rename inputs
3. local search boxes
4. scroll anchors
5. expanded/collapsed sections

Rules:
1. do not lift this state into the app context unless multiple distant surfaces truly need it

## Workstream 0: Observability and Render Baseline

### Problem
We need evidence that rerender boundaries are improving and refresh storms are shrinking.

### Plan
1. Add lightweight instrumentation for key frontend actions:
   - workspace selection
   - conversation create/send/approval resolve
   - run launch/control
   - notification burst handling
2. Add targeted tests around patch functions and refresh suppression.
3. Add a simple render-count harness for the most expensive screens/components.
4. Document allowed reasons to call heavyweight refresh helpers.

### Primary Files
1. `apps/desktop/src/hooks/useAppState.ts`
2. `apps/desktop/src/hooks/useWorkspace.ts`
3. `apps/desktop/src/hooks/useConversation.ts`
4. `apps/desktop/src/hooks/useRuns.ts`
5. desktop hook/component test files

### Acceptance
1. We can show fewer broad refresh calls in routine flows.
2. We can show reduced rerenders for unrelated tabs/components during active thread/run updates.

## Workstream 1: Split the Monolithic App Context

### Problem
The current `AppContext` guarantees broad rerender fan-out.

### Plan
1. Replace the single omnibus context with multiple focused providers or selector-backed stores.
2. Initial split should be practical rather than theoretical:
   - `ShellContext`
   - `WorkspaceContext`
   - `ConversationContext`
   - `RunContext`
   - `FilesContext`
   - `UIContext`
3. Keep action references stable within each provider.
4. Convert high-traffic screens to consume only the contexts they need.
5. If rerender pressure remains high after the split, evaluate a small `useSyncExternalStore`-backed store for the most frequently updated stream domains.

### Primary Files
1. `apps/desktop/src/context/AppContext.tsx`
2. `apps/desktop/src/hooks/useAppState.ts`
3. `apps/desktop/src/components/AppShell.tsx`
4. `apps/desktop/src/components/Sidebar.tsx`
5. `apps/desktop/src/components/ThreadsTab.tsx`
6. `apps/desktop/src/components/RunsTab.tsx`
7. `apps/desktop/src/components/FilesTab.tsx`

### Acceptance
1. A conversation stream update does not rerender run-only consumers by default.
2. A file preview/edit update does not rerender shell metrics or run panes by default.
3. A run timeline update does not rerender thread-only consumers by default.

## Workstream 2: Workspace Patch Layer

### Problem
Workspace state still lacks a complete local-first mutation layer, so many features fall back to `refreshWorkspaceSurface()`.

### Plan
1. Expand workspace patchers to cover all shared workspace-facing collections:
   - recent conversations
   - recent runs
   - workspace summary counters
   - approval inbox
   - notifications
2. Introduce helper operations such as:
   - `upsertConversationSummary`
   - `upsertRunSummary`
   - `removeConversationSummary`
   - `removeRunSummary`
   - `setConversationProcessing`
   - `setRunStatus`
   - `resolveApprovalItem`
   - `increment/decrement pending counters`
3. Make these patch functions the default integration path for thread/run/inbox hooks.
4. Restrict `refreshWorkspaceSurface()` to:
   - initial workspace load
   - reconnect recovery
   - manual refresh
   - explicit "server truth may have drifted" recovery cases

### Primary Files
1. `apps/desktop/src/hooks/useWorkspace.ts`
2. `apps/desktop/src/utils.ts`
3. related hook tests

### Acceptance
1. Most post-action update paths no longer call `refreshWorkspaceSurface()`.
2. Workspace overview/sidebar values stay in sync through patchers.
3. Approval counts and recent items update without a full workspace reload.

## Workstream 3: Conversation Hook Refactor

### Problem
`useConversation` still escalates local thread events into broad workspace refreshes.

### Plan
1. Add explicit workspace patch callbacks for conversation lifecycle events:
   - create conversation
   - rename conversation
   - send first message / auto-title
   - turn started
   - turn completed
   - turn interrupted
   - approval requested/resolved
   - ask-user prompt surfaced/resolved
2. Remove `includeWorkspaceSurface` from routine conversation refresh scheduling.
3. Keep `refreshConversation()` as the recovery mechanism for missed SSE events or reconnects.
4. Patch recent thread rows directly when:
   - title changes
   - `is_active` flips
   - `last_active_at` changes
   - linked run ids change
5. After conversation creation, insert/select the new conversation directly instead of reloading workspace overview first.
6. Keep optimistic turn microstate local to the conversation domain.

### Primary Files
1. `apps/desktop/src/hooks/useConversation.ts`
2. `apps/desktop/src/hooks/useWorkspace.ts`
3. `apps/desktop/src/components/ThreadsTab.tsx`
4. `apps/desktop/src/components/Sidebar.tsx`

### Acceptance
1. Normal thread turns do not trigger workspace-surface refreshes.
2. Conversation create/rename/title-sync flows update sidebar and overview from local patchers.
3. Retry/recovery paths still work when SSE state is missing or stale.

## Workstream 4: Run Hook Completion

### Problem
Run status synchronization has started moving in the right direction, but the rest of the run domain still needs the same treatment.

### Plan
1. Extend run patching beyond status:
   - goal/process label updates
   - changed-file counts
   - timestamps
   - linked conversation ids
   - delete/restart/remove flows
2. Patch workspace recent-run collections directly on:
   - launch
   - pause/resume/cancel
   - completion/failure/cancelled
   - delete
3. Reserve workspace refreshes for:
   - newly created run bootstrap when the server returns insufficient metadata
   - reconnect recovery
   - explicit recovery after mismatch
4. Keep timeline/artifact hydration inside the run domain.

### Primary Files
1. `apps/desktop/src/hooks/useRuns.ts`
2. `apps/desktop/src/hooks/useWorkspace.ts`
3. `apps/desktop/src/components/RunsTab.tsx`
4. `apps/desktop/src/components/Sidebar.tsx`
5. `apps/desktop/src/components/OverviewTab.tsx`

### Acceptance
1. Run lifecycle events update recent-run UI without workspace overview reloads.
2. Deleting/restarting a run updates local collections immediately.
3. Overview/sidebar/run detail remain consistent under rapid status transitions.

## Workstream 5: Inbox and Notification Refactor

### Problem
Approval and notification flows still use redundant refetch chains.

### Plan
1. On approval reply, patch or remove the affected approval item locally first.
2. Patch the relevant conversation/run state directly when the affected item is currently open.
3. Stop calling both `refreshWorkspaceSurface()` and `refreshApprovalInbox()` for the same approval mutation.
4. Treat notification stream events as state updates, not just invalidation signals.
5. Only schedule overview refreshes for notification types whose downstream state cannot be derived locally.

### Primary Files
1. `apps/desktop/src/hooks/useInbox.ts`
2. `apps/desktop/src/hooks/useWorkspace.ts`
3. `apps/desktop/src/hooks/useConversation.ts`
4. `apps/desktop/src/hooks/useRuns.ts`

### Acceptance
1. Approval actions no longer trigger overlapping refresh cascades.
2. Notification handling updates visible UI primarily via local patches.
3. Pending-approval counts stay correct without redundant calls.

## Workstream 6: Shell Bootstrap and Connection Decoupling

### Problem
Shell bootstrap, workspace-list refresh, and archived-list filtering are still coupled.

### Plan
1. Split shell bootstrap from workspace-list loading.
2. Changing `showArchivedWorkspaces` should only fetch workspace summaries.
3. Connection health polling should only update runtime connectivity and readiness state.
4. Preserve existing shell data during reconnect attempts wherever safe.
5. Add a dedicated workspace-list refresh helper instead of re-running the full shell bootstrap.

### Primary Files
1. `apps/desktop/src/hooks/useConnection.ts`
2. `apps/desktop/src/hooks/useWorkspace.ts`
3. `apps/desktop/src/components/WorkspaceModal.tsx`

### Acceptance
1. Archived workspace toggles do not reload models/settings/runtime.
2. Reconnect behavior remains correct.
3. Workspace list updates become cheaper and more predictable.

## Workstream 7: Preserve Local Draft and Interaction State

### Problem
Some local drafts reset when unrelated shared data changes.

### Plan
1. Stop keying local workspace edit drafts off the entire `workspaces` array.
2. Only reset workspace drafts when:
   - the selected workspace changes
   - the specific selected workspace metadata changes and the draft is not dirty
3. Keep file tree mode, expanded sections, and local filters independent from unrelated summary churn.
4. Audit other components for the same pattern:
   - rename inputs
   - launcher selections
   - inbox reply drafts
   - tab-local searches

### Primary Files
1. `apps/desktop/src/hooks/useWorkspace.ts`
2. `apps/desktop/src/hooks/useFiles.ts`
3. `apps/desktop/src/components/Sidebar.tsx`
4. `apps/desktop/src/components/OverviewTab.tsx`
5. `apps/desktop/src/components/RunsTab.tsx`

### Acceptance
1. Unrelated workspace summary updates do not reset in-progress edits.
2. File tree and editor state remain stable during run/thread activity.

## Workstream 8: Reduce Polling and Let Local State Drive Activity

### Problem
Several parts of the UI still poll for state that the app mostly already knows.

### Plan
1. Keep runtime health polling, but scope it narrowly to runtime status.
2. Reduce activity polling by relying on:
   - local conversation status
   - local run status
   - stream health heuristics
3. Continue using recovery polling only when stream health goes stale.
4. Document the small set of states that still require polling because they cannot be inferred locally.

### Primary Files
1. `apps/desktop/src/hooks/useDesktopActivity.ts`
2. `apps/desktop/src/hooks/useConversation.ts`
3. `apps/desktop/src/hooks/useRuns.ts`
4. `apps/desktop/src/hooks/useConnection.ts`

### Acceptance
1. Polling intervals and fetch counts drop during healthy active sessions.
2. Activity indicators remain correct when streams are healthy.
3. Recovery remains robust after disconnects or missed events.

## Workstream 9: Component Subscription and Render Boundary Cleanup

### Problem
Even with better state patching, broad subscriptions will keep render costs high.

### Plan
1. Move components off broad `useApp()` consumption and onto domain-specific hooks/contexts.
2. Split heavy screens into data containers and presentational children.
3. Memoize only where boundaries are now meaningful; do not paper over bad subscriptions with blanket memoization.
4. Keep virtualization where it already pays off, especially in thread/run histories.
5. Add targeted memo boundaries for rows/cards whose props are already narrow and stable.

### Primary Files
1. `apps/desktop/src/components/AppShell.tsx`
2. `apps/desktop/src/components/Sidebar.tsx`
3. `apps/desktop/src/components/ThreadsTab.tsx`
4. `apps/desktop/src/components/RunsTab.tsx`
5. `apps/desktop/src/components/FilesTab.tsx`

### Acceptance
1. Shell/sidebar do not rerender for deep timeline token changes unless their own props changed.
2. Run and thread panes remain responsive while other domains update.
3. The React tree is easier to reason about because data ownership follows UI ownership.

## Workstream 10: Test Strategy

### Plan
1. Add hook tests for workspace patchers:
   - run upsert/update/remove
   - conversation upsert/update/remove
   - approval resolve/remove
   - notification append/coalesce
2. Add regression tests proving routine flows do not call broad refresh helpers.
3. Add render-isolation tests for key components after context splitting.
4. Add recovery tests proving reconnect/manual-refresh paths still heal stale UI.

### Primary Test Areas
1. `apps/desktop/src/hooks/*.test.tsx`
2. `apps/desktop/src/components/*.test.tsx`

### Acceptance
1. We can prove the frontend stays in sync without broad reloads in normal flows.
2. We can prove recovery behavior still works when local state diverges.

## Implementation Order
1. Add instrumentation and "allowed refresh path" documentation.
2. Split the monolithic app context into domain providers.
3. Build the workspace patch layer.
4. Migrate conversation flows to patch-based updates.
5. Finish run-domain migration to patch-based updates.
6. Refactor inbox/notification flows to patch locally first.
7. Decouple shell bootstrap from workspace-list filtering and health refresh.
8. Preserve local draft state and clean up ownership boundaries.
9. Reduce remaining polling based on stream health.
10. Tighten render boundaries and add regression coverage.

## Risks and Mitigations
1. Risk: patch logic drifts from server truth.
   - Mitigation: keep explicit recovery refreshes and test patch/recovery parity.
2. Risk: context splitting increases wiring complexity.
   - Mitigation: split by domain in phases, with compatibility adapters during migration.
3. Risk: optimistic updates introduce brief inconsistencies.
   - Mitigation: keep server reconciliation paths and add targeted tests for roll-forward/rollback.
4. Risk: overusing memoization makes the app harder to maintain.
   - Mitigation: fix subscription boundaries first, then add memoization only where props are already narrow.

## Acceptance Criteria for the Refactor Wave
1. `refreshWorkspaceSurface()` is no longer used as the default post-action path for normal thread/run/approval updates.
2. Conversation turn completion, approval resolution, and run completion update visible UI through local patchers first.
3. Archived workspace filtering no longer reloads the full shell snapshot.
4. Local draft state does not reset because an unrelated workspace summary changed.
5. Large screens subscribe only to the domains they need.
6. Render-count and request-count measurements show clear reductions during active thread/run scenarios.

## Deliverables
1. Refactored context/store structure for the desktop app.
2. A complete workspace patch/update layer for shared collections and counters.
3. Conversation, run, inbox, and notification flows that default to local-first updates.
4. Reduced polling and clearer recovery-only refresh paths.
5. Regression coverage proving the new React state model stays synchronized.

## Suggested Naming and Tracking
Implementation should be tracked as a single frontend refactor program with sub-PRs by workstream, not as one giant all-or-nothing change.

Recommended branch/PR slicing:
1. context split
2. workspace patch layer
3. conversation local-first updates
4. run local-first updates
5. inbox/notification local-first updates
6. shell/bootstrap decoupling
7. draft-state preservation and render-boundary cleanup
