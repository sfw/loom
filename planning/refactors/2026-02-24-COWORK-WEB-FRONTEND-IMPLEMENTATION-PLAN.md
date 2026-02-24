# Cowork Web Frontend Implementation Plan (2026-02-24)

## Objective
Expose Loom cowork as a staff-facing web experience with three first-class capabilities:
1. Conversational cowork (streaming assistant + tool activity).
2. Drag-and-drop file upload into the active workspace.
3. A dedicated logs panel, separate from chat, for live operational visibility.

## Why This Plan
Cowork runtime logic already exists and is stable in the TUI, but there is no cowork API surface for browsers. This plan adds a thin web layer around existing cowork primitives instead of reimplementing execution behavior.

## MVP Scope
1. Multi-session cowork over HTTP/WebSocket.
2. Upload one or more files via multipart and persist to workspace safely.
3. Persist and stream cowork logs for a separate panel.
4. Staff-usable web UI with chat, uploads, and logs.

## Non-Goals (MVP)
1. Full enterprise auth/RBAC system in Loom itself.
2. Full workspace file explorer/editor parity with the TUI.
3. Distributed multi-node session coordination.
4. Replacing existing task API behavior.

## Current Baseline (Repo-Accurate)
1. `CoworkSession` already emits streaming chunks, `ToolCallEvent` start/complete, and final `CoworkTurn` summaries.
2. Conversation persistence exists in SQLite: `cowork_sessions` and `conversation_turns`.
3. API routes are task-oriented only (`/tasks/*`, `/models`, `/tools`, `/health`, `/config`) with no cowork endpoints.
4. Event persistence (`events` table) is task-scoped (`task_id` foreign key), so cowork logs cannot directly reuse it.
5. `write_file` is text-only and capped at 1 MB, so it is not suitable for drag/drop binary upload.
6. Path safety helpers already exist in tool infrastructure (`_resolve_path`, workspace boundary checks).
7. API server currently defaults to local host and local-only CORS.

## Proposed Architecture

### 1) Cowork API Layer
Add cowork-specific API endpoints and a runtime session manager:
1. Session lifecycle (create/list/get).
2. Turn history retrieval.
3. Live stream transport for chat + tool events.
4. File upload endpoint bound to session workspace.
5. Log history endpoint.

### 2) Cowork Session Runtime Manager
Create a small in-memory manager attached to API `Engine`:
1. `session_id -> CoworkSession` map.
2. Per-session `asyncio.Lock` to serialize in-flight turns.
3. Lazy resume from `ConversationStore` when session exists in DB but not memory.
4. Fan-out publisher for live log events to connected WebSocket clients.

### 3) Log Persistence Path
Add a dedicated cowork log table (separate from task `events`):
1. Session-scoped append-only records with `session_id`, `seq`, `event_type`, `payload`, `created_at`.
2. Backfill endpoint for logs panel replay and reconnect.
3. Keep streaming text chunks out of durable logs to avoid DB bloat; persist milestones instead.

### 4) Web Frontend Shell
Build a dedicated frontend app with a 3-panel layout:
1. Left: workspace/upload panel (dropzone + recent uploads).
2. Center: cowork chat stream.
3. Right: live logs panel.

## API Contract Draft (v1)

### Session + History
1. `POST /cowork/sessions`
2. `GET /cowork/sessions`
3. `GET /cowork/sessions/{session_id}`
4. `GET /cowork/sessions/{session_id}/turns?offset=<int>&limit=<int>`

### Live Interaction
1. `WS /cowork/sessions/{session_id}/stream`
2. Client message types:
3. `user_message` (required payload: `message`)
4. `ping`
5. Server message types:
6. `text_delta`
7. `tool_call_started`
8. `tool_call_completed`
9. `turn_completed`
10. `awaiting_user_input` (for `ask_user`)
11. `log_event` (for logs panel updates)
12. `error`

### Workspace Upload
1. `POST /cowork/sessions/{session_id}/uploads` (multipart/form-data)
2. Form fields:
3. `files` (one or many)
4. `target_dir` (optional, workspace-relative, defaults to `"."`)
5. `overwrite` (optional bool, default `false`)
6. Response includes per-file status, relative path, size, and error details.

### Logs
1. `GET /cowork/sessions/{session_id}/logs?cursor=<seq>&limit=<int>`
2. Returns chronological log events for panel hydration and pagination.

## Data Model Changes

### SQLite Schema
Add table:
1. `cowork_events`
2. Columns: `id`, `session_id`, `seq`, `event_type`, `payload`, `created_at`
3. Indexes: `(session_id, seq)`, `(session_id, created_at)`, `(event_type)`

### Conversation Store
Extend `ConversationStore`:
1. `append_event(...)`
2. `get_events(session_id, cursor, limit)`
3. `get_event_count(session_id)`

## Upload Design Details (Drag/Drop)
1. Use `UploadFile` streaming to avoid loading full files in memory.
2. Enforce path confinement to workspace root (`target_dir` plus sanitized filename).
3. Enforce configurable size limits (per-file + per-request).
4. Reject path traversal (`..`), absolute paths, and symlink escape attempts.
5. Emit log milestones:
6. `upload_started`
7. `upload_completed`
8. `upload_failed`

Recommended config additions:
1. `[server] max_upload_file_bytes` (default e.g. 50 MB)
2. `[server] max_upload_request_bytes` (default e.g. 250 MB)

Dependency note:
1. Add `python-multipart` to support FastAPI multipart parsing.

## Log Panel Design Details
Persist and stream high-signal operational events:
1. `session_started`
2. `user_message_received`
3. `tool_call_started`
4. `tool_call_completed` (success/failure + elapsed ms)
5. `turn_completed` (tokens/model)
6. `upload_*`
7. `session_error`

Do not persist:
1. Raw text token deltas.
2. Full file bytes.
3. Secret-bearing payload fields.

## Workstreams

### W1: Cowork API Foundation
Files:
1. `/Users/sfw/Development/loom/src/loom/api/engine.py`
2. `/Users/sfw/Development/loom/src/loom/api/schemas.py`
3. `/Users/sfw/Development/loom/src/loom/api/routes.py` (or split cowork routes into a new module)
4. `/Users/sfw/Development/loom/src/loom/state/conversation_store.py`

Deliverables:
1. Session lifecycle endpoints.
2. Turn history endpoint.
3. Engine wiring for conversation store + cowork manager.

Acceptance:
1. Browser client can create and resume cowork sessions through API only.

### W2: Live Streaming Transport
Files:
1. `/Users/sfw/Development/loom/src/loom/api/routes.py` (or new cowork routes module)
2. `/Users/sfw/Development/loom/src/loom/cowork/session.py` (if minimal adapters needed)

Deliverables:
1. WebSocket endpoint bridging `send_streaming(...)` events to typed JSON frames.
2. Session-level concurrency guard (one active turn per session).

Acceptance:
1. Client receives text/tool/turn events in order for each submitted message.

### W3: Workspace Upload Endpoint
Files:
1. `/Users/sfw/Development/loom/src/loom/api/routes.py` (or new cowork routes module)
2. `/Users/sfw/Development/loom/src/loom/config.py`
3. `/Users/sfw/Development/loom/pyproject.toml`

Deliverables:
1. Multipart upload endpoint with safe path resolution.
2. Size-limit enforcement and per-file status reporting.
3. Upload events emitted to logs.

Acceptance:
1. User can drag/drop multiple files; uploaded files appear in workspace safely.
2. Traversal and oversize attempts fail with clear errors.

### W4: Cowork Log Persistence + Query
Files:
1. `/Users/sfw/Development/loom/src/loom/state/schema.sql`
2. `/Users/sfw/Development/loom/src/loom/state/conversation_store.py`
3. `/Users/sfw/Development/loom/src/loom/api/routes.py` (or new cowork routes module)

Deliverables:
1. `cowork_events` table and store methods.
2. Log query endpoint with cursor/limit pagination.
3. Server-side event emission for tool/upload/turn milestones.

Acceptance:
1. Logs panel can recover full recent event history after reconnect.

### W5: Web Frontend MVP
Files (new app, proposed):
1. `/Users/sfw/Development/loom/web/cowork/` (or agreed frontend location)

Deliverables:
1. Session picker/create flow.
2. Chat panel with incremental streaming text.
3. Drag/drop upload panel.
4. Separate logs panel wired to `log_event` and history API.

Acceptance:
1. Staff can complete end-to-end cowork interaction without TUI.

### W6: Staff Access Hardening
Files:
1. `/Users/sfw/Development/loom/src/loom/api/server.py`
2. `/Users/sfw/Development/loom/docs/CONFIG.md`
3. Deployment docs (new or existing ops docs)

Deliverables:
1. Documented deployment patterns for staff usage (reverse proxy auth, TLS, network boundaries).
2. Optional allowlist/token middleware if needed for direct exposure.

Acceptance:
1. Clear, tested deployment path exists for non-localhost staff access.

### W7: Test Coverage
Files:
1. `/Users/sfw/Development/loom/tests/test_api.py` (or split cowork API tests)
2. `/Users/sfw/Development/loom/tests/test_conversation_store.py`
3. New frontend tests under web app.

Deliverables:
1. Endpoint tests for cowork session lifecycle.
2. WebSocket event ordering and serialization tests.
3. Upload safety tests (traversal, size limits, overwrite behavior).
4. Log persistence/query tests.

Acceptance:
1. New cowork web surfaces are covered with deterministic tests.

### W8: Docs + Rollout
Files:
1. `/Users/sfw/Development/loom/docs/agent-integration.md`
2. `/Users/sfw/Development/loom/docs/tutorial.html`
3. `/Users/sfw/Development/loom/docs/CONFIG.md`

Deliverables:
1. Cowork web API contract docs.
2. Frontend setup and usage instructions.
3. Upload and logs semantics documented for operators.

Acceptance:
1. Teams can adopt without reading source code.

## Phase Plan
1. Phase A (Backend core): W1 + W2.
2. Phase B (Uploads + logs durability): W3 + W4.
3. Phase C (Frontend MVP): W5.
4. Phase D (Hardening + docs): W6 + W7 + W8.

## Primary Risks and Mitigations
1. Risk: Session memory growth for many concurrent users.
2. Mitigation: idle session eviction with lazy DB resume.
3. Risk: Log table growth over long-running sessions.
4. Mitigation: event selection policy + retention/compaction policy.
5. Risk: File upload abuse or path escape.
6. Mitigation: strict workspace confinement + upload size ceilings + safe filename normalization.
7. Risk: Staff exposure without auth controls.
8. Mitigation: keep local-only defaults; require authenticated reverse proxy for shared deployment.

## Open Decisions Before Build Starts
1. Frontend location and stack (`web/cowork` vs another convention).
2. Whether to add built-in API auth now or require reverse-proxy auth for v1.
3. Exact upload size limits for your staff workflows.
4. Whether logs panel should include model invocation diagnostics by default.

## Definition of Done (MVP)
1. A staff user can open the web app, create/resume a cowork session, and chat in real time.
2. The user can drag/drop files into workspace and receive clear success/failure results.
3. A separate logs panel shows live tool/upload/turn events and survives reconnect via log backfill.
4. Existing TUI cowork behavior remains unchanged and tests remain green.
