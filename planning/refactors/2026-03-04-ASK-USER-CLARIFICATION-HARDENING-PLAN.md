# Ask User Clarification Hardening Plan (2026-03-04)

## Executive Summary
`ask_user` works in cowork but is effectively non-interactive in process/task execution. The production fix is to make clarification a first-class runtime primitive for runs, not a placeholder tool response.

This plan introduces:
1. `ask_user` schema v2 for structured choices and custom responses.
2. A durable `QuestionManager` runtime (parallel to approvals) for blocking question/answer flow in orchestrated runs.
3. TUI and API answer paths for pending questions.
4. Explicit executor base-prompt guidance so models ask instead of guessing.
5. Rollout guards, telemetry, and failure-mode hardening.

## Problem Statement
Current behavior leaves runs/processes with weak clarification semantics:
1. The model can call `ask_user`, but in non-cowork paths the tool returns placeholder text.
2. Runner does not pause subtask execution awaiting a real answer.
3. `/run` progress handling has no question lifecycle support.
4. Executor prompt does not explicitly require `ask_user` when critical information is missing.

Result: models either guess, fail late, or continue without explicit user direction.

## Current-State Audit (Code-Backed)
1. `src/loom/tools/ask_user.py` only supports `question` plus optional `options: list[str]`.
2. `src/loom/engine/runner.py` executes all tools through generic registry execution and has no `ask_user` interception.
3. `src/loom/tui/app.py` cowork flow handles `ask_user` (`_handle_ask_user`) but `/run` path (`_execute_process_run`, `_on_process_progress_event`) has no question request/answer handshake.
4. `src/loom/tools/delegate_task.py` streams orchestrator events, but no `ask_user_*` events exist today.
5. `src/loom/prompts/templates/executor.yaml` says “Do NOT substitute or guess…”, but does not explicitly instruct the model to call `ask_user`.
6. `tests/test_ask_user.py` and cowork tests exist; no end-to-end run/process question flow tests exist.

## Goals
1. Clarifications during runs/processes must be truly interactive and block execution when needed.
2. Question UX must support single-choice, multi-choice, and custom free-text answers.
3. Question state must be durable enough for crash/recovery and API clients.
4. Prompt guidance must explicitly teach when and how to use `ask_user`.
5. Backward compatibility with existing `options: list[str]` calls must be preserved.

## Non-Goals
1. Building a full conversational agent framework beyond focused clarification.
2. Replacing existing approval gating logic.
3. Reworking cowork conversation architecture.

## Decision: Recommended Architecture
Use an event-driven `QuestionManager` with explicit lifecycle and durable storage.

Why this is the best fit:
1. Aligns with existing `ApprovalManager` pattern and event bus architecture.
2. Supports TUI, API, and future web clients uniformly.
3. Provides deterministic pause/resume semantics inside runner loops.
4. Handles unattended policies explicitly (`block`, `timeout_default`, `fail_closed`).
5. Avoids brittle prompt-only mitigation.

## Detailed Design

## Design Tightening (Critique-Driven)
The following constraints are required for a production-safe implementation.

1. Ownership and wiring must be explicit.
2. `QuestionManager` is owned by `Orchestrator` (same lifecycle as `ApprovalManager`).
3. `SubtaskRunner` receives `question_manager` via constructor injection.
4. API engine exposes the same manager instance used by active orchestrators for answer resolution.
5. `delegate_task` progress subscriptions include `ask_user_*` events so `/run` gets live question prompts.

1. Task status semantics must avoid breaking current control flow.
2. MVP does not introduce a new `TaskStatus` enum value.
3. While a question is pending, task status remains `executing` and waiting is handled inside runner/question manager await.
4. Pause/cancel controls must still preempt question waits.
5. Optional future state (`waiting_user_input`) can be added only after end-to-end status compatibility tests.

1. Question identity must be deterministic and idempotent.
2. `question_id` should be derived from `(task_id, subtask_id, tool_call_id, retry_attempt)` when tool call id exists.
3. If no tool-call ID exists, generate UUID and persist immediately.
4. Replayed duplicate request with same deterministic key returns existing pending question instead of creating a new row.

1. Waiting contract must be cancellation-safe.
2. `request_question` wait loop checks answer event plus task control window every 100ms.
3. If task becomes cancelled, resolve question as `cancelled` and unblock runner.
4. If task paused, wait continues but remains resumable without losing pending question.

1. Clarification loops must be bounded.
2. Add per-subtask cap: `execution.ask_user_max_questions_per_subtask` (default 3).
3. Add minimum spacing: `execution.ask_user_min_seconds_between_questions` (default 10s).
4. Exceeding caps returns deterministic tool failure with guidance to proceed using best explicit assumption + risk note.

## 1) AskUser Contract v2
Upgrade `ask_user` request schema to structured forms:
1. `question: str` (required).
2. `question_type: "free_text" | "single_choice" | "multi_choice"` (default `free_text`).
3. `options: [{id: str, label: str, description?: str}]`.
4. `allow_custom_response: bool` (default `true`).
5. `min_selections: int` and `max_selections: int` for multi-choice.
6. `context_note: str` (short reason shown to user).
7. `urgency: "low" | "normal" | "high"`.
8. `default_option_id: str` (optional for timeout-default policy).

Normalize legacy payloads:
1. If `options` is `list[str]`, map to `{id, label}` entries.
2. If `question_type` missing and options exist, default to `single_choice`.
3. Preserve existing simple call path to avoid breaking current model behavior.

Standardized answer payload:
1. `question_id`.
2. `response_type: "text" | "single_choice" | "multi_choice" | "cancelled" | "timeout"`.
3. `selected_option_ids`.
4. `selected_labels`.
5. `custom_response`.
6. `answered_at`.
7. `source: "tui" | "api" | "policy_default"`.

## 2) Question Lifecycle and Runtime Manager
Add `src/loom/recovery/questions.py` with:
1. `QuestionRequest` dataclass.
2. `QuestionAnswer` dataclass.
3. `QuestionManager` class with:
4. `request_question(task_id, subtask_id, request) -> QuestionAnswer`.
5. `answer_question(task_id, question_id, answer_payload) -> bool`.
6. `list_pending_questions(task_id) -> list`.
7. timeout and cancellation resolution.

Lifecycle states:
1. `pending`.
2. `answered`.
3. `timeout`.
4. `cancelled`.

Critical runtime rules:
1. One active pending question per `(task_id, subtask_id)` in MVP.
2. Idempotent answer handling (duplicate answer submissions ignored after first commit).
3. While waiting, cancellation and pause signals remain honored.
4. Timeout behavior depends on policy.

## 3) Runner Integration (Core Fix)
Runner change in `src/loom/engine/runner.py`:
1. Detect `tc.name == "ask_user"` before generic tool execution.
2. Convert tool args to normalized `QuestionRequest`.
3. Call `QuestionManager.request_question(...)` and await resolved answer.
4. Serialize answer back as `ToolResult` so model sees concrete response.
5. Persist answer to memory as `entry_type="user_instruction"` and optional `decision` when structured option chosen.

Behavior policy from config:
1. `execution.ask_user_policy = "block" | "timeout_default" | "fail_closed"`.
2. `execution.ask_user_timeout_seconds`.
3. `execution.ask_user_max_pending_per_task`.
4. `execution.ask_user_default_response`.
5. `execution.ask_user_max_questions_per_subtask`.
6. `execution.ask_user_min_seconds_between_questions`.

Timeout-default validity rules:
1. For `single_choice`, default must map to an existing option id.
2. For `multi_choice`, default must satisfy `min_selections` and `max_selections`.
3. For `free_text`, default must be a non-empty string.
4. Invalid defaults force `fail_closed` behavior for that request.

## 4) Events and Progress Streaming
Add new event constants in `src/loom/events/types.py`:
1. `ASK_USER_REQUESTED`.
2. `ASK_USER_ANSWERED`.
3. `ASK_USER_TIMEOUT`.
4. `ASK_USER_CANCELLED`.

Emit these from `QuestionManager` and include:
1. `task_id`.
2. `subtask_id`.
3. `question_id`.
4. normalized question payload.
5. answer metadata (on answered/timeout/cancelled).

Update `src/loom/tools/delegate_task.py` event subscriptions to include new events so `_progress_callback` receives question events in `/run`.

## 5) TUI Process-Run Clarification UX
Update `src/loom/tui/app.py` process event handling:
1. On `ask_user_requested`, open an ask-user modal scoped to run.
2. Support:
3. single-select options,
4. multi-select options,
5. custom text response when enabled.
6. Submit answer through question-resolution hook.
7. Render question/answer in run transcript and chat replay.

Update `src/loom/tui/screens/ask_user.py`:
1. accept structured options and mode flags.
2. validate required selection counts.
3. support option descriptions.
4. preserve keyboard-first flow.

## 6) API Clarification Endpoints
Add endpoints in `src/loom/api/routes.py` and schemas in `src/loom/api/schemas.py`:
1. `GET /tasks/{task_id}/questions` returns pending/active questions.
2. `POST /tasks/{task_id}/questions/{question_id}/answer` resolves with single/multi/custom payload.

Requirements:
1. idempotent response semantics.
2. 404 for unknown question.
3. 409 for non-pending question.
4. optional metadata (`answered_by`, `client_id`) for audit.
5. return current question status on duplicate answer submissions instead of failing.

## 7) Durability and Recovery
Add table in `src/loom/state/schema.sql`:
1. `task_questions` with fields:
2. `question_id`, `task_id`, `subtask_id`, `status`.
3. `request_payload`, `answer_payload` (JSON).
4. `created_at`, `updated_at`, `resolved_at`, `timeout_at`.

Add persistence APIs in `src/loom/state/memory.py`:
1. upsert pending question.
2. resolve question.
3. list pending questions by task.

Recovery behavior:
1. On restart, pending questions are rehydrated.
2. Runs resume in waiting state until question is answered or policy resolves timeout.
3. Durable task-run lease handoff must not auto-resolve pending questions; only explicit answer/timeout/cancel may resolve.

## 8) Prompt Hardening (Base Prompt Requirement)
Update `src/loom/prompts/templates/executor.yaml` with explicit clarification protocol:
1. If critical inputs/constraints are missing, call `ask_user` before proceeding.
2. Prefer `single_choice` with 2-5 concrete options when the decision can be bounded.
3. Allow custom response unless policy forbids.
4. Ask one high-value question at a time.
5. Never fabricate assumptions for safety-critical or goal-defining unknowns.

Optional prompt alignment:
1. Add concise hints in planner/replanner templates to propagate clarification behavior earlier when uncertainty is discovered.

## 9) Safeguards and Abuse Controls
1. Enforce max question length and option count.
2. Enforce max pending questions per task to prevent spam loops.
3. Detect repeated near-duplicate questions and fail with guidance after threshold.
4. Reject malformed answer payloads with explicit validation errors.

## 10) Telemetry and SLOs
Add metrics:
1. `ask_user_requested_total`.
2. `ask_user_answered_total`.
3. `ask_user_timeout_total`.
4. `ask_user_cancelled_total`.
5. `ask_user_answer_latency_ms` (p50/p95/p99).
6. `ask_user_resolution_outcome_ratio` by policy and surface (`tui`, `api`).
7. `% subtasks blocked due to missing context that were resolved by ask_user`.

Rollout gates:
1. timeout rate < 5% for interactive runs (p7d).
2. median answer latency < 45s in TUI interactive runs.
3. duplicate-pending-question rate < 1%.
4. no deadlock incidents in soak tests (24h run with pause/cancel/resume chaos).

## Implementation Workstreams
1. Workstream A: Tool schema v2 and normalization.
Files: `src/loom/tools/ask_user.py`, `tests/test_ask_user.py`.
2. Workstream B: QuestionManager + events + persistence.
Files: `src/loom/recovery/questions.py`, `src/loom/events/types.py`, `src/loom/state/schema.sql`, `src/loom/state/memory.py`.
3. Workstream C: Runner interception and answer-to-tool-result bridge.
Files: `src/loom/engine/runner.py`, runner/orchestrator tests.
4. Workstream D: Delegate streaming + `/run` TUI integration.
Files: `src/loom/tools/delegate_task.py`, `src/loom/tui/app.py`, `src/loom/tui/screens/ask_user.py`, `tests/test_tui.py`.
5. Workstream E: API question endpoints.
Files: `src/loom/api/schemas.py`, `src/loom/api/routes.py`, `tests/test_api.py`.
6. Workstream F: Prompt updates and prompt tests.
Files: `src/loom/prompts/templates/executor.yaml`, optional planner/replanner templates, `tests/test_prompts.py`.

## Test Plan
1. Unit tests:
2. schema normalization, option validation, policy timeout behavior.
3. question manager state transitions and idempotency.
4. Integration tests:
5. runner blocks on `ask_user`, resumes after answer.
6. `/run` receives `ask_user_requested`, modal answer resolves run.
7. API answer endpoint unblocks pending question.
8. Recovery tests:
9. restart during pending question rehydrates and resumes correctly.
10. Regression tests:
11. cowork existing `ask_user` flow unchanged.
12. legacy `options: list[str]` still works.

## Rollout Strategy
1. Phase 0: ship behind flags, disabled by default.
2. Phase 1: enable prompt hardening + schema v2 normalization only.
3. Phase 2: enable runtime blocking in TUI interactive runs.
4. Phase 3: enable API endpoints and durability.
5. Phase 4: default-on for all process runs after SLO stability.

Rollback contract:
1. Disable `execution.ask_user_runtime_blocking_enabled` to revert to current non-blocking placeholder behavior.
2. Keep schema v2 normalization enabled during rollback to preserve compatibility.
3. Preserve pending `task_questions` rows for later replay or manual resolution.

Feature flags:
1. `execution.ask_user_v2_enabled`.
2. `execution.ask_user_runtime_blocking_enabled`.
3. `execution.ask_user_durable_state_enabled`.
4. `execution.ask_user_api_enabled`.

## Acceptance Criteria
1. In `/run` and process execution, `ask_user` causes a real pause and resumes only after answer or explicit policy resolution.
2. Multi-choice and custom response paths are supported end-to-end (tool schema, TUI, API, runner).
3. Executor base prompt explicitly instructs clarification behavior and reduces silent guessing on critical unknowns.
4. Pending questions survive restart when durability flag is enabled.
5. Telemetry shows measurable question lifecycle outcomes and latency.
6. Cowork ask-user behavior remains backward compatible.

## Risks and Mitigations
1. Risk: deadlocks when waiting for answer.
Mitigation: explicit timeout policy, cancel handling, and pending-question caps.
2. Risk: repeated low-value clarification loops.
Mitigation: prompt rule “one high-value question at a time” plus duplicate-question throttling.
3. Risk: recovery inconsistency for in-flight questions.
Mitigation: durable `task_questions` source of truth and idempotent answer resolution.
