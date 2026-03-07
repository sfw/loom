# Telemetry Event Catalog

This document describes Loom's typed telemetry contract and lifecycle policy.

## Lifecycle Policy

Each event type in [`src/loom/events/types.py`](../src/loom/events/types.py) is explicitly marked as one of:

- `active`: expected to be emitted and test-referenced.
- `internal_only`: high-volume or diagnostic-only telemetry that may be omitted from operator UI surfaces.
- `deprecated`: retained for compatibility and scheduled for removal.

The source of truth is:

- `EVENT_LIFECYCLE`
- `ACTIVE_EVENT_TYPES`
- `INTERNAL_ONLY_EVENT_TYPES`

## Runtime Verbosity Modes

Operator sinks (`/tasks/{id}/stream`, `/tasks/{id}/tokens`, delegate/TUI progress surfaces)
apply one shared policy over effective runtime mode:

- `off`
- `active`
- `all_typed`
- `debug`

`internal_only` is accepted as a temporary input alias and normalized to `all_typed`.
Runtime override scope is process-local in the current phase.

Always-pass-through operator events (including `off`):

- `task_created`
- `task_run_acquired`
- `task_run_heartbeat`
- `task_run_recovered`
- `task_cancel_requested`
- `task_cancel_ack`
- `task_cancel_timeout`
- `task_paused`
- `task_resumed`
- `task_injected`
- `approval_requested`
- `ask_user_requested`
- `task_completed`
- `task_failed`
- `task_cancelled`

Diagnostics channel behavior:

- `telemetry_diagnostic` is typed `internal_only`.
- It is only delivered to operator sinks in `debug`.
- It is rate-limited and recursion-guarded in `EventBus`.

## Sink Matrix

| Sink class | Example sinks | Mode suppression |
| --- | --- | --- |
| Compliance | SQLite `events` persistence | Not suppressed by operator mode in phase A |
| Operator | SSE streams, delegate progress, TUI progress | Filtered by `should_deliver_operator(event_type, effective_mode)` |

## Troubleshooting by Sink/Mode

- Symptom: `token_streamed` is missing from SSE/delegate/TUI while runs still complete.
  Cause: expected in `off` mode; token streaming is suppressed outside passthrough set.
  Action: check effective mode via `GET /settings/telemetry` (or TUI `/telemetry`) and switch to `active`/`all_typed`/`debug`.
- Symptom: approvals or ask-user prompts appear stalled in operator views.
  Cause: usually not telemetry filtering; `approval_requested` and `ask_user_requested` are passthrough in all modes.
  Action: verify upstream task state and pending human-loop handlers, then inspect `events` table for corresponding records.
- Symptom: audit rows appear missing in `off`.
  Cause: should not happen in phase A because compliance persistence is mode-independent.
  Action: query SQLite `events` directly and inspect `telemetry_diagnostic` rows for persistence failures.
- Symptom: diagnostics are missing in operator sinks.
  Cause: `telemetry_diagnostic` is operator-visible only in `debug`, and may be rate-limited.
  Action: set mode to `debug`, then reproduce once with low event volume.

## Common Payload Contract

Event payloads are normalized in [`src/loom/events/bus.py`](../src/loom/events/bus.py) and include:

- `task_id`
- `timestamp`
- `event_id`
- `correlation_id`
- `sequence`
- `schema_version`
- `source_component`
- `run_id` (when available)

Event-family and event-specific required keys are defined in
[`src/loom/events/contracts.py`](../src/loom/events/contracts.py).

## Redaction Policy

Before dispatch and persistence, payloads are redacted for sensitive key families
(for example auth/token/secret/password/credential markers). URL fields are sanitized
to remove query-string and fragment content.

## Control-Plane Telemetry

Control-plane task actions are tracked with typed events:

- `task_paused`
- `task_resumed`
- `task_injected`
- `task_cancel_requested`
- `task_cancel_ack`
- `task_cancel_timeout`
- `steer_instruction`
- `telemetry_mode_changed`
- `telemetry_settings_warning`

## Verification Lifecycle Telemetry

Verification emits explicit lifecycle states:

- `verification_started`
- `verification_passed`
- `verification_failed`

And supporting audit events, including:

- `verification_outcome`
- `verification_contradiction_detected`
- `rule_failure_by_type`
- `claim_verification_summary`

## Webhook Delivery Telemetry

Webhook delivery is auditable via:

- `webhook_delivery_attempted`
- `webhook_delivery_succeeded`
- `webhook_delivery_failed`
- `webhook_delivery_dropped`

Webhook telemetry stores host/hash metadata only, never raw query strings.

## Database Migration Diagnostics

Migration lifecycle diagnostics are emitted as internal telemetry events:

- `db_migration_start`
- `db_migration_applied`
- `db_migration_verify_failed`
- `db_migration_failed`
- `db_schema_ready`

Diagnostic payloads include migration ID/phase context (when applicable),
error-class metadata, and actionable suggestion keys for recovery flows.

## Cowork Replay Boundary

Task-run telemetry (`events` table / `EventBus`) and cowork replay telemetry
(`cowork_chat_events` table) are intentionally distinct streams:

- Task-run telemetry tracks orchestration lifecycle, verification, remediation,
  and control-plane actions for auditable run reconstruction.
- Cowork replay telemetry tracks chat UX/state progression for session replay.

Cowork context-window telemetry is persisted in replay payloads and should be
treated as replay/audit metadata for conversation scope:

- `context_tokens`
- `context_messages`
- `omitted_messages`
- `recall_index_used`

These fields are part of cowork replay expectations and should not be mixed
with task-run lifecycle counters.

## Audit Query Examples

Example SQLite queries against `~/.loom/loom.db`:

```sql
-- Reconstruct a run's ordered event timeline.
SELECT task_id, run_id, sequence, event_type, timestamp
FROM events
WHERE run_id = 'run-abc123'
ORDER BY sequence ASC, id ASC;
```

```sql
-- Inspect terminal verification outcomes and reason codes for a task.
SELECT event_type,
       json_extract(data, '$.subtask_id') AS subtask_id,
       json_extract(data, '$.reason_code') AS reason_code,
       timestamp
FROM events
WHERE task_id = 'task-123'
  AND event_type IN ('verification_outcome', 'verification_failed', 'verification_passed')
ORDER BY sequence ASC, id ASC;
```

```sql
-- Review webhook delivery lifecycle for a task.
SELECT event_type,
       json_extract(data, '$.delivery_target_host') AS host,
       json_extract(data, '$.attempt') AS attempt,
       json_extract(data, '$.reason') AS reason,
       timestamp
FROM events
WHERE task_id = 'task-123'
  AND event_type LIKE 'webhook_delivery_%'
ORDER BY sequence ASC, id ASC;
```

## CI Drift Guard

Run the checker locally:

```bash
uv run python scripts/check_telemetry_coverage.py --report telemetry-coverage.json
```

CI fails when any `active` event is missing emission coverage or test references.
