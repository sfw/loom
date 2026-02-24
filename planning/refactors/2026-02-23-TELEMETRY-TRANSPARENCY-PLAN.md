# Telemetry Transparency Plan (2026-02-23)

## Objective
Increase runtime transparency in Loom event logs so operators can answer, from a single `.events.jsonl` stream:
1. What the system decided.
2. Why it decided it.
3. What data shape changed as a result.

Primary gap to close now:
- `web_fetch` file classification/parsing decisions are not first-class events in run logs, even though metadata exists in tool results and artifact manifests.

## Current State and Gap
Today, for fetched files:
1. `web_fetch` computes `content_kind`, handler, extraction stats, and artifact metadata.
2. Artifact manifests are written under `.loom_artifacts/.../manifest.jsonl`.
3. Run logs mostly show `tool_call_started` / `tool_call_completed` snapshots.

Missing in run logs:
1. Explicit ingest lifecycle events (classification, handler selection, persistence outcome).
2. Structured retention cleanup outcomes.
3. A compact, operator-friendly summary tying artifact ingest to compaction/overflow behavior.

## Design Principles
1. Event bus is the single telemetry transport.
2. Emit at orchestration boundaries, not inside parsing utilities.
3. Metadata only in logs: never dump raw extracted document text.
4. High signal over high volume: prefer milestone events to chatty internals.
5. Deterministic payloads with stable field names.
6. Backward-compatible rollout behind a flag, then default-on.

## Non-Goals
1. Full distributed tracing stack.
2. Logging every parser micro-step.
3. Logging sensitive payload bodies (document text, secrets, auth headers, raw binary).

## Target End State
A heavy-document run should make these questions trivial to answer from the run log:
1. Was a fetch treated as text, html, pdf, office, image, archive, or unknown binary?
2. Which handler processed it?
3. Was an artifact persisted, and what ref/path metadata was produced?
4. Did retention cleanup delete older artifacts?
5. Did compaction policy skip/compact and why?
6. Did overflow fallback run, and how much context was rewritten?

## Event Model (v1)

### 1) Artifact Ingest Events (new)
Add first-class event types:
1. `artifact_ingest_classified`
2. `artifact_ingest_completed`
3. `artifact_retention_pruned`
4. `artifact_read_completed`

Required payload fields (common):
1. `subtask_id`
2. `tool`
3. `url` (sanitized)
4. `content_kind`
5. `content_type`
6. `status` (`ok|error`)

Artifact-specific fields (where available):
1. `artifact_ref`
2. `artifact_workspace_relpath` (preferred over absolute path)
3. `size_bytes`
4. `declared_size_bytes`
5. `handler`
6. `extracted_chars`
7. `extraction_truncated`
8. `handler_metadata` (size-bounded)

Retention-specific fields:
1. `scopes_scanned`
2. `files_deleted`
3. `bytes_deleted`

### 2) Compaction/Overflow Transparency Events (new, narrow)
Add targeted decision events (not duplicating existing model-invocation telemetry):
1. `compaction_policy_decision`
2. `overflow_fallback_applied`

Required payload fields:
1. `subtask_id`
2. `pressure_ratio`
3. `policy_mode`
4. `decision` (`skip|compact_tool|compact_history|fallback_rewrite`)
5. `reason` (deterministic short code)

Overflow-specific fields:
1. `rewritten_messages`
2. `chars_reduced`
3. `preserved_recent_messages`

### 3) Run Summary Event (new)
Emit once at task end:
1. `telemetry_run_summary`

Payload fields:
1. `artifact_ingests`
2. `artifact_reads`
3. `artifact_retention_deletes`
4. `compaction_policy_decisions`
5. `overflow_fallback_count`
6. `compactor_warning_count`

## Emission Boundaries

### Where to emit (and why)
1. `src/loom/engine/runner.py`
- Has task/subtask context and event bus access.
- Emits ingest and decision events after tool completion/compaction decisions.

2. `src/loom/tools/read_artifact.py`
- Produces artifact-read outcomes; return structured data.
- Runner emits event using that result data.

3. `src/loom/ingest/artifacts.py`
- Keep storage/cleanup pure.
- Return cleanup stats to caller; caller decides whether to emit.

Do not emit directly from low-level handler classes to `.events.jsonl`.

## Data Safety and Redaction Policy
1. Never log extracted document text.
2. Prefer `artifact_workspace_relpath`; avoid absolute paths unless no relative path exists.
3. Keep `handler_metadata` bounded (max chars/keys).
4. Exclude auth headers, query params with credentials, and binary payload snippets.

## Rollout Strategy

### Phase 1: Additive, flag-gated
1. Introduce event types and emission paths.
2. Keep existing `tool_call_*` events untouched.
3. Gate with `limits.runner.enable_artifact_telemetry_events` (default `false`).

### Phase 2: Default on
1. Promote flag default to `true` after validation.
2. Keep flag for rollback.

### Phase 3: Optional verbosity controls
1. Add optional payload detail mode if needed (`standard|debug`) only after observing real log volume.

## Workstreams

### Workstream 1: Event Taxonomy + Schema Contract
Files:
1. `/Users/sfw/Development/loom/src/loom/events/types.py`
2. `/Users/sfw/Development/loom/docs/CONFIG.md`

Deliverables:
1. New event constants.
2. Payload field contract documented with examples.

Acceptance:
1. Event names and required fields stable and test-covered.

### Workstream 2: Artifact Ingest Telemetry
Files:
1. `/Users/sfw/Development/loom/src/loom/engine/runner.py`
2. `/Users/sfw/Development/loom/src/loom/tools/web.py`

Deliverables:
1. Emit `artifact_ingest_classified` and `artifact_ingest_completed` for `web_fetch`/`web_fetch_html` artifact paths.
2. Emit only when artifact metadata exists.

Acceptance:
1. A PDF fetch shows handler + artifact metadata in run log.
2. No raw extracted text appears in event payload.

### Workstream 3: Retention + Artifact Read Telemetry
Files:
1. `/Users/sfw/Development/loom/src/loom/engine/runner.py`
2. `/Users/sfw/Development/loom/src/loom/tools/read_artifact.py`
3. `/Users/sfw/Development/loom/src/loom/ingest/artifacts.py`

Deliverables:
1. Emit `artifact_retention_pruned` when cleanup deletes files.
2. Emit `artifact_read_completed` for `read_artifact` success/failure.

Acceptance:
1. Cleanup side effects are visible in run logs.
2. Read events include ref + handler + extraction stats.

### Workstream 4: Compaction/Overflow Decision Telemetry
Files:
1. `/Users/sfw/Development/loom/src/loom/engine/runner.py`

Deliverables:
1. Emit `compaction_policy_decision` with reason codes.
2. Emit `overflow_fallback_applied` with rewrite stats.

Acceptance:
1. Operators can explain compaction/overflow actions without reading code.

### Workstream 5: Delegate Log Forwarding Coverage
Files:
1. `/Users/sfw/Development/loom/src/loom/tools/delegate_task.py`

Deliverables:
1. Include new event types in observed event forwarding set.
2. Preserve current JSONL compatibility.

Acceptance:
1. New telemetry events appear in delegate-run `.events.jsonl` streams.

### Workstream 6: Tests + Docs + Rollout Validation
Files:
1. `/Users/sfw/Development/loom/tests/test_orchestrator.py`
2. `/Users/sfw/Development/loom/tests/test_web_tool.py`
3. `/Users/sfw/Development/loom/tests/test_ingest_router.py`
4. `/Users/sfw/Development/loom/tests/test_read_artifact_tool.py`
5. `/Users/sfw/Development/loom/tests/test_config.py`
6. `/Users/sfw/Development/loom/docs/CONFIG.md`
7. `/Users/sfw/Development/loom/docs/tutorial.html`

Deliverables:
1. Event payload assertions for each new event type.
2. Config flag parse/default tests.
3. Docs for operators.

Acceptance:
1. All telemetry additions validated in unit/integration tests.
2. Existing orchestration behavior unchanged.

## Proposed Config Additions
Under `[limits.runner]`:
1. `enable_artifact_telemetry_events = false` (rollout flag)
2. `artifact_telemetry_max_metadata_chars = 1200` (defensive payload bound)

Optional future key (defer unless needed):
1. `artifact_telemetry_detail_mode = "standard"` (`standard|debug`)

## Test Plan

### Unit
1. Artifact ingest event emitter redacts/bounds metadata.
2. Retention prune event emitted only when deletes > 0.
3. Compaction decision reason mapping deterministic.

### Integration
1. `web_fetch` PDF fixture produces `artifact_ingest_*` events with expected fields.
2. Overflow scenario emits `overflow_fallback_applied` once.
3. `read_artifact` emits `artifact_read_completed`.

### Regression
1. `tool_call_completed` payload contract unchanged.
2. No change to deliverable quality path.
3. No sensitive payload leakage.

## Success Metrics
1. 100% of artifact-producing `web_fetch` calls emit `artifact_ingest_completed`.
2. 100% of retention deletions emit `artifact_retention_pruned`.
3. Overflow fallback occurrences always include rewrite metrics.
4. Mean time-to-diagnose file ingest issues reduced (operator validation target: < 2 minutes from log only).

## Risks and Mitigations
1. Risk: log volume growth.
- Mitigation: milestone-only events + payload bounds + rollout flag.

2. Risk: accidental sensitive metadata leakage.
- Mitigation: explicit allowlist fields + tests for disallowed keys.

3. Risk: duplicated/contradictory signals with existing `model_invocation` events.
- Mitigation: decision events stay summary-level; low-level compactor internals remain in existing model events.

## Implementation Sequence (Suggested PRs)
1. PR-1: event types + runner artifact ingest emit + tests.
2. PR-2: retention/read events + delegate forwarding + tests.
3. PR-3: compaction/overflow decision events + summary event.
4. PR-4: config/docs rollout defaults and final regression pass.

## Exit Criteria
1. Run logs clearly show file handler selection and artifact metadata for ingest paths.
2. Retention and overflow behavior are visible without code inspection.
3. Telemetry remains concise, deterministic, and safe.
