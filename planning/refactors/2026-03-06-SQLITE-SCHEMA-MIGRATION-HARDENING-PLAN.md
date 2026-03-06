# SQLite Schema Migration Hardening Plan (2026-03-06)

## Executive Summary
Loom currently initializes persistence by executing the full `schema.sql` and then applying targeted migrations. This ordering can fail on older databases when `schema.sql` contains indexes that reference columns introduced after the original table shape. The observed startup error (`no such column: sequence`) is a concrete example.

This plan introduces a durable, versioned migration subsystem with strict startup guarantees:
1. Existing databases are upgraded deterministically before runtime uses new schema fields.
2. Fresh databases still bootstrap quickly from canonical schema.
3. Migration failures are surfaced as explicit upgrade failures, not silently downgraded to ephemeral mode.
4. Future schema changes are required to ship with explicit upgrade steps and fixture-based upgrade tests.

## Incident Anchor
Observed startup output:
1. `Warning: database init failed: no such column: sequence`
2. `Warning: database unavailable, running in ephemeral mode.`

Downstream impact:
1. Persistence store unavailable.
2. TUI falls back to ephemeral execution.
3. `/run` may reach unbound delegation flows and fail during run execution.

Root cause class:
1. Schema upgrade ordering and policy gap, not data corruption.

## Why Current Behavior Fails
Current initialization flow in `Database.initialize()`:
1. `executescript(schema.sql)`
2. `_migrate_events_table()`
3. `_ensure_events_indexes()`

Failure mode:
1. Legacy DB already has `events` table without `sequence`.
2. `schema.sql` attempts `CREATE INDEX ... events(sequence)` before migration adds `sequence`.
3. Initialization aborts before migration can run.

Policy gap:
1. Startup catches init exception and falls back to ephemeral mode.
2. Upgrade failure is treated as optional rather than blocking for persistence-backed features.

## Goals
1. Make schema upgrades explicit, versioned, and idempotent.
2. Ensure migrations run before any SQL that depends on newly-added columns.
3. Prevent silent persistence downgrade when an existing DB fails upgrade.
4. Guarantee deterministic behavior across app upgrades.
5. Add CI coverage that boots new code against old DB snapshots.

## Non-Goals
1. Replacing SQLite.
2. Rewriting all persistence APIs.
3. Performing non-essential data normalization in this pass.

## Invariants
1. Existing DB + newer app must either migrate successfully or fail startup with a clear upgrade error.
2. No migration may require manual DB deletion as primary upgrade path.
3. Migrations are idempotent and safe to re-run.
4. Migration order is deterministic and audited.
5. Fresh installs and upgraded installs converge to the same effective schema.

## Hardening Requirements
1. Documentation rollout is mandatory; migration behavior must be documented for users, operators, and contributors.
2. Agent/contributor guardrails must be enforceable via CI and PR checks, not prose only.
3. Release process must include migration validation before release artifacts are cut.
4. Schema ownership workflow must be explicit ("how to change schema" is contractually defined).
5. Fallback policy must define exact mode semantics and CLI UX contract (`--ephemeral` behavior and messaging).

## Target Architecture

## 1) Versioned Migration Framework
Add a migration runner under `src/loom/state/migrations/`:
1. `registry.py`: ordered list of migration definitions.
2. `runner.py`: orchestration, locking, transaction boundaries, status recording.
3. `steps/*.py`: one file per migration unit.

Migration record storage:
1. `schema_migrations` table with:
   1. `id TEXT PRIMARY KEY`
   2. `applied_at TEXT NOT NULL`
   3. `duration_ms INTEGER NOT NULL`
   4. `checksum TEXT NOT NULL`
   5. `notes TEXT DEFAULT ''`

Migration definition contract:
1. `id`: stable, monotonic identifier.
2. `description`: human-readable purpose.
3. `apply(conn)`: additive SQL and data backfills.
4. `verify(conn)`: postcondition checks for required columns/indexes/tables.

## 2) Startup Decision Tree
On database startup:
1. Open DB with `BEGIN IMMEDIATE` migration lock.
2. Ensure `schema_migrations` table exists.
3. Detect DB state:
   1. Fresh DB (no Loom tables): apply canonical base schema, stamp bootstrap marker.
   2. Existing DB: run pending migrations in registry order only.
4. Run global schema verification.
5. Commit and continue startup.

If migration fails:
1. Roll back transaction.
2. Emit actionable failure (migration id + reason + next steps).
3. Abort persistence startup.
4. Do not auto-fallback to ephemeral mode for existing DBs unless explicitly requested via opt-in flag.

## 3) Canonical Schema Split
Split schema responsibilities:
1. `schema/base.sql`: only baseline table/index definitions safe for fresh installs.
2. Incremental changes: migration steps only.
3. Avoid placing indexes in base schema that reference columns introduced by later migration steps unless those columns are also part of base fresh schema and guaranteed to exist on fresh create.

Practical rule:
1. For existing-table evolution, keep dependent index creation in the same migration step as column addition.

## 4) Strict Upgrade Policy
Differentiate failure types:
1. Existing DB migration failure: hard error, startup blocked.
2. New DB creation failure (permissions/path): explicit startup error unless user opts into `--ephemeral`.
3. Optional developer fallback for local experimentation can remain behind an explicit switch, not default behavior.

## Documentation Update Requirements
This refactor must ship with synchronized documentation updates. Treat docs as part of the feature, not follow-up work.

Required updates:
1. [README.md](/Users/sfw/Development/loom/README.md)
   1. Add "Database Upgrades" section describing automatic migrations, failure behavior, and recovery commands.
2. [docs/CONFIG.md](/Users/sfw/Development/loom/docs/CONFIG.md)
   1. Document any new migration/fallback config flags and startup policy.
3. [docs/agent-integration.md](/Users/sfw/Development/loom/docs/agent-integration.md)
   1. Document migration expectations for agent-driven workflows and non-interactive runs.
4. [docs/2026-02-28-SYSTEM-TECHNICAL-DESIGN.md](/Users/sfw/Development/loom/docs/2026-02-28-SYSTEM-TECHNICAL-DESIGN.md)
   1. Update persistence architecture and startup lifecycle diagrams to include migration runner.
5. [CHANGELOG.md](/Users/sfw/Development/loom/CHANGELOG.md)
   1. Add explicit upgrade notice and operational notes.
6. New developer-focused guide:
   1. `docs/DB-MIGRATIONS.md` (or `docs/development/db-migrations.md`) describing authoring rules, testing, and rollback expectations.

Documentation acceptance criteria:
1. No schema-affecting PR merges without doc deltas.
2. Commands and failure messages in docs match actual CLI/runtime output.

## Coding-Agent and Contributor Guardrails
Schema evolution must be enforceable by default for both humans and coding agents.

Policy:
1. Database schema updates must be introduced via migration steps, not by editing runtime assumptions only.
2. Any PR that changes persistence schema must include:
   1. migration step file
   2. migration verification logic
   3. upgrade fixture test updates
   4. documentation updates

Enforcement mechanisms:
1. Add repository policy guidance for agents/contributors (preferred in root `AGENTS.md` when present; otherwise contributor docs + PR template).
2. Add CI check script:
   1. if `src/loom/state/schema.sql` changes, require corresponding `src/loom/state/migrations/steps/*` change.
   2. require at least one migration-focused test update.
3. Add PR template checklist item: "Schema changes included migration + fixture upgrade tests + docs."
4. Add unit test guard that fails if migration registry latest version is behind schema expectations.

Agent-facing instruction set (to include in contributor/agent docs):
1. Never ask users to delete DB as normal upgrade path.
2. Never add new DB column usage in runtime code without a migration entry.
3. Always add an old-schema fixture reproducing the pre-change state.
4. Always run migration tests locally before proposing schema PR completion.

## Extracted Migration Units from Current Pending Work
These pending schema deltas should be captured as explicit migrations, independent of telemetry/business logic:

1. `events` evolution:
   1. add columns: `run_id`, `event_id`, `sequence`, `source_component`, `schema_version`
   2. add indexes:
      1. `idx_events_task_sequence`
      2. `idx_events_run_sequence`
      3. `idx_events_event_id` (partial unique on non-empty `event_id`)

2. durable ask-user table:
   1. create `task_questions`
   2. create supporting indexes including partial unique `idx_task_questions_active_scope`

3. validity lineage tables:
   1. `artifact_claims`
   2. `claim_evidence_links`
   3. `claim_verification_results`
   4. `artifact_validity_summaries`
   5. required indexes

Each migration step must include:
1. existence checks (table/column/index) via `PRAGMA table_info` + `sqlite_master`.
2. deterministic verify assertions after apply.

## Migration Strategy for Legacy Databases
For DBs created before these columns:
1. Add missing columns with `ALTER TABLE`.
2. Backfill deterministic defaults where needed:
   1. `sequence`: default `0` for historical rows.
   2. `event_id`: default empty string unless backfill policy introduced later.
3. Create new indexes only after columns exist.

No destructive operations in this wave:
1. no table rebuild.
2. no column drops.
3. no type rewrites.

## CLI and Operator Tooling
Add `loom db` command group:
1. `loom db status`:
   1. DB path
   2. current migration state
   3. pending migrations
   4. schema health checks
2. `loom db migrate`:
   1. apply pending migrations
   2. print applied steps and timings
3. `loom db doctor`:
   1. schema verification
   2. index consistency checks
4. `loom db backup`:
   1. online copy to timestamped file before manual intervention

Startup behavior:
1. app startup may invoke same runner directly.
2. CLI commands provide explicit recovery path.

## Observability and Diagnostics
Emit structured migration diagnostics:
1. `db_migration_start`
2. `db_migration_applied`
3. `db_migration_verify_failed`
4. `db_migration_failed`
5. `db_schema_ready`

Include fields:
1. migration id
2. duration
3. db path
4. error class
5. actionable suggestion key

## Testing Plan

## 1) Fixture Matrix
Create upgrade fixtures representing historical DB shapes:
1. `v1_initial` (events without `sequence` and companion columns)
2. `v2_pre_task_questions`
3. `v3_pre_validity_tables`
4. latest

Test each fixture through current startup:
1. migration succeeds
2. verification passes
3. core reads/writes work

## 2) Regression Tests
1. `test_initialize_migrates_legacy_events_before_sequence_indexes`
2. `test_existing_db_migration_failure_blocks_startup`
3. `test_new_db_bootstrap_sets_latest_migration_state`
4. `test_migrations_idempotent_on_second_run`
5. `test_db_status_reports_pending_and_applied_migrations`

## 3) Failure Injection Tests
1. simulate locked DB.
2. simulate malformed schema with partial prior edits.
3. verify rollback preserves pre-migration integrity.

## 4) Performance Checks
1. time migration steps on representative DB sizes.
2. confirm no pathological startup regression.

## Implementation Plan

## Phase P0: Immediate Safety Hotfix
1. Ensure any SQL referencing newly-added columns does not run before that column exists.
2. Stop silent ephemeral fallback for existing DB migration failures.
3. Add targeted test for `events.sequence` upgrade path.

Exit criteria:
1. legacy DB with old `events` schema upgrades successfully.
2. startup no longer reports `no such column: sequence`.

## Phase P1: Migration Subsystem Foundation
1. introduce `schema_migrations` metadata table.
2. implement registry/runner.
3. move current ad hoc migration logic into named steps.
4. wire startup to runner.

Exit criteria:
1. all startup schema upgrades route through one runner.
2. migration status queryable.

## Phase P2: CLI + Diagnostics + Policy Hardening
1. add `loom db` commands.
2. add structured migration events/logging.
3. enforce explicit-flag-only ephemeral fallback.
4. add CI and PR-template guardrails for schema-change policy.
5. publish developer/agent migration authoring guide.

Exit criteria:
1. operators can diagnose and run migrations without source edits.
2. migration failures are actionable.
3. schema-affecting PRs cannot bypass migration policy checks.

## Phase P3: Historical Fixture Coverage
1. add committed DB fixtures for prior schema versions.
2. add CI jobs validating upgrade path from each fixture.
3. gate merges for schema-touching PRs on upgrade tests.

Exit criteria:
1. schema-evolution regressions are caught pre-merge.

## Code Ownership and Change Boundaries
Primary areas:
1. `src/loom/state/memory.py`
2. `src/loom/state/schema.sql` (or split base schema file)
3. `src/loom/state/migrations/*` (new)
4. `src/loom/__main__.py` startup policy updates
5. `tests/test_memory.py` + dedicated migration tests
6. CLI tests for `loom db` commands

Separation requirement:
1. Schema migration PRs should be isolated from telemetry/runtime feature logic where possible.

## Risks and Mitigations
1. Risk: partial migration on crash.
   1. Mitigation: transactional steps with `BEGIN IMMEDIATE` and rollback.
2. Risk: hidden legacy variants not in fixtures.
   1. Mitigation: add introspection-based guards and broaden fixture corpus.
3. Risk: startup hard-fail may surprise users currently relying on fallback.
   1. Mitigation: clear error text + explicit `--ephemeral` escape hatch.
4. Risk: index creation time on large DBs.
   1. Mitigation: measure timings, display progress, keep migrations additive.

## Success Criteria
1. Existing user DBs upgrade in place across releases with no manual deletion.
2. No startup path produces `no such column` for known migrated fields.
3. Persistence-required features do not run under accidental ephemeral fallback.
4. Every schema change in future releases ships with:
   1. named migration step
   2. verification logic
   3. fixture upgrade test
   4. required documentation updates
   5. passing schema-policy CI guardrails

## Open Questions
1. Should ephemeral fallback be completely disabled by default for all TUI launches, or only disabled when an existing DB file is present?
2. Do we want `PRAGMA user_version` only, `schema_migrations` only, or both (user_version mirror + detailed table)?
3. Should we backfill non-empty deterministic `event_id` for historical events now, or defer until an event replay requirement appears?
4. Where should durable coding-agent policy live in this repo: root `AGENTS.md`, contributor docs, or both?

## Proposed Immediate Next Action
Implement P0 as a focused fix branch:
1. reorder/guard schema application so legacy `events` tables can be migrated before sequence-dependent index creation.
2. add regression test reproducing old-DB failure.
3. block silent ephemeral fallback for upgrade failures on existing DBs.
4. add initial schema-policy CI check and migration-authoring doc scaffold.
