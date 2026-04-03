# Database Migrations

Loom uses explicit, versioned SQLite migrations for all schema evolution.

This document defines the required workflow for schema changes and the runtime
behavior for upgrades.

## Runtime Guarantees

1. Existing databases are upgraded in place before normal runtime starts.
2. Existing database upgrade failures are blocking errors (no silent fallback).
3. New database creation failures are blocking by default.
4. Ephemeral fallback requires explicit CLI opt-in (`--ephemeral`).
5. Migrations are ordered, idempotent, and verified after apply.

## Migration System Layout

- Existing-DB canonical schema: `src/loom/state/schema.sql`
- Fresh-install baseline schema: `src/loom/state/schema/base.sql`
- Migration runner: `src/loom/state/migrations/runner.py`
- Migration registry: `src/loom/state/migrations/registry.py`
- Migration steps: `src/loom/state/migrations/steps/*.py`
- Migration metadata table: `schema_migrations`

Current desktop/workspace-first tables added through migrations include:

- `workspaces`
- `workspace_settings`
- `conversation_run_links`

Recent authority/freshness metadata added through migrations includes:

- `tasks.state_snapshot_updated_at` for mirrored task-row freshness
- `cowork_sessions.session_state_through_turn` for checkpoint trust boundaries
- `cowork_sessions.chat_journal_through_turn` / `chat_journal_through_seq` for transcript coverage

`schema_migrations` stores:
- `id`
- `applied_at`
- `duration_ms`
- `checksum`
- `notes`

## Startup Behavior

On startup, `Database.initialize()` does the following:

1. Open the configured SQLite database.
2. Detect whether this is an existing user DB.
3. Ensure `schema_migrations` exists.
4. Apply pending migration steps in registry order.
5. Apply `schema.sql` idempotently for existing-DB baseline/index creation.
6. Fresh DB bootstrap uses `schema/base.sql`.
6. Verify schema postconditions via migration `verify()` hooks.

If startup fails on an existing DB, Loom exits with an actionable upgrade error.
If startup fails on a new DB path, Loom exits unless `--ephemeral` is provided.

## Operator Commands

Use the `loom db` command group:

- `uv run loom db status`: show applied and pending migrations
- `uv run loom db migrate`: apply pending migrations
- `uv run loom db doctor`: run schema verification plus authority/projection sanity checks
- `uv run loom db backup [--output <path>]`: create DB backup copy via SQLite
  online backup API

Recommended recovery flow:

1. `uv run loom db backup`
2. `uv run loom db doctor`
3. `uv run loom db migrate`
4. Retry normal Loom startup

When doctor/migrate succeeds after the authority-unification migration, old DBs keep
their existing data and gain explicit freshness/coverage markers rather than requiring
database deletion or clean-state rebuilds.

Doctor warnings now also flag cowork sessions that still have legacy chat-journal rows
without explicit coverage metadata, so operators can distinguish "schema is valid" from
"all projections are fully covered".

## Required Workflow for Schema Changes

Any schema change must include all of the following:

1. Update `src/loom/state/schema.sql` and `src/loom/state/schema/base.sql` as needed.
2. Add/update a migration step in `src/loom/state/migrations/steps/`.
3. Register the step in `src/loom/state/migrations/registry.py`.
4. Add/extend migration tests in `tests/`:
   - legacy-shape upgrade path
   - idempotency / repeated init
   - verification expectations
5. Update docs:
   - `docs/DB-MIGRATIONS.md` (if workflow/rules changed)
   - user-facing config/ops docs (`README.md`, `docs/CONFIG.md`, etc.)
   - `CHANGELOG.md`

## CI Guardrails

CI runs `scripts/check_db_migration_policy.py` to enforce policy:

1. `schema.sql` or `schema/base.sql` changes require migration step changes.
2. Migration step changes require registry changes.
3. `schema.sql` or `schema/base.sql` changes require docs + tests changes.

## Rules for Coding Agents and Contributors

1. Do not tell users to delete their DB as the normal upgrade path.
2. Do not add runtime code that depends on new DB fields without a migration.
3. Do not edit historical migration behavior casually; add a new migration step.
4. Always add/refresh old-schema upgrade tests for schema-affecting PRs.
5. Run migration-focused tests before marking a schema PR ready.
