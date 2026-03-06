# AGENTS.md

Repository-specific guidance for coding agents and contributors.

## Database Schema Changes

Treat schema evolution as a migration-first workflow.

Required for any DB schema change:

1. Update `src/loom/state/schema.sql` and `src/loom/state/schema/base.sql`
   to reflect the latest canonical + fresh-bootstrap shape.
2. Add/modify migration step files under `src/loom/state/migrations/steps/`.
3. Register steps in `src/loom/state/migrations/registry.py`.
4. Add migration/upgrade tests in `tests/`.
5. Update migration/operator docs and changelog.

Do not:

- Ask users to delete their database as a normal upgrade path.
- Add runtime usage of new columns/tables without migration coverage.
- Rely on silent fallback-to-ephemeral for existing DB upgrade failures.

Use these commands during development and debugging:

- `uv run loom db status`
- `uv run loom db migrate`
- `uv run loom db doctor`
- `uv run loom db backup`

Policy enforcement:

- CI runs `scripts/check_db_migration_policy.py` and will fail schema PRs that
  do not include required migration/test/docs updates.
