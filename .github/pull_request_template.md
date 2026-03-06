## Summary

- 

## Validation

- [ ] `uv run ruff check src tests`
- [ ] `uv run pytest -v --tb=short`

## Schema Change Checklist (if applicable)

- [ ] I updated `src/loom/state/schema.sql` and `src/loom/state/schema/base.sql` as needed.
- [ ] I added/updated migration step(s) under `src/loom/state/migrations/steps/`.
- [ ] I updated `src/loom/state/migrations/registry.py`.
- [ ] I added/updated migration upgrade/idempotency tests.
- [ ] I updated docs/changelog (`docs/DB-MIGRATIONS.md`, README/config/agent docs, `CHANGELOG.md`).
