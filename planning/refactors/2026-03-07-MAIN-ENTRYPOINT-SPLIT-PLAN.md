# 2026-03-07 Main Entrypoint Split Plan

## Scope
- Requested file: `src/loom/main.py` (not present in this checkout).
- Equivalent large entrypoint found and targeted: `src/loom/__main__.py` (~4,195 LOC).

## Goal
- Split the CLI entrypoint into a package-oriented command system while preserving:
  - `python -m loom`
  - `loom` command behavior
  - Click command names/options/help output
  - existing imports used by tests (`from loom.__main__ import ...`)

## Current Shape
- 80+ top-level functions in one module.
- Mixed concerns:
  - CLI bootstrap/context/config
  - TUI launch + DB init
  - task/server HTTP helpers
  - auth commands + auth profile commands
  - MCP commands + MCP auth commands
  - process/install/uninstall commands
  - DB commands + learned/setup commands

## Target Layout (Package-First)
- `src/loom/__main__.py`
  - Thin facade and runtime entrypoint (`main()` + `cli` re-export).
- `src/loom/cli/__init__.py`
  - Public CLI exports.
- `src/loom/cli/context.py`
  - Config/workspace/auth/MCP context helper functions.
- `src/loom/cli/persistence.py`
  - `PersistenceInitError`, `_init_persistence`, persistence bootstrap helpers.
- `src/loom/cli/http_tasks.py`
  - `_run_task`, `_check_status`, `_cancel_task`, task-id validation.
- `src/loom/cli/commands/root.py`
  - top-level `cli` group and base commands (`cowork`, `serve`, `run`, `status`, `cancel`, `models`, `mcp-serve`).
- `src/loom/cli/commands/auth.py`
  - `auth` group commands except profile subgroup.
- `src/loom/cli/commands/auth_profile.py`
  - `auth profile ...` subgroup.
- `src/loom/cli/commands/mcp.py`
  - `mcp` group commands (list/show/status/connect/...).
- `src/loom/cli/commands/mcp_auth.py`
  - `mcp auth ...` subgroup.
- `src/loom/cli/commands/process.py`
  - `process`, `process test`, install/uninstall/process listing.
- `src/loom/cli/commands/db.py`
  - `db` group (`status`, `migrate`, `doctor`, `backup`).
- `src/loom/cli/commands/maintenance.py`
  - `learned`, `reset-learning`, `setup`.

## Incremental Plan

### Phase 1: Scaffolding and Facade Safety
- Add `loom/cli/` package.
- Move shared helpers (context + persistence + HTTP helpers) first.
- Keep `src/loom/__main__.py` exporting existing names for test compatibility.
 - Keep `python -m loom` boot path intact (do not convert `__main__.py` into a package).

### Phase 2: Root Commands
- Move top-level non-group commands to `commands/root.py`.
- Register them through one central `build_cli()` function.

### Phase 3: Auth Command Family
- Move `auth` and `auth profile` commands into dedicated modules.
- Keep output text and JSON payload schemas unchanged.

### Phase 4: MCP Command Family
- Move `mcp` and `mcp auth` command implementations.
- Preserve runtime registry open/close lifecycle behavior.

### Phase 5: Process/DB/Maintenance Commands
- Move process/install/uninstall/db/learned/setup paths.
- Keep migration and doctor command semantics untouched.

### Phase 6: Shrink `__main__.py`
- Leave only:
  - `from loom.cli import cli, main`
  - very thin backward-compatible helper re-exports required by tests.

## Risks and Hardening
- Risk: Click registration order drift can change help output and command dispatch.
  - Hardening: snapshot `loom --help` and key subgroup help outputs before/after.
- Risk: context object (`ctx.obj`) contract drift.
  - Hardening: add CLI smoke tests for each major command family with same flags.
- Risk: test imports from `loom.__main__` break.
  - Hardening: temporary re-export shim for `_init_persistence`, `_check_status`, `_cancel_task`, `PersistenceInitError`, `cli`.

## Adherence to Prior Plan Standards
- Module-to-package conversion rule:
  - Not applicable to `__main__.py` because Python module entrypoint semantics require the file to remain `src/loom/__main__.py`.
- Facade import rule:
  - Internal CLI command modules must not import via `loom.cli.__init__`; import concrete sibling modules only.
- Revertable slices:
  - Keep each phase as a single PR slice (bootstrap/helpers, then command families one-by-one).
- Import/cycle guardrails:
  - Add import-cycle smoke tests for `loom.cli.commands.*`.
- Shared compatibility checks:
  - Extend a central import-contract suite to assert `loom.__main__.cli` and helper re-export availability for current tests.

## Test Strategy
- Existing:
  - `tests/test_cli.py`
  - `tests/test_error_handling.py`
  - `tests/test_tui.py` (covers `_init_persistence` imports)
- New (non-overlapping package):
  - `tests/cli/test_import_contracts.py`
  - `tests/cli/test_help_snapshots.py`
  - `tests/cli/test_context_contract.py`
  - `tests/cli/test_import_cycles.py`
  - `tests/test_cli_import_contracts.py` (central CLI import compatibility checks)

## Exit Criteria
- `src/loom/__main__.py` <= 250 LOC.
- All command families live under `src/loom/cli/commands/`.
- CLI help and command behavior parity validated.
- Existing CLI/TUI/error-handling tests pass.
