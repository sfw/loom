# 2026-03-07 Pyproject Version Source of Truth Plan

## Scope

- Standardize Loom package versioning so `[project].version` in `pyproject.toml` is the sole authoritative value.
- Eliminate version drift across runtime outputs and primary documentation.
- Explicitly exclude planning documents from scope.

## Goal

- One edit location for version bumps: `pyproject.toml`.
- All programmatic outputs report the same value.
- Primary docs examples show the same value.
- CI enforces consistency so regressions fail fast.

## Non-Goals

- Reworking changelog historical release entries.
- Introducing git-tag-derived versioning (e.g., setuptools-scm) in this pass.
- Broad documentation system redesign beyond version token standardization.

## Current State Summary

Authoritative candidate:
- `pyproject.toml` sets `version = "0.1.0"`.

Runtime version paths:
- `src/loom/__init__.py` hardcodes `__version__ = "0.1.0"`.
- CLI and health outputs import `__version__`.
- Some runtime surfaces hardcode `"0.1.0"` directly (`api/server.py`, `integrations/mcp_tools.py`, `tui/app/process_runs/adhoc.py`).

Primary docs paths with hardcoded version examples:
- `INSTALL.md`
- `docs/tutorial.html`

Drift indicator:
- `CHANGELOG.md` includes historical releases up to `0.4.0` while runtime/package metadata still report `0.1.0`.

## Design Decision

Use `pyproject.toml` as the only authority and derive runtime value from installed package metadata (which is generated from `pyproject.toml` during build/install).

Mechanism:
- Add a dedicated version helper module that resolves version via `importlib.metadata.version("loom")`.
- Add a source-tree fallback that reads `pyproject.toml` when metadata is unavailable (tests or direct source execution).
- Re-export that value as `loom.__version__` for backward compatibility.

Why this design:
- Preserves your preferred authority (`pyproject.toml`).
- Avoids duplicate hardcoded constants.
- Works in both installed and in-repo dev execution modes.

## Target Architecture

- `pyproject.toml`
  - remains the single editable version source.
- `src/loom/version.py` (new)
  - `get_version()` function (cached), returns version from metadata or pyproject fallback.
  - `__version__` constant bound from `get_version()`.
- `src/loom/__init__.py`
  - imports and exports `__version__` from `loom.version`.
- All runtime emitters import `loom.__version__` or `loom.version.__version__`; no literal semver strings.

## Detailed Implementation Plan

### Phase 1: Introduce Version Resolver

- Add `src/loom/version.py` with:
  - `importlib.metadata.version("loom")` primary path.
  - `PackageNotFoundError` fallback to parse `pyproject.toml` with `tomllib`.
  - stable path resolution from `src/loom/version.py` to repo root.
  - `functools.lru_cache` to avoid repeated file/metadata reads.
- Update `src/loom/__init__.py` to re-export resolver-backed `__version__`.

Deliverable:
- Version can be consumed from `loom.__version__` without direct literals.

### Phase 2: Programmatic Output Standardization

Replace hardcoded literals with imported version in:
- `src/loom/api/server.py` (`FastAPI(... version=...)`)
- `src/loom/integrations/mcp_tools.py` (both `clientInfo.version` payloads)
- `src/loom/tui/app/process_runs/adhoc.py` (`serialize_process_for_package` payload)

Maintain existing imports/serialization contracts except value source.

Deliverable:
- All runtime version output paths use a single derived source.

### Phase 3: Primary Documentation Standardization

Options (choose one and enforce consistently):

Option A (recommended): tokenized docs + render/check
- Replace explicit version examples in primary docs with `{{LOOM_VERSION}}` token.
- Add a lightweight render/check step used in CI/release.

Option B (minimum-change): direct literal sync + CI check
- Keep literals in docs but enforce exact match to `pyproject.toml` version.

Primary docs in scope:
- `INSTALL.md`
- `docs/tutorial.html`

Deliverable:
- Primary docs always aligned with package version.

### Phase 4: Tests and Guardrails

- Update tests to avoid brittle hardcoded semver where they verify Loom version output:
  - `tests/test_cli.py` should compare against imported `__version__`.
- Add `tests/test_version_consistency.py` to validate:
  - `pyproject.toml` version equals `loom.__version__` in source execution.
  - key runtime surfaces emit same value.
- Add `scripts/check_version_consistency.py` and run in CI:
  - parse `pyproject.toml` authoritative version.
  - assert allowed file set contains no mismatched hardcoded Loom version literal.
  - fail with explicit per-file guidance.

Deliverable:
- Drift becomes a CI failure rather than a release surprise.

### Phase 5: Release Workflow Update

- Document version bump workflow in contributor docs:
  1. Edit `pyproject.toml` version.
  2. Run consistency script.
  3. Refresh lockfile/doc render outputs if required.
  4. Run targeted tests.
- Add make/uv helper command (optional):
  - `uv run python scripts/check_version_consistency.py`

Deliverable:
- Clear operator playbook for future bumps.

## File Change Plan

Likely touched code/config:
- `pyproject.toml` (authority retained; no structural change required unless adding tooling hooks)
- `src/loom/version.py` (new)
- `src/loom/__init__.py`
- `src/loom/api/server.py`
- `src/loom/integrations/mcp_tools.py`
- `src/loom/tui/app/process_runs/adhoc.py`
- `tests/test_cli.py`
- `tests/test_version_consistency.py` (new)
- `scripts/check_version_consistency.py` (new)
- `.github/workflows/ci.yml` (wire consistency check)
- `INSTALL.md`
- `docs/tutorial.html`

## Risks and Mitigations

- Risk: `importlib.metadata` missing package metadata in some local runs.
  - Mitigation: pyproject fallback path with explicit error if neither source is available.
- Risk: docs token/render flow adds maintenance overhead.
  - Mitigation: allow minimum-change mode (literal + CI consistency check) if tokenization is too heavy.
- Risk: process-package `version` field semantic ambiguity.
  - Mitigation: keep behavior but standardize source now; schedule separate semantic decision if process definition version should diverge from app version.

## Validation Strategy

Functional checks:
- `uv run loom --version` reports `pyproject.toml` version.
- `GET /health` returns same version.
- MCP initialize `clientInfo.version` matches same value.

Automated checks:
- `uv run pytest tests/test_cli.py tests/test_version_consistency.py`
- `uv run python scripts/check_version_consistency.py`
- Full CI includes consistency check.

Manual spot checks:
- Confirm primary doc snippets show same version string as `pyproject.toml`.

## Rollout Strategy

- Land as one cohesive PR if small enough.
- If split needed, use two PRs:
  - PR1: runtime + tests + consistency checker.
  - PR2: docs standardization + CI wiring.

## Exit Criteria

- Version editable in exactly one authoritative place: `pyproject.toml`.
- No hardcoded Loom package semver literals remain in runtime output paths.
- Primary docs version examples are aligned and enforced.
- CI blocks any future version drift.
