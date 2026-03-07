#!/usr/bin/env python3
"""Enforce schema-change policy for Loom SQLite migrations."""

from __future__ import annotations

import os
import subprocess

SCHEMA_PATHS = (
    "src/loom/state/schema.sql",
    "src/loom/state/schema/base.sql",
)
MIGRATION_STEP_PREFIX = "src/loom/state/migrations/steps/"
MIGRATION_REGISTRY = "src/loom/state/migrations/registry.py"
DOC_REQUIRED = {
    "README.md",
    "docs/CONFIG.md",
    "docs/agent-integration.md",
    "docs/DB-MIGRATIONS.md",
}
MIGRATION_TEST_HINTS = (
    "tests/test_memory.py",
    "tests/test_cli.py",
    "tests/tui/<module>.py",
    "tests/test_migrations.py",
)


def _is_migration_focused_test(path: str) -> bool:
    normalized = str(path or "").strip()
    if not normalized.startswith("tests/"):
        return False
    if normalized.startswith("tests/tui/"):
        return True
    if normalized in MIGRATION_TEST_HINTS:
        return True
    filename = normalized.rsplit("/", 1)[-1].lower()
    return "migration" in filename or "schema" in filename


def _run_git(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _resolve_diff_range() -> str:
    explicit = str(os.getenv("DB_POLICY_DIFF_RANGE", "")).strip()
    if explicit:
        return explicit

    event = str(os.getenv("GITHUB_EVENT_NAME", "")).strip().lower()
    if event == "pull_request":
        base_ref = str(os.getenv("GITHUB_BASE_REF", "main")).strip() or "main"
        subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", base_ref],
            check=False,
            capture_output=True,
            text=True,
        )
        return f"origin/{base_ref}...HEAD"

    # Push/default fallback: compare HEAD with previous commit when available.
    has_parent = _run_git(["rev-parse", "--verify", "HEAD~1"])
    if has_parent:
        return "HEAD~1...HEAD"
    return ""


def _changed_files() -> set[str]:
    diff_range = _resolve_diff_range()
    if diff_range:
        output = _run_git(["diff", "--name-only", diff_range])
    else:
        output = _run_git(["diff", "--name-only"])
    return {line.strip() for line in output.splitlines() if line.strip()}


def main() -> int:
    changed = _changed_files()
    if not changed:
        print("db-policy: no file changes detected; skipping.")
        return 0

    errors: list[str] = []
    schema_changed = any(path in changed for path in SCHEMA_PATHS)
    migration_steps_changed = any(path.startswith(MIGRATION_STEP_PREFIX) for path in changed)
    registry_changed = MIGRATION_REGISTRY in changed

    if schema_changed and not migration_steps_changed:
        errors.append(
            "Schema changed (schema.sql or schema/base.sql) but no "
            "migration step file changed under "
            f"`{MIGRATION_STEP_PREFIX}`."
        )

    if migration_steps_changed and not registry_changed:
        errors.append(
            "Migration step files changed but migration registry was not updated "
            f"(`{MIGRATION_REGISTRY}`)."
        )

    if schema_changed:
        docs_changed = changed.intersection(DOC_REQUIRED)
        if not docs_changed:
            errors.append(
                "Schema changed but required docs were not updated. "
                f"Update at least one of: {', '.join(sorted(DOC_REQUIRED))}."
            )
        migration_tests_changed = any(_is_migration_focused_test(path) for path in changed)
        if not migration_tests_changed:
            errors.append(
                "Schema changed but no migration-focused tests were updated. "
                "Update one of: "
                f"{', '.join(MIGRATION_TEST_HINTS)} "
                "or add a dedicated migration/schema test file."
            )

    if errors:
        print("db-policy: FAILED")
        for item in errors:
            print(f"- {item}")
        return 1

    print("db-policy: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
