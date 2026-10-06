"""Narrow, policy-safe execution for deterministic correction actions."""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from loom.engine.correction.types import CorrectionDecision, CorrectionHandler


@dataclass(frozen=True)
class CorrectionExecutionResult:
    applied: bool
    resolved_locally: bool
    changed_targets: tuple[str, ...] = ()
    reason: str = ""


class CorrectionExecutor:
    """Execute only corrections that can be proven safe without broad agent work."""

    def execute(
        self,
        *,
        workspace: Path | None,
        decision: CorrectionDecision,
    ) -> CorrectionExecutionResult:
        if workspace is None:
            return CorrectionExecutionResult(False, False, reason="workspace_unavailable")
        if decision.handler != CorrectionHandler.SCHEMA_REPAIR:
            return CorrectionExecutionResult(False, False, reason="handler_not_deterministic")
        if not decision.actions:
            return CorrectionExecutionResult(False, False, reason="missing_action")
        return self._repair_csv_trailing_empty_overflow(
            workspace=workspace,
            arguments=decision.actions[0].arguments,
        )

    @staticmethod
    def _safe_target(workspace: Path, raw_target: str) -> tuple[Path, str] | None:
        root = workspace.resolve(strict=False)
        target = (root / str(raw_target or "")).resolve(strict=False)
        try:
            relative = target.relative_to(root).as_posix()
        except ValueError:
            return None
        if not relative or target.suffix.lower() != ".csv":
            return None
        if target.is_symlink() or not target.is_file():
            return None
        return target, relative

    def _repair_csv_trailing_empty_overflow(
        self,
        *,
        workspace: Path,
        arguments: dict[str, object],
    ) -> CorrectionExecutionResult:
        diagnostics = arguments.get("diagnostics", [])
        if not isinstance(diagnostics, list) or not diagnostics:
            return CorrectionExecutionResult(False, False, reason="diagnostics_unavailable")

        by_target: dict[str, list[dict[str, object]]] = {}
        for item in diagnostics:
            if not isinstance(item, dict):
                return CorrectionExecutionResult(False, False, reason="invalid_diagnostic")
            target = str(item.get("target", "") or "").strip()
            if not target:
                return CorrectionExecutionResult(False, False, reason="target_unavailable")
            by_target.setdefault(target, []).append(item)

        prepared: list[tuple[Path, str, list[list[str]]]] = []
        for raw_target, target_diagnostics in by_target.items():
            safe = self._safe_target(workspace, raw_target)
            if safe is None:
                return CorrectionExecutionResult(False, False, reason="target_not_safe")
            path, relative = safe
            with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
                rows = list(csv.reader(handle))
            if not rows:
                return CorrectionExecutionResult(False, False, reason="empty_csv")

            changed = False
            for diagnostic in target_diagnostics:
                try:
                    row_number = int(diagnostic["row_number"])
                    actual = int(diagnostic["actual_columns"])
                    expected = int(diagnostic["expected_columns"])
                except (KeyError, TypeError, ValueError):
                    return CorrectionExecutionResult(
                        False,
                        False,
                        reason="incomplete_diagnostic",
                    )
                if row_number < 2 or row_number > len(rows):
                    return CorrectionExecutionResult(False, False, reason="row_out_of_range")
                row = rows[row_number - 1]
                if len(row) != actual or actual <= expected:
                    return CorrectionExecutionResult(False, False, reason="diagnostic_stale")
                overflow = row[expected:]
                if any(cell.strip() for cell in overflow):
                    return CorrectionExecutionResult(
                        False,
                        False,
                        reason="ambiguous_nonempty_overflow",
                    )
                rows[row_number - 1] = row[:expected]
                changed = True
            if changed:
                prepared.append((path, relative, rows))

        if not prepared:
            return CorrectionExecutionResult(False, False, reason="no_safe_change")

        changed_targets: list[str] = []
        for path, relative, rows in prepared:
            temp_name = ""
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="",
                    dir=path.parent,
                    prefix=f".{path.name}.repair-",
                    delete=False,
                ) as handle:
                    temp_name = handle.name
                    writer = csv.writer(handle, lineterminator="\n")
                    writer.writerows(rows)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_name, path)
                changed_targets.append(relative)
            finally:
                if temp_name and os.path.exists(temp_name):
                    os.unlink(temp_name)

        return CorrectionExecutionResult(
            applied=True,
            resolved_locally=True,
            changed_targets=tuple(changed_targets),
            reason="trimmed_trailing_empty_overflow",
        )
