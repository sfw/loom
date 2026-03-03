"""Gate-driven iteration evaluation for process phases."""

from __future__ import annotations

import asyncio
import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from loom.engine.runner import SubtaskResult
from loom.engine.verification import VerificationResult
from loom.processes.schema import IterationGate, IterationPolicy


@dataclass
class IterationGateResult:
    """Normalized result for one iteration gate evaluation."""

    gate_id: str
    gate_type: str
    blocking: bool
    passed: bool
    status: str  # pass | fail | unevaluable
    reason_code: str = ""
    measured_value: object = None
    threshold_value: object = None
    detail: str = ""


@dataclass
class IterationEvaluation:
    """Aggregate decision across all configured iteration gates."""

    results: list[IterationGateResult] = field(default_factory=list)
    blocking_failures: list[IterationGateResult] = field(default_factory=list)
    advisory_failures: list[IterationGateResult] = field(default_factory=list)
    score_hint: float | None = None

    @property
    def all_blocking_passed(self) -> bool:
        return not self.blocking_failures


class _GateUnevaluableError(RuntimeError):
    def __init__(self, reason_code: str, detail: str):
        super().__init__(detail)
        self.reason_code = str(reason_code or "gate_unevaluable").strip().lower()
        self.detail = str(detail or "gate evaluation unavailable").strip()


class IterationGateEvaluator:
    """Evaluate per-phase iteration gates against run outputs."""

    _OPERATORS = frozenset({"gte", "lte", "eq", "contains", "not_contains"})
    _ARTIFACT_REGEX_TARGETS = frozenset({
        "",
        "auto",
        "deliverables",
        "changed_files",
        "summary",
        "output",
    })

    def __init__(
        self,
        *,
        command_allowlisted_prefixes: list[str] | None = None,
        enable_command_exit: bool = False,
    ):
        raw_prefixes = command_allowlisted_prefixes or []
        self._command_allowlisted_prefixes = [
            tuple(token for token in shlex.split(item) if token)
            for item in raw_prefixes
            if str(item).strip()
        ]
        self._enable_command_exit = bool(enable_command_exit)

    async def evaluate(
        self,
        *,
        policy: IterationPolicy,
        result: SubtaskResult,
        verification: VerificationResult,
        workspace: Path | None,
        expected_deliverables: list[str] | None = None,
    ) -> IterationEvaluation:
        outcomes: list[IterationGateResult] = []
        for gate in policy.gates:
            outcomes.append(
                await self._evaluate_gate_with_retry(
                    gate=gate,
                    result=result,
                    verification=verification,
                    workspace=workspace,
                    expected_deliverables=expected_deliverables or [],
                ),
            )

        blocking_failures = [item for item in outcomes if item.blocking and not item.passed]
        advisory_failures = [item for item in outcomes if (not item.blocking) and not item.passed]
        return IterationEvaluation(
            results=outcomes,
            blocking_failures=blocking_failures,
            advisory_failures=advisory_failures,
            score_hint=self._extract_score_hint(outcomes),
        )

    async def _evaluate_gate_with_retry(
        self,
        *,
        gate: IterationGate,
        result: SubtaskResult,
        verification: VerificationResult,
        workspace: Path | None,
        expected_deliverables: list[str],
    ) -> IterationGateResult:
        last_error: _GateUnevaluableError | None = None
        for attempt in range(2):
            try:
                return await self._evaluate_gate(
                    gate=gate,
                    result=result,
                    verification=verification,
                    workspace=workspace,
                    expected_deliverables=expected_deliverables,
                )
            except _GateUnevaluableError as exc:
                last_error = exc
                if attempt == 0:
                    continue
        return IterationGateResult(
            gate_id=gate.id,
            gate_type=gate.type,
            blocking=bool(gate.blocking),
            passed=False,
            status="unevaluable",
            reason_code="gate_unevaluable",
            detail=(last_error.detail if last_error is not None else "gate evaluation unavailable"),
            threshold_value=gate.value,
        )

    async def _evaluate_gate(
        self,
        *,
        gate: IterationGate,
        result: SubtaskResult,
        verification: VerificationResult,
        workspace: Path | None,
        expected_deliverables: list[str],
    ) -> IterationGateResult:
        gate_type = str(gate.type or "").strip().lower()
        if gate_type == "tool_metric":
            return self._evaluate_tool_metric_gate(gate=gate, result=result)
        if gate_type == "artifact_regex":
            return self._evaluate_artifact_regex_gate(
                gate=gate,
                workspace=workspace,
                expected_deliverables=expected_deliverables,
                result=result,
            )
        if gate_type == "command_exit":
            return await self._evaluate_command_exit_gate(
                gate=gate,
                workspace=workspace,
            )
        if gate_type == "verifier_field":
            return self._evaluate_verifier_field_gate(
                gate=gate,
                verification=verification,
            )
        raise _GateUnevaluableError("gate_unevaluable", f"Unsupported gate type: {gate_type}")

    def _evaluate_tool_metric_gate(
        self,
        *,
        gate: IterationGate,
        result: SubtaskResult,
    ) -> IterationGateResult:
        tool_name = str(gate.tool or "").strip()
        if not tool_name:
            raise _GateUnevaluableError("gate_missing_tool", "tool_metric gate missing tool")
        metric_path = str(gate.metric_path or "").strip()
        if not metric_path:
            raise _GateUnevaluableError(
                "gate_missing_metric_path",
                "tool_metric gate missing metric_path",
            )

        chosen_call = None
        for call in reversed(result.tool_calls):
            if str(getattr(call, "tool", "")).strip() != tool_name:
                continue
            if bool(getattr(getattr(call, "result", None), "success", False)):
                chosen_call = call
                break
            if chosen_call is None:
                chosen_call = call
        if chosen_call is None:
            raise _GateUnevaluableError(
                "gate_tool_not_called",
                f"tool_metric gate references missing tool call: {tool_name}",
            )

        payload = getattr(getattr(chosen_call, "result", None), "data", None)
        if not isinstance(payload, dict):
            raise _GateUnevaluableError(
                "gate_metric_payload_missing",
                f"Tool '{tool_name}' returned no structured data",
            )
        measured = self._resolve_path(payload, metric_path)
        if measured is _MISSING:
            raise _GateUnevaluableError(
                "gate_metric_path_missing",
                f"Metric path '{metric_path}' not found",
            )

        return self._build_comparison_result(
            gate=gate,
            measured=measured,
            default_reason="gate_threshold_not_met",
        )

    def _evaluate_artifact_regex_gate(
        self,
        *,
        gate: IterationGate,
        workspace: Path | None,
        expected_deliverables: list[str],
        result: SubtaskResult,
    ) -> IterationGateResult:
        pattern_text = str(gate.pattern or "")
        if not pattern_text:
            raise _GateUnevaluableError(
                "gate_pattern_missing",
                "artifact_regex gate missing pattern",
            )
        try:
            pattern = re.compile(pattern_text)
        except re.error as exc:
            raise _GateUnevaluableError("gate_pattern_invalid", str(exc)) from exc

        target = str(gate.target or "").strip().lower()
        if target not in self._ARTIFACT_REGEX_TARGETS:
            raise _GateUnevaluableError(
                "gate_target_invalid",
                f"unsupported artifact_regex target: {target}",
            )

        if target in {"summary", "output"}:
            summary = str(getattr(result, "summary", "") or "")
            if not summary.strip():
                raise _GateUnevaluableError(
                    "gate_summary_missing",
                    "artifact_regex target summary/output has no text",
                )
            haystack = [summary]
        else:
            if workspace is None:
                raise _GateUnevaluableError(
                    "gate_workspace_missing",
                    "artifact_regex requires a workspace for file-based targets",
                )
            candidates: list[str] = []
            if target in {"", "auto", "deliverables"}:
                candidates.extend(
                    str(path).strip() for path in expected_deliverables if str(path).strip()
                )
            if target in {"", "auto", "changed_files"}:
                candidates.extend(self._changed_files_from_result(result))
            deduped_candidates: list[str] = []
            for rel_path in candidates:
                if rel_path not in deduped_candidates:
                    deduped_candidates.append(rel_path)

            if not deduped_candidates:
                raise _GateUnevaluableError(
                    "gate_no_artifacts",
                    "artifact_regex gate has no files to scan",
                )

            haystack = []
            for rel_path in deduped_candidates:
                text = self._read_small_text_file(workspace / rel_path)
                if text:
                    haystack.append(text)
            if not haystack:
                raise _GateUnevaluableError(
                    "gate_artifacts_unreadable",
                    "No readable artifact contents found for artifact_regex",
                )

        merged = "\n".join(haystack)
        matched = bool(pattern.search(merged))
        expect_match = bool(gate.expect_match)
        passed = matched == expect_match
        return IterationGateResult(
            gate_id=gate.id,
            gate_type=gate.type,
            blocking=bool(gate.blocking),
            passed=passed,
            status="pass" if passed else "fail",
            reason_code="" if passed else "gate_regex_expectation_failed",
            measured_value=matched,
            threshold_value=expect_match,
            detail=(
                "regex expectation satisfied"
                if passed
                else f"regex expectation failed (expect_match={expect_match}, matched={matched})"
            ),
        )

    async def _evaluate_command_exit_gate(
        self,
        *,
        gate: IterationGate,
        workspace: Path | None,
    ) -> IterationGateResult:
        command = [str(token).strip() for token in gate.command if str(token).strip()]
        if not command:
            raise _GateUnevaluableError("gate_command_missing", "command_exit gate missing command")
        if not self._enable_command_exit:
            raise _GateUnevaluableError(
                "gate_command_exit_disabled",
                "command_exit gate is disabled by runtime policy",
            )
        if not self._command_allowlisted(command):
            raise _GateUnevaluableError(
                "gate_command_not_allowlisted",
                f"Command not allowlisted: {' '.join(command)}",
            )

        timeout_seconds = max(1, int(gate.timeout_seconds or 60))
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(workspace) if workspace is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=float(timeout_seconds),
                )
            except TimeoutError as exc:
                proc.kill()
                await proc.communicate()
                raise _GateUnevaluableError(
                    "gate_command_timeout",
                    f"command_exit timed out after {timeout_seconds}s",
                ) from exc
        except OSError as exc:
            raise _GateUnevaluableError("gate_command_exec_error", str(exc)) from exc

        expected_exit = gate.value if isinstance(gate.value, int) else 0
        operator = str(gate.operator or "eq").strip().lower() or "eq"
        if operator not in self._OPERATORS:
            operator = "eq"
        passed = self._compare(operator=operator, measured=proc.returncode, expected=expected_exit)
        detail = ""
        if not passed:
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            detail = stderr_text or stdout_text
            if len(detail) > 400:
                detail = detail[:400].rstrip() + "..."
        return IterationGateResult(
            gate_id=gate.id,
            gate_type=gate.type,
            blocking=bool(gate.blocking),
            passed=passed,
            status="pass" if passed else "fail",
            reason_code="" if passed else "gate_command_exit_mismatch",
            measured_value=proc.returncode,
            threshold_value=expected_exit,
            detail=detail,
        )

    def _evaluate_verifier_field_gate(
        self,
        *,
        gate: IterationGate,
        verification: VerificationResult,
    ) -> IterationGateResult:
        field_path = str(gate.verifier_field or "").strip()
        if not field_path:
            raise _GateUnevaluableError("gate_field_missing", "verifier_field gate missing field")
        payload = {
            "passed": verification.passed,
            "outcome": verification.outcome,
            "reason_code": verification.reason_code,
            "severity_class": verification.severity_class,
            "confidence": verification.confidence,
            "feedback": verification.feedback,
            "metadata": verification.metadata,
            "checks": [
                {
                    "name": getattr(item, "name", ""),
                    "passed": bool(getattr(item, "passed", False)),
                    "detail": getattr(item, "detail", None),
                }
                for item in verification.checks
            ],
        }
        measured = self._resolve_path(payload, field_path)
        if measured is _MISSING:
            raise _GateUnevaluableError(
                "gate_field_missing",
                f"verifier_field path '{field_path}' not found",
            )
        return self._build_comparison_result(
            gate=gate,
            measured=measured,
            default_reason="gate_verifier_field_mismatch",
        )

    def _build_comparison_result(
        self,
        *,
        gate: IterationGate,
        measured: object,
        default_reason: str,
    ) -> IterationGateResult:
        operator = str(gate.operator or "eq").strip().lower() or "eq"
        if operator not in self._OPERATORS:
            raise _GateUnevaluableError(
                "gate_operator_invalid",
                f"unsupported operator: {operator}",
            )
        passed = self._compare(operator=operator, measured=measured, expected=gate.value)
        return IterationGateResult(
            gate_id=gate.id,
            gate_type=gate.type,
            blocking=bool(gate.blocking),
            passed=passed,
            status="pass" if passed else "fail",
            reason_code="" if passed else default_reason,
            measured_value=measured,
            threshold_value=gate.value,
            detail="" if passed else f"comparison failed ({operator})",
        )

    def _command_allowlisted(self, argv: list[str]) -> bool:
        if not self._command_allowlisted_prefixes:
            return False
        command = tuple(argv)
        for prefix in self._command_allowlisted_prefixes:
            if not prefix:
                continue
            if len(command) < len(prefix):
                continue
            if command[: len(prefix)] == prefix:
                return True
        return False

    @staticmethod
    def _extract_score_hint(results: list[IterationGateResult]) -> float | None:
        for result in results:
            if result.status == "unevaluable":
                continue
            if result.gate_type != "tool_metric":
                continue
            value = result.measured_value
            if isinstance(value, (int, float)):
                return float(value)
        return None

    @staticmethod
    def _changed_files_from_result(result: SubtaskResult) -> list[str]:
        changed_files: list[str] = []
        seen: set[str] = set()
        for call in result.tool_calls:
            changed = getattr(getattr(call, "result", None), "files_changed", [])
            if not isinstance(changed, list):
                continue
            for value in changed:
                path = str(value).strip()
                if not path or path in seen:
                    continue
                seen.add(path)
                changed_files.append(path)
        return changed_files

    @staticmethod
    def _read_small_text_file(path: Path, *, max_bytes: int = 300_000) -> str:
        try:
            if not path.exists() or not path.is_file():
                return ""
            if path.stat().st_size > max_bytes:
                return ""
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    @staticmethod
    def _resolve_path(payload: object, dotted_path: str) -> object:
        current = payload
        for segment in [part for part in str(dotted_path or "").split(".") if part.strip()]:
            key = segment.strip()
            if isinstance(current, dict):
                if key not in current:
                    return _MISSING
                current = current.get(key)
                continue
            if isinstance(current, list):
                try:
                    idx = int(key)
                except ValueError:
                    return _MISSING
                if idx < 0 or idx >= len(current):
                    return _MISSING
                current = current[idx]
                continue
            return _MISSING
        return current

    @staticmethod
    def _coerce_number(value: object) -> float | None:
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip().replace(",", "")
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None
        return None

    def _compare(self, *, operator: str, measured: object, expected: object) -> bool:
        if operator == "eq":
            measured_num = self._coerce_number(measured)
            expected_num = self._coerce_number(expected)
            if measured_num is not None and expected_num is not None:
                return measured_num == expected_num
            return measured == expected

        if operator in {"gte", "lte"}:
            measured_num = self._coerce_number(measured)
            expected_num = self._coerce_number(expected)
            if measured_num is None or expected_num is None:
                return False
            if operator == "gte":
                return measured_num >= expected_num
            return measured_num <= expected_num

        measured_text = self._stringify(measured)
        expected_text = self._stringify(expected)
        if operator == "contains":
            return expected_text in measured_text
        if operator == "not_contains":
            return expected_text not in measured_text
        return False

    @staticmethod
    def _stringify(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(value)


_MISSING = object()
