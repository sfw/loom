"""Golden-corpus replay tests for verification policy hardening."""

from __future__ import annotations

import json
from pathlib import Path

from loom.engine.verification import VerificationGates, VerificationResult
from loom.recovery.retry import RetryManager

GOLDEN_DIR = Path(__file__).parent / "golden" / "verification"


def _load_cases() -> list[dict]:
    paths = sorted(GOLDEN_DIR.glob("*.json"))
    assert paths, f"No golden verification cases found under {GOLDEN_DIR}"
    cases: list[dict] = []
    for path in paths:
        case = json.loads(path.read_text(encoding="utf-8"))
        case["_path"] = str(path)
        cases.append(case)
    return cases


def _result_from_payload(payload: dict) -> VerificationResult:
    return VerificationResult(
        tier=int(payload.get("tier", 2) or 2),
        passed=bool(payload.get("passed", False)),
        confidence=float(payload.get("confidence", 0.5) or 0.5),
        outcome=str(payload.get("outcome", "") or ""),
        reason_code=str(payload.get("reason_code", "") or ""),
        severity_class=str(payload.get("severity_class", "") or ""),
    )


def test_golden_shadow_diff_replay_quality_gate():
    """Replay corpus and enforce shadow-diff quality gates."""
    metrics = {
        "old_fail_new_pass": 0,
        "old_pass_new_fail": 0,
        "both_fail_reason_diff": 0,
        "no_diff": 0,
    }
    for case in _load_cases():
        context = case.get("context", {})
        assert isinstance(context, dict), f"invalid context for {case.get('id')}"
        assert str(context.get("phase_id", "")).strip()
        tool_calls = context.get("tool_calls", [])
        assert isinstance(tool_calls, list), f"invalid tool_calls for {case.get('id')}"

        legacy = _result_from_payload(case["legacy"])
        policy = _result_from_payload(case["policy"])
        observed = VerificationGates.classify_shadow_diff(legacy, policy)
        expected = str(case["expected"]["shadow_classification"])
        assert observed == expected, (
            f"golden case {case.get('id')} ({case.get('_path')}): "
            f"expected {expected}, observed {observed}"
        )
        metrics[observed] += 1

    # Hardening quality gates.
    assert metrics["old_pass_new_fail"] == 0
    assert metrics["old_fail_new_pass"] >= 1
    assert metrics["both_fail_reason_diff"] >= 1


def test_golden_retry_strategy_replay():
    """Structured retry routing should match expected strategy in corpus."""
    for case in _load_cases():
        expected = str(case.get("expected", {}).get("retry_strategy", "")).strip()
        if not expected:
            continue
        contract = case.get("verification_contract")
        if not isinstance(contract, dict):
            contract = case.get("policy", {})
        strategy, _markets = RetryManager.classify_failure(
            verification_feedback=str(case.get("context", {}).get("raw_verifier_output", "")),
            execution_error="",
            verification=contract,
        )
        assert strategy.value == expected, (
            f"golden case {case.get('id')} ({case.get('_path')}): "
            f"expected {expected}, observed {strategy.value}"
        )
