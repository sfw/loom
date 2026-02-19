"""Guard tests that keep core runtime free of domain-specific hardcoding."""

from __future__ import annotations

import re
from pathlib import Path

CORE_FILES = (
    "src/loom/state/evidence.py",
    "src/loom/engine/verification.py",
    "src/loom/recovery/retry.py",
    "src/loom/engine/orchestrator.py",
    "src/loom/prompts/templates/verifier.yaml",
)

FORBIDDEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "domain-specific market term": re.compile(r"\bmarkets?\b", re.IGNORECASE),
    "domain-specific recommendation term": re.compile(
        r"\brecommendations?\b",
        re.IGNORECASE,
    ),
    "market-specific field formatting": re.compile(r"market\s*=", re.IGNORECASE),
    "legacy market targeting key": re.compile(r"\bmissing_markets\b"),
    "legacy recommendation counter": re.compile(
        r"\brecommendation_unconfirmed_count\b"
    ),
    "legacy synthesis partition flag": re.compile(
        r"\bsynthesis_partition_present\b"
    ),
    "hardcoded remediation wording": re.compile(
        r"CONFIRM-OR-PRUNE REMEDIATION", re.IGNORECASE
    ),
    "legacy market evidence phrase": re.compile(
        r"no successful tool-call evidence found for market",
        re.IGNORECASE,
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_guard_files_exist():
    root = _repo_root()
    for rel in CORE_FILES:
        assert (root / rel).exists(), f"Missing guard target file: {rel}"


def test_no_domain_specific_hardcoding_in_core_runtime():
    root = _repo_root()
    violations: list[str] = []

    for rel in CORE_FILES:
        content = (root / rel).read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN_PATTERNS.items():
            for match in pattern.finditer(content):
                line_no = content.count("\n", 0, match.start()) + 1
                violations.append(f"{rel}:{line_no} -> {label}: {match.group(0)!r}")

    assert not violations, "Domain-specific hardcoding detected:\n" + "\n".join(violations)
