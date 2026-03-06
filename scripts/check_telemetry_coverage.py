#!/usr/bin/env python3
"""Static telemetry drift checker.

Compares active declared event types against known source emission references
and test references. Fails with a non-zero exit code on missing coverage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from loom.events.types import ACTIVE_EVENT_TYPES, EVENT_NAME_TO_TYPE

SRC_ROOT = Path("src/loom")
TEST_ROOT = Path("tests")

_CONST_PATTERN = re.compile(r"\b([A-Z][A-Z0-9_]+)\b")


def _iter_py_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _scan_emission_sites() -> tuple[set[str], dict[str, list[str]]]:
    events: set[str] = set()
    hits: dict[str, list[str]] = {}
    const_to_value = dict(EVENT_NAME_TO_TYPE)
    active_values = set(ACTIVE_EVENT_TYPES)

    for path in _iter_py_files(SRC_ROOT):
        if path.name == "types.py" and path.parent.name == "events":
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line_number, line in enumerate(lines, start=1):
            for const_name in _CONST_PATTERN.findall(line):
                value = const_to_value.get(const_name)
                if value and value in active_values:
                    events.add(value)
                    hits.setdefault(value, []).append(f"{path}:{line_number}")
            for event_value in active_values:
                if f'"{event_value}"' in line or f"'{event_value}'" in line:
                    events.add(event_value)
                    hits.setdefault(event_value, []).append(f"{path}:{line_number}")
    return events, hits


def _scan_test_references() -> tuple[set[str], dict[str, list[str]]]:
    events: set[str] = set()
    hits: dict[str, list[str]] = {}
    const_to_value = dict(EVENT_NAME_TO_TYPE)
    active_values = set(ACTIVE_EVENT_TYPES)

    for path in _iter_py_files(TEST_ROOT):
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if "ACTIVE_EVENT_TYPES" in line:
                for event_value in active_values:
                    events.add(event_value)
                    hits.setdefault(event_value, []).append(f"{path}:{line_number}")
            for const_name in _CONST_PATTERN.findall(line):
                value = const_to_value.get(const_name)
                if value and value in active_values:
                    events.add(value)
                    hits.setdefault(value, []).append(f"{path}:{line_number}")
            for event_value in active_values:
                if f'"{event_value}"' in line or f"'{event_value}'" in line:
                    events.add(event_value)
                    hits.setdefault(event_value, []).append(f"{path}:{line_number}")
    return events, hits


def _build_report() -> dict[str, object]:
    active_events = sorted(ACTIVE_EVENT_TYPES)
    emitted_events, emission_hits = _scan_emission_sites()
    tested_events, test_hits = _scan_test_references()

    missing_emission = sorted(set(active_events) - set(emitted_events))
    missing_tests = sorted(set(active_events) - set(tested_events))

    return {
        "active_event_count": len(active_events),
        "emitted_event_count": len(emitted_events),
        "tested_event_count": len(tested_events),
        "missing_emission": missing_emission,
        "missing_tests": missing_tests,
        "emission_hits": {
            event: emission_hits.get(event, [])[:12]
            for event in sorted(emission_hits)
        },
        "test_hits": {
            event: test_hits.get(event, [])[:12]
            for event in sorted(test_hits)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check telemetry declaration/emission/test drift.")
    parser.add_argument(
        "--report",
        default="",
        help="Optional path to write JSON report.",
    )
    args = parser.parse_args()

    report = _build_report()
    if args.report:
        report_path = Path(args.report)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    missing_emission = report["missing_emission"]
    missing_tests = report["missing_tests"]

    if missing_emission or missing_tests:
        print("Telemetry coverage check failed.")
        if missing_emission:
            print("Missing emission coverage:")
            for event in missing_emission:
                print(f"  - {event}")
        if missing_tests:
            print("Missing test coverage:")
            for event in missing_tests:
                print(f"  - {event}")
        return 1

    print(
        "Telemetry coverage check passed: "
        f"{report['active_event_count']} active events, "
        f"{report['emitted_event_count']} emitted, "
        f"{report['tested_event_count']} test-referenced.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
