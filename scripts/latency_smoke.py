#!/usr/bin/env python3
"""Quick local latency smoke checks for startup/discovery hot paths.

Usage:
  uv run python scripts/latency_smoke.py
  uv run python scripts/latency_smoke.py --iterations 5 --workspace /path/to/ws
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from loom.config import Config, load_config
from loom.mcp.config import apply_mcp_overrides
from loom.processes.schema import ProcessLoader
from loom.tools import create_default_registry


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((q / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def _run_benchmark(
    label: str,
    fn: Callable[[], None],
    *,
    iterations: int,
) -> None:
    samples_ms: list[float] = []
    for _ in range(iterations):
        start = time.monotonic()
        fn()
        samples_ms.append((time.monotonic() - start) * 1000.0)

    mean_ms = statistics.fmean(samples_ms)
    p50_ms = _percentile(samples_ms, 50)
    p95_ms = _percentile(samples_ms, 95)
    print(
        f"{label:42s} "
        f"mean={mean_ms:8.2f}ms p50={p50_ms:8.2f}ms p95={p95_ms:8.2f}ms"
    )


def _load_effective_config(workspace: Path) -> Config:
    loaded = load_config()
    if isinstance(loaded, Config):
        return apply_mcp_overrides(loaded, workspace=workspace)
    return loaded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per check.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace used for process scans/config overlays.",
    )
    parser.add_argument(
        "--wait-background-seconds",
        type=float,
        default=1.5,
        help="Sleep after background registry create to allow MCP warmup thread to run.",
    )
    args = parser.parse_args()

    iterations = max(1, int(args.iterations))
    workspace = args.workspace.expanduser().resolve()
    config = _load_effective_config(workspace)

    print(f"Workspace: {workspace}")
    print(f"Iterations: {iterations}")
    print("")

    extra = [Path(p) for p in config.process.search_paths]
    loader = ProcessLoader(
        workspace=workspace,
        extra_search_paths=extra,
        require_rule_scope_metadata=bool(
            getattr(config.process, "require_rule_scope_metadata", False),
        ),
        require_v2_contract=bool(
            getattr(config.process, "require_v2_contract", False),
        ),
    )

    _run_benchmark(
        "ProcessLoader.list_available()",
        lambda: loader.list_available(),
        iterations=iterations,
    )

    _run_benchmark(
        "create_default_registry(sync MCP)",
        lambda: create_default_registry(config, mcp_startup_mode="sync"),
        iterations=iterations,
    )

    def _background_build() -> None:
        registry = create_default_registry(config, mcp_startup_mode="background")
        registry.list_tools()
        if args.wait_background_seconds > 0:
            time.sleep(args.wait_background_seconds)

    _run_benchmark(
        "create_default_registry(background MCP)",
        _background_build,
        iterations=iterations,
    )

    print("")
    print("Tip: enable runtime diagnostics with LOOM_LATENCY_DIAGNOSTICS=1")
    print("Example: LOOM_LATENCY_DIAGNOSTICS=1 uv run loom")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
