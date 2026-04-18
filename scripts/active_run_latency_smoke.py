#!/usr/bin/env python3
"""Synthetic active-run latency smoke checks for hot desktop API surfaces.

Usage:
  uv run python scripts/active_run_latency_smoke.py
  uv run python scripts/active_run_latency_smoke.py --iterations 5 --event-count 500
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from loom.api.engine import create_engine
from loom.api.routes import router
from loom.config import Config
from loom.events.bus import Event
from loom.events.types import TASK_EXECUTING, TASK_RUN_HEARTBEAT


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((q / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


async def _run_benchmark(
    label: str,
    fn,
    *,
    iterations: int,
) -> None:
    samples_ms: list[float] = []
    for _ in range(iterations):
        started = time.monotonic()
        await fn()
        samples_ms.append((time.monotonic() - started) * 1000.0)

    mean_ms = statistics.fmean(samples_ms)
    p50_ms = _percentile(samples_ms, 50)
    p95_ms = _percentile(samples_ms, 95)
    print(
        f"{label:42s} "
        f"mean={mean_ms:8.2f}ms p50={p50_ms:8.2f}ms p95={p95_ms:8.2f}ms"
    )


def _benchmark_config(root: Path) -> Config:
    base = Config()
    return replace(
        base,
        workspace=replace(base.workspace, scratch_dir=str(root / "scratch")),
        memory=replace(base.memory, database_path=str(root / "loom.db")),
        mcp=replace(base.mcp, servers={}),
    )


def _build_app(engine) -> FastAPI:
    app = FastAPI()
    app.state.engine = engine
    app.include_router(router)
    return app


async def _seed_fixture(engine, workspace: Path) -> tuple[str, str, str]:
    workspace.mkdir(parents=True, exist_ok=True)
    workspace_row = await engine.workspace_registry.ensure_workspace(str(workspace))
    if workspace_row is None:
        raise RuntimeError("Failed to seed workspace fixture.")

    task_id = "latency-run-1"
    execution_run_id = "exec-run-latency-1"
    await engine.database.insert_task(
        task_id=task_id,
        goal="Latency smoke active run",
        workspace_path=str(workspace),
        status="executing",
    )
    await engine.database.insert_task_run(
        run_id=execution_run_id,
        task_id=task_id,
        status="running",
        process_name="latency-smoke",
    )
    for sequence in range(1, 51):
        await engine.database.insert_event(
            task_id=task_id,
            correlation_id=f"run:{execution_run_id}",
            run_id=execution_run_id,
            event_type=TASK_EXECUTING if sequence % 5 == 0 else TASK_RUN_HEARTBEAT,
            data={
                "run_id": execution_run_id,
                "status": "executing",
                "message": f"tick-{sequence}",
                "source_component": "latency_smoke",
            },
            sequence=sequence,
            source_component="latency_smoke",
        )

    session_id = await engine.conversation_store.create_session(
        workspace=str(workspace),
        model_name="latency-smoke-model",
    )
    await engine.conversation_store.link_run(session_id, task_id)
    for turn_number in range(1, 41):
        role = "user" if turn_number % 2 else "assistant"
        await engine.conversation_store.append_turn(
            session_id=session_id,
            turn_number=turn_number,
            role=role,
            content=f"{role} turn {turn_number}",
        )

    await engine.database.upsert_pending_task_question(
        question_id="latency-question-1",
        task_id=task_id,
        subtask_id="subtask-latency-1",
        request_payload={
            "question_id": "latency-question-1",
            "question": "Choose the next step",
            "question_type": "single_choice",
        },
    )

    return str(workspace_row["id"]), task_id, session_id


async def _emit_active_run_burst(
    engine,
    *,
    task_id: str,
    execution_run_id: str,
    event_count: int,
) -> None:
    for index in range(max(0, int(event_count))):
        event_type = TASK_EXECUTING if index % 5 == 0 else TASK_RUN_HEARTBEAT
        engine.event_bus.emit(
            Event(
                event_type=event_type,
                task_id=task_id,
                data={
                    "run_id": execution_run_id,
                    "status": "executing",
                    "message": f"burst-{index}",
                    "source_component": "latency_smoke",
                },
            ),
        )
        if index % 25 == 0:
            await asyncio.sleep(0)


async def _consume_stream(client: AsyncClient, path: str) -> None:
    async with client.stream("GET", path) as response:
        async for _line in response.aiter_lines():
            pass


async def main_async(iterations: int, event_count: int) -> int:
    with tempfile.TemporaryDirectory(prefix="loom-active-run-smoke-") as temp_root:
        root = Path(temp_root)
        config = _benchmark_config(root)
        workspace = root / "workspace"
        engine = await create_engine(config, runtime_role="api")
        app = _build_app(engine)
        execution_run_id = "exec-run-latency-1"
        try:
            workspace_id, task_id, session_id = await _seed_fixture(engine, workspace)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                emitter = asyncio.create_task(
                    _emit_active_run_burst(
                        engine,
                        task_id=task_id,
                        execution_run_id=execution_run_id,
                        event_count=event_count,
                    ),
                )
                try:
                    print(f"Iterations: {iterations}")
                    print(f"Event count: {event_count}")
                    print("")

                    await _run_benchmark(
                        "GET /workspaces/{id}/overview",
                        lambda: client.get(f"/workspaces/{workspace_id}/overview"),
                        iterations=iterations,
                    )
                    await _run_benchmark(
                        "GET /workspaces/{id}/artifacts",
                        lambda: client.get(f"/workspaces/{workspace_id}/artifacts"),
                        iterations=iterations,
                    )
                    await _run_benchmark(
                        "GET /approvals",
                        lambda: client.get(f"/approvals?workspace_id={workspace_id}"),
                        iterations=iterations,
                    )
                    await _run_benchmark(
                        "GET /conversations/{id}",
                        lambda: client.get(f"/conversations/{session_id}"),
                        iterations=iterations,
                    )
                    await _run_benchmark(
                        "GET /conversations/{id}/messages",
                        lambda: client.get(
                            f"/conversations/{session_id}/messages?latest=true&limit=250",
                        ),
                        iterations=iterations,
                    )
                    await _run_benchmark(
                        "GET /conversations/{id}/events",
                        lambda: client.get(f"/conversations/{session_id}/events?limit=250"),
                        iterations=iterations,
                    )
                    await _run_benchmark(
                        "GET /conversations/{id}/status",
                        lambda: client.get(f"/conversations/{session_id}/status"),
                        iterations=iterations,
                    )
                    await _run_benchmark(
                        "GET /runs/{id}",
                        lambda: client.get(f"/runs/{task_id}"),
                        iterations=iterations,
                    )
                    await _run_benchmark(
                        "GET /runs/{id}/timeline",
                        lambda: client.get(f"/runs/{task_id}/timeline?limit=1000"),
                        iterations=iterations,
                    )
                    await _run_benchmark(
                        "GET /runs/{id}/stream?follow=false",
                        lambda: _consume_stream(client, f"/runs/{task_id}/stream?follow=false"),
                        iterations=iterations,
                    )
                finally:
                    await emitter
                    if engine.event_persister is not None:
                        await engine.event_persister.drain(timeout=5.0)

            print("")
            print(f"database_stats={engine.database.stats_snapshot()}")
            print(f"event_bus_stats={engine.event_bus.stats_snapshot()}")
            if engine.event_persister is not None:
                print(f"event_persister_stats={engine.event_persister.stats_snapshot()}")
        finally:
            await engine.shutdown()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per endpoint benchmark.",
    )
    parser.add_argument(
        "--event-count",
        type=int,
        default=250,
        help="Number of synthetic run events to emit during the benchmark.",
    )
    args = parser.parse_args()
    return asyncio.run(
        main_async(
            iterations=max(1, int(args.iterations)),
            event_count=max(0, int(args.event_count)),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
