"""HTTP task helpers used by CLI commands."""

from __future__ import annotations

import sys

import click


def _validate_task_id(task_id: str) -> str:
    """Validate task_id contains only safe characters for URL interpolation."""
    import re

    if not re.match(r"^[a-zA-Z0-9_-]+$", task_id):
        click.echo(f"Invalid task ID: {task_id}", err=True)
        sys.exit(1)
    return task_id


async def _run_task(
    server_url: str,
    goal: str,
    workspace: str | None,
    process_name: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Submit task and stream progress."""
    import httpx

    try:
        async with httpx.AsyncClient(base_url=server_url, timeout=300) as client:
            payload: dict = {"goal": goal}
            if workspace:
                payload["workspace"] = workspace
            if process_name:
                payload["process"] = process_name
            if metadata:
                payload["metadata"] = metadata

            response = await client.post("/tasks", json=payload)
            if response.status_code != 201:
                try:
                    error_payload = response.json()
                except Exception:
                    click.echo(f"Error: {response.text}", err=True)
                    sys.exit(1)
                detail = error_payload.get("detail")
                if isinstance(detail, dict) and detail.get("code") == "auth_unresolved":
                    click.echo(
                        "Error: Auth preflight failed: unresolved required auth resources.",
                        err=True,
                    )
                    unresolved = detail.get("unresolved", [])
                    if isinstance(unresolved, list):
                        for item in unresolved:
                            if not isinstance(item, dict):
                                continue
                            provider = str(item.get("provider", "")).strip() or "unknown"
                            source = str(item.get("source", "")).strip() or "unknown"
                            reason = str(item.get("reason", "")).strip() or "unresolved"
                            message = str(item.get("message", "")).strip()
                            click.echo(
                                f"  - provider={provider} source={source} reason={reason}",
                                err=True,
                            )
                            if message:
                                click.echo(f"    {message}", err=True)
                            candidates = item.get("candidates", [])
                            if isinstance(candidates, list) and candidates:
                                click.echo(
                                    f"    candidates: {', '.join(str(c) for c in candidates)}",
                                    err=True,
                                )
                    remediation = detail.get("remediation", {})
                    commands = (
                        remediation.get("commands", [])
                        if isinstance(remediation, dict)
                        else []
                    )
                    if isinstance(commands, list) and commands:
                        click.echo("  Suggested commands:", err=True)
                        for command in commands[:6]:
                            click.echo(f"    {command}", err=True)
                    sys.exit(1)
                click.echo(f"Error: {response.text}", err=True)
                sys.exit(1)

            task = response.json()
            task_id = task["task_id"]
            click.echo(f"Task created: {task_id}")

            # Stream events
            async with client.stream("GET", f"/tasks/{task_id}/stream") as stream:
                async for line in stream.aiter_lines():
                    if not line.strip() or line.startswith(":"):
                        continue
                    if line.startswith("data: "):
                        click.echo(line[6:])
    except httpx.ConnectError:
        click.echo(f"Error: Cannot connect to server at {server_url}", err=True)
        sys.exit(1)
    except httpx.TimeoutException:
        click.echo("Error: Request timed out", err=True)
        sys.exit(1)


async def _check_status(server_url: str, task_id: str) -> None:
    """Fetch and display task status."""
    task_id = _validate_task_id(task_id)
    import httpx

    try:
        async with httpx.AsyncClient(base_url=server_url) as client:
            response = await client.get(f"/tasks/{task_id}")
            if response.status_code == 404:
                click.echo(f"Task not found: {task_id}", err=True)
                sys.exit(1)
            data = response.json()
            click.echo(f"Task:   {data['task_id']}")
            click.echo(f"Status: {data['status']}")
            click.echo(f"Goal:   {data.get('goal', 'N/A')}")
    except httpx.ConnectError:
        click.echo(
            f"Error: Cannot connect to server at {server_url}",
            err=True,
        )
        sys.exit(1)


async def _cancel_task(server_url: str, task_id: str) -> None:
    """Cancel a task."""
    task_id = _validate_task_id(task_id)
    import httpx

    try:
        async with httpx.AsyncClient(base_url=server_url) as client:
            response = await client.delete(f"/tasks/{task_id}")
            if response.status_code == 404:
                click.echo(f"Task not found: {task_id}", err=True)
                sys.exit(1)
            if response.status_code >= 400:
                msg = f"Cancel failed ({response.status_code}): {response.text[:200]}"
                click.echo(msg, err=True)
                sys.exit(1)
            click.echo(f"Task {task_id} cancelled.")
    except httpx.ConnectError:
        click.echo(
            f"Error: Cannot connect to server at {server_url}",
            err=True,
        )
        sys.exit(1)
