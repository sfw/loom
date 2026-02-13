"""CLI entry point for Loom."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from loom import __version__
from loom.config import load_config


@click.group()
@click.version_option(version=__version__, prog_name="loom")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to loom.toml configuration file.",
)
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None) -> None:
    """Loom — Local model orchestration engine."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)


@cli.command()
@click.option("--host", default=None, help="Override server host.")
@click.option("--port", default=None, type=int, help="Override server port.")
@click.pass_context
def serve(ctx: click.Context, host: str | None, port: int | None) -> None:
    """Start the Loom API server."""
    config = ctx.obj["config"]
    actual_host = host or config.server.host
    actual_port = port or config.server.port

    click.echo(f"Starting Loom server on {actual_host}:{actual_port}")

    import uvicorn

    from loom.api.server import create_app

    app = create_app(config)
    uvicorn.run(app, host=actual_host, port=actual_port, log_level="info")


@cli.command()
@click.option("--server", "server_url", default=None, help="Server URL to connect to.")
@click.pass_context
def tui(ctx: click.Context, server_url: str | None) -> None:
    """Launch the terminal UI."""
    config = ctx.obj["config"]
    url = server_url or f"http://{config.server.host}:{config.server.port}"
    click.echo(f"Connecting TUI to {url}")
    # TUI implementation in Phase 2 (Spec 09)
    click.echo("TUI not yet implemented. Use 'loom serve' and the REST API.")


@cli.command()
@click.argument("goal")
@click.option("--workspace", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--server", "server_url", default=None, help="Server URL.")
@click.pass_context
def run(ctx: click.Context, goal: str, workspace: Path | None, server_url: str | None) -> None:
    """Submit a task and stream progress inline."""
    config = ctx.obj["config"]
    url = server_url or f"http://{config.server.host}:{config.server.port}"
    ws = str(workspace.resolve()) if workspace else None

    click.echo(f"Submitting task to {url}: {goal}")
    if ws:
        click.echo(f"Workspace: {ws}")

    import asyncio

    asyncio.run(_run_task(url, goal, ws))


async def _run_task(server_url: str, goal: str, workspace: str | None) -> None:
    """Submit task and stream progress."""
    import httpx

    async with httpx.AsyncClient(base_url=server_url, timeout=300) as client:
        payload: dict = {"goal": goal}
        if workspace:
            payload["workspace"] = workspace

        response = await client.post("/tasks", json=payload)
        if response.status_code != 201:
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


@cli.command()
@click.argument("task_id")
@click.option("--server", "server_url", default=None, help="Server URL.")
@click.pass_context
def status(ctx: click.Context, task_id: str, server_url: str | None) -> None:
    """Check status of a task."""
    config = ctx.obj["config"]
    url = server_url or f"http://{config.server.host}:{config.server.port}"

    import asyncio

    asyncio.run(_check_status(url, task_id))


async def _check_status(server_url: str, task_id: str) -> None:
    """Fetch and display task status."""
    import httpx

    async with httpx.AsyncClient(base_url=server_url) as client:
        response = await client.get(f"/tasks/{task_id}")
        if response.status_code == 404:
            click.echo(f"Task not found: {task_id}", err=True)
            sys.exit(1)
        data = response.json()
        click.echo(f"Task:   {data['task_id']}")
        click.echo(f"Status: {data['status']}")
        click.echo(f"Goal:   {data.get('goal', 'N/A')}")


@cli.command()
@click.argument("task_id")
@click.option("--server", "server_url", default=None, help="Server URL.")
@click.pass_context
def cancel(ctx: click.Context, task_id: str, server_url: str | None) -> None:
    """Cancel a running task."""
    config = ctx.obj["config"]
    url = server_url or f"http://{config.server.host}:{config.server.port}"

    import asyncio

    asyncio.run(_cancel_task(url, task_id))


async def _cancel_task(server_url: str, task_id: str) -> None:
    """Cancel a task."""
    import httpx

    async with httpx.AsyncClient(base_url=server_url) as client:
        response = await client.post(f"/tasks/{task_id}/cancel")
        if response.status_code == 404:
            click.echo(f"Task not found: {task_id}", err=True)
            sys.exit(1)
        click.echo(f"Task {task_id} cancelled.")


@cli.command()
@click.option("--server", "server_url", default=None, help="Server URL.")
@click.pass_context
def models(ctx: click.Context, server_url: str | None) -> None:
    """List available models and their status."""
    config = ctx.obj["config"]

    if not config.models:
        click.echo("No models configured. Add model sections to loom.toml.")
        return

    for name, model in config.models.items():
        roles = ", ".join(model.roles)
        click.echo(f"  {name}: {model.model} ({model.provider}) [{roles}]")
        click.echo(f"    URL: {model.base_url}")


def main() -> None:
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
