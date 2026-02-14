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

    try:
        app = create_app(config)
        uvicorn.run(
            app, host=actual_host, port=actual_port, log_level="info",
        )
    except Exception as e:
        click.echo(f"Server failed to start: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--workspace", "-w",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Workspace directory. Defaults to current directory.",
)
@click.option("--model", "-m", default=None, help="Model name from config to use.")
@click.pass_context
def tui(ctx: click.Context, workspace: Path | None, model: str | None) -> None:
    """Launch the Textual TUI for interactive cowork.

    Same capabilities as 'loom cowork' but with a richer terminal interface:
    modal dialogs for tool approval and ask_user, scrollable chat log, etc.
    No server required — runs the model directly.
    """
    config = ctx.obj["config"]
    ws = (workspace or Path.cwd()).resolve()

    from loom.models.router import ModelRouter
    from loom.tools import create_default_registry
    from loom.tui.app import LoomApp

    router = ModelRouter.from_config(config)
    if model:
        provider = None
        for name, p in router._providers.items():
            if name == model:
                provider = p
                break
        if provider is None:
            click.echo(f"Model '{model}' not found in config.", err=True)
            sys.exit(1)
    else:
        try:
            provider = router.select(role="executor")
        except Exception as e:
            click.echo(f"No model available: {e}", err=True)
            sys.exit(1)

    tools = create_default_registry()
    app = LoomApp(model=provider, tools=tools, workspace=ws)
    app.run()


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

    try:
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
    except httpx.ConnectError:
        click.echo(f"Error: Cannot connect to server at {server_url}", err=True)
        sys.exit(1)
    except httpx.TimeoutException:
        click.echo("Error: Request timed out", err=True)
        sys.exit(1)


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

    try:
        async with httpx.AsyncClient(base_url=server_url) as client:
            response = await client.post(f"/tasks/{task_id}/cancel")
            if response.status_code == 404:
                click.echo(f"Task not found: {task_id}", err=True)
                sys.exit(1)
            click.echo(f"Task {task_id} cancelled.")
    except httpx.ConnectError:
        click.echo(
            f"Error: Cannot connect to server at {server_url}",
            err=True,
        )
        sys.exit(1)


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


@cli.command(name="mcp-serve")
@click.option("--server", "server_url", default=None, help="Loom API server URL.")
@click.pass_context
def mcp_serve(ctx: click.Context, server_url: str | None) -> None:
    """Start Loom as an MCP server (stdio transport)."""
    config = ctx.obj["config"]
    url = server_url or f"http://{config.server.host}:{config.server.port}"

    import asyncio

    from loom.integrations.mcp_server import LoomMCPServer

    server = LoomMCPServer(engine_url=url)
    click.echo(f"Starting Loom MCP server (engine: {url})", err=True)
    asyncio.run(server.run_stdio())


@cli.command()
@click.option(
    "--workspace", "-w",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Workspace directory. Defaults to current directory.",
)
@click.option("--model", "-m", default=None, help="Model name from config to use.")
@click.pass_context
def cowork(ctx: click.Context, workspace: Path | None, model: str | None) -> None:
    """Start an interactive cowork session.

    Opens a conversation loop where you and the AI collaborate directly.
    No planning phase, no subtask decomposition — just a continuous
    tool-calling loop driven by natural conversation.
    """
    import asyncio

    config = ctx.obj["config"]
    ws = (workspace or Path.cwd()).resolve()

    asyncio.run(_cowork_session(config, ws, model))


async def _cowork_session(config, workspace: Path, model_name: str | None) -> None:
    """Run an interactive cowork session."""
    from loom.cowork.approval import ToolApprover, async_terminal_approval_prompt
    from loom.cowork.display import (
        display_ask_user,
        display_error,
        display_goodbye,
        display_text_chunk,
        display_tool_complete,
        display_tool_start,
        display_turn_summary,
        display_welcome,
    )
    from loom.cowork.session import (
        CoworkSession,
        CoworkTurn,
        ToolCallEvent,
        build_cowork_system_prompt,
    )
    from loom.models.router import ModelRouter
    from loom.tools import create_default_registry

    # Set up model
    router = ModelRouter.from_config(config)
    if model_name:
        # Try to get the specific named provider
        provider = None
        for name, p in router._providers.items():
            if name == model_name:
                provider = p
                break
        if provider is None:
            display_error(f"Model '{model_name}' not found in config.")
            return
    else:
        try:
            provider = router.select(role="executor")
        except Exception as e:
            display_error(f"No model available: {e}")
            return

    # Set up tools and approval
    tools = create_default_registry()
    approver = ToolApprover(prompt_callback=async_terminal_approval_prompt)

    # Build session
    system_prompt = build_cowork_system_prompt(workspace)
    session = CoworkSession(
        model=provider,
        tools=tools,
        workspace=workspace,
        system_prompt=system_prompt,
        approver=approver,
    )

    display_welcome(workspace, provider.name)

    while True:
        try:
            user_input = input("\033[1m> \033[0m")
        except (EOFError, KeyboardInterrupt):
            display_goodbye()
            break

        if not user_input.strip():
            continue

        # Handle special commands
        if user_input.strip().lower() in ("/quit", "/exit", "/q"):
            display_goodbye()
            break

        if user_input.strip().lower() == "/help":
            sys.stdout.write(
                "Commands: /quit, /exit, /help\n"
                "Type anything else to interact with the AI.\n"
            )
            continue

        try:
            sys.stdout.write("\n")

            streamed_text = False
            async for event in session.send_streaming(user_input):
                if isinstance(event, ToolCallEvent):
                    if event.result is None:
                        display_tool_start(event)
                    else:
                        display_tool_complete(event)

                        # Special handling for ask_user
                        if event.name == "ask_user" and event.result:
                            answer = display_ask_user(event)
                            if answer:
                                sys.stdout.write("\n")
                                async for follow_event in session.send_streaming(answer):
                                    if isinstance(follow_event, ToolCallEvent):
                                        if follow_event.result is None:
                                            display_tool_start(follow_event)
                                        else:
                                            display_tool_complete(follow_event)
                                    elif isinstance(follow_event, CoworkTurn):
                                        display_turn_summary(follow_event)
                                    elif isinstance(follow_event, str):
                                        display_text_chunk(follow_event)

                elif isinstance(event, CoworkTurn):
                    # Text was already streamed incrementally — only
                    # display if no streaming occurred (fallback).
                    if event.text and not streamed_text:
                        sys.stdout.write(f"\n{event.text}\n")
                    display_turn_summary(event)

                elif isinstance(event, str):
                    if not streamed_text:
                        sys.stdout.write("\n")
                        streamed_text = True
                    display_text_chunk(event)

            sys.stdout.write("\n")

        except KeyboardInterrupt:
            sys.stdout.write("\n\033[2m(interrupted)\033[0m\n")
            continue
        except Exception as e:
            display_error(str(e))


@cli.command(name="reset-learning")
@click.confirmation_option(prompt="Are you sure you want to clear all learned patterns?")
@click.pass_context
def reset_learning(ctx: click.Context) -> None:
    """Clear all learned patterns from the database."""
    import asyncio

    from loom.learning.manager import LearningManager
    from loom.state.memory import Database

    config = ctx.obj["config"]

    async def _reset():
        db = Database(str(Path(config.memory.database_path).expanduser()))
        await db.initialize()
        manager = LearningManager(db)
        await manager.clear_all()
        click.echo("Learning database cleared.")

    asyncio.run(_reset())


def main() -> None:
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
