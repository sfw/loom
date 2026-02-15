"""CLI entry point for Loom."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from loom import __version__
from loom.config import Config, load_config


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
    actual_host = host if host is not None else config.server.host
    actual_port = port if port is not None else config.server.port

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
@click.option(
    "--process", "process_name", default=None,
    help="Process definition name or path.",
)
@click.pass_context
def run(
    ctx: click.Context, goal: str, workspace: Path | None,
    server_url: str | None, process_name: str | None,
) -> None:
    """Submit a task and stream progress inline."""
    config = ctx.obj["config"]
    url = server_url or f"http://{config.server.host}:{config.server.port}"
    ws = str(workspace.resolve()) if workspace else None

    click.echo(f"Submitting task to {url}: {goal}")
    if ws:
        click.echo(f"Workspace: {ws}")
    if process_name:
        click.echo(f"Process: {process_name}")

    import asyncio

    asyncio.run(_run_task(url, goal, ws, process_name=process_name))


def _validate_task_id(task_id: str) -> str:
    """Validate task_id contains only safe characters for URL interpolation."""
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', task_id):
        click.echo(f"Invalid task ID: {task_id}", err=True)
        sys.exit(1)
    return task_id


async def _run_task(
    server_url: str, goal: str, workspace: str | None,
    process_name: str | None = None,
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
    task_id = _validate_task_id(task_id)
    import httpx

    try:
        async with httpx.AsyncClient(base_url=server_url) as client:
            response = await client.post(f"/tasks/{task_id}/cancel")
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


@cli.command()
@click.pass_context
def models(ctx: click.Context) -> None:
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
@click.option("--resume", "resume_session", default=None, help="Resume a previous session by ID.")
@click.option(
    "--process", "process_name", default=None,
    help="Process definition name or path.",
)
@click.pass_context
def cowork(
    ctx: click.Context,
    workspace: Path | None,
    model: str | None,
    resume_session: str | None,
    process_name: str | None,
) -> None:
    """Start an interactive cowork session.

    Opens a conversation loop where you and the AI collaborate directly.
    No planning phase, no subtask decomposition — just a continuous
    tool-calling loop driven by natural conversation.

    All conversation history is persisted to SQLite. Use --resume to
    continue a previous session.
    """
    import asyncio

    config = ctx.obj["config"]
    ws = (workspace or Path.cwd()).resolve()

    asyncio.run(_cowork_session(config, ws, model, resume_session, process_name))


async def _cowork_session(
    config: Config, workspace: Path, model_name: str | None,
    resume_id: str | None = None,
    process_name: str | None = None,
) -> None:
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
    from loom.state.conversation_store import ConversationStore
    from loom.state.memory import Database
    from loom.tools import create_default_registry

    # Set up model
    router = ModelRouter.from_config(config)
    if model_name:
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

    # Set up database and conversation store
    if hasattr(config, "workspace"):
        data_dir = Path(config.workspace.scratch_dir).expanduser()
    else:
        data_dir = Path.home() / ".loom"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "loom.db"
    db = Database(db_path)
    await db.initialize()
    store = ConversationStore(db)

    # Set up tools
    tools = create_default_registry()

    # Register conversation_recall tool (needs binding after session creation)
    from loom.tools.conversation_recall import ConversationRecallTool
    recall_tool = ConversationRecallTool()
    tools.register(recall_tool)

    # Register delegate_task tool
    from loom.tools.delegate_task import DelegateTaskTool
    delegate_tool = DelegateTaskTool()
    tools.register(delegate_tool)

    approver = ToolApprover(prompt_callback=async_terminal_approval_prompt)

    # Load process definition if specified
    process_defn = None
    if process_name:
        from loom.processes.schema import ProcessLoader

        extra = [Path(p) for p in config.process.search_paths]
        loader = ProcessLoader(workspace=workspace, extra_search_paths=extra)
        try:
            process_defn = loader.load(process_name)
            click.echo(f"Loaded process: {process_defn.name} v{process_defn.version}")

            # Apply tool exclusions from process
            if process_defn.tools.excluded:
                for tool_name in process_defn.tools.excluded:
                    if tool_name in tools._tools:
                        del tools._tools[tool_name]
        except Exception as e:
            display_error(f"Failed to load process '{process_name}': {e}")
            return

    # Build session
    system_prompt = build_cowork_system_prompt(workspace)

    # Inject process persona and guidance into system prompt
    if process_defn:
        process_additions = []
        if process_defn.persona:
            process_additions.append(
                f"\n\nDOMAIN ROLE:\n{process_defn.persona.strip()}"
            )
        if process_defn.tool_guidance:
            process_additions.append(
                f"\n\nDOMAIN TOOL GUIDANCE:\n{process_defn.tool_guidance.strip()}"
            )
        if process_additions:
            system_prompt += "".join(process_additions)

    if resume_id:
        # Resume existing session
        session = CoworkSession(
            model=provider,
            tools=tools,
            workspace=workspace,
            system_prompt=system_prompt,
            approver=approver,
            store=store,
        )
        try:
            await session.resume(resume_id)
            sys.stdout.write(
                f"\033[2mResumed session {resume_id} "
                f"({session.session_state.turn_count} turns in archive)\033[0m\n"
            )
        except (ValueError, RuntimeError) as e:
            display_error(str(e))
            return
    else:
        # Check for previous sessions in this workspace
        previous = await store.list_sessions(workspace=str(workspace))
        session_id = ""

        if previous:
            sys.stdout.write("\n\033[2mPrevious sessions for this workspace:\033[0m\n")
            for i, s in enumerate(previous[:5], 1):
                turns = s.get("turn_count", 0)
                started = s.get("started_at", "?")[:16]
                sid = s["id"]
                sys.stdout.write(f"  \033[2m[{i}]\033[0m {started} — {turns} turns (id: {sid})\n")
            sys.stdout.write("  \033[2m[n]\033[0m Start new session\n")
            sys.stdout.flush()

            try:
                choice = input("\033[2mResume or new? \033[0m").strip().lower()
            except (EOFError, KeyboardInterrupt):
                display_goodbye()
                return

            if choice.isdigit() and 1 <= int(choice) <= len(previous[:5]):
                session_id = previous[int(choice) - 1]["id"]
            elif choice and choice != "n":
                display_error(f"Invalid choice: {choice}")
                return

        if session_id:
            # Resume selected session
            session = CoworkSession(
                model=provider,
                tools=tools,
                workspace=workspace,
                system_prompt=system_prompt,
                approver=approver,
                store=store,
            )
            await session.resume(session_id)
            sys.stdout.write(
                f"\033[2mResumed session {session_id} "
                f"({session.session_state.turn_count} turns in archive)\033[0m\n"
            )
        else:
            # New session
            session_id = await store.create_session(
                workspace=str(workspace),
                model_name=provider.name,
                system_prompt=system_prompt,
            )
            session = CoworkSession(
                model=provider,
                tools=tools,
                workspace=workspace,
                system_prompt=system_prompt,
                approver=approver,
                store=store,
                session_id=session_id,
            )

    # -- Helper: create or switch to a session --------------------------------

    async def _orchestrator_factory():
        from loom.engine.orchestrator import Orchestrator
        from loom.events.bus import EventBus
        from loom.prompts.assembler import PromptAssembler
        from loom.state.memory import MemoryManager
        from loom.state.task_state import TaskStateManager
        from loom.tools import create_default_registry as _create_tools

        return Orchestrator(
            model_router=router,
            tool_registry=_create_tools(),
            memory_manager=MemoryManager(db),
            prompt_assembler=PromptAssembler(),
            state_manager=TaskStateManager(data_dir),  # uses data_dir from outer scope
            event_bus=EventBus(),
            config=config,
        )

    def _bind_session_tools(s: CoworkSession) -> None:
        """Rebind tools that hold a reference to the active session."""
        recall_tool.bind(store=store, session_id=s.session_id, session_state=s.session_state)
        delegate_tool.bind(_orchestrator_factory)

    async def _new_session(ws: Path) -> CoworkSession:
        """Create a fresh session for the given workspace."""
        prompt = build_cowork_system_prompt(ws)
        sid = await store.create_session(
            workspace=str(ws), model_name=provider.name, system_prompt=prompt,
        )
        s = CoworkSession(
            model=provider, tools=tools, workspace=ws,
            system_prompt=prompt, approver=approver,
            store=store, session_id=sid,
        )
        _bind_session_tools(s)
        return s

    async def _resume_session(sid: str) -> CoworkSession:
        """Resume an existing session by ID."""
        sess_row = await store.get_session(sid)
        if sess_row is None:
            raise ValueError(f"Session not found: {sid}")
        ws = Path(sess_row["workspace_path"])
        prompt = build_cowork_system_prompt(ws)
        s = CoworkSession(
            model=provider, tools=tools, workspace=ws,
            system_prompt=prompt, approver=approver, store=store,
        )
        await s.resume(sid)
        _bind_session_tools(s)
        return s

    async def _switch_session() -> CoworkSession | None:
        """Interactive session picker. Returns new session or None on cancel."""
        all_sessions = await store.list_sessions()

        if not all_sessions:
            sys.stdout.write("\033[2mNo previous sessions. Starting new.\033[0m\n")
            return await _new_session(workspace)

        # Group by workspace
        by_ws: dict[str, list[dict]] = {}
        for s in all_sessions:
            ws_path = s.get("workspace_path", "?")
            by_ws.setdefault(ws_path, []).append(s)

        sys.stdout.write("\n\033[1mSessions:\033[0m\n")
        flat: list[dict] = []
        for ws_path, sessions in by_ws.items():
            sys.stdout.write(f"  \033[36m{ws_path}\033[0m\n")
            for s in sessions[:5]:
                idx = len(flat) + 1
                turns = s.get("turn_count", 0)
                started = s.get("started_at", "?")[:16]
                sid = s["id"]
                active = " \033[32m(active)\033[0m" if s.get("is_active") else ""
                sys.stdout.write(
                    f"    \033[2m[{idx}]\033[0m {started} — {turns} turns ({sid}){active}\n"
                )
                flat.append(s)
        sys.stdout.write("    \033[2m[n]\033[0m New session (current workspace)\n")
        sys.stdout.write("    \033[2m[c]\033[0m Cancel\n")
        sys.stdout.flush()

        try:
            choice = input("\033[2mChoice: \033[0m").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None

        if choice == "c" or not choice:
            return None
        if choice == "n":
            return await _new_session(workspace)
        if choice.isdigit() and 1 <= int(choice) <= len(flat):
            selected = flat[int(choice) - 1]
            s = await _resume_session(selected["id"])
            sys.stdout.write(
                f"\033[2mSwitched to session {s.session_id} "
                f"({s.session_state.turn_count} turns, "
                f"workspace: {s.workspace})\033[0m\n"
            )
            return s

        display_error(f"Invalid choice: {choice}")
        return None

    # -- Initialize first session -------------------------------------------

    _bind_session_tools(session)
    display_welcome(workspace, provider.name)

    # -- Main loop -----------------------------------------------------------

    while True:
        try:
            user_input = input("\033[1m> \033[0m")
        except (EOFError, KeyboardInterrupt):
            await store.update_session(session.session_id, is_active=False)
            display_goodbye()
            break

        if not user_input.strip():
            continue

        cmd = user_input.strip().lower()

        # Handle special commands
        if cmd in ("/quit", "/exit", "/q"):
            await store.update_session(session.session_id, is_active=False)
            display_goodbye()
            break

        if cmd == "/help":
            sys.stdout.write(
                "Commands:\n"
                "  /sessions  — list and switch between sessions\n"
                "  /new       — start a new session (current workspace)\n"
                "  /session   — show current session info\n"
                "  /quit      — exit (also /exit, /q)\n"
                "  /help      — this message\n"
                "Type anything else to interact with the AI.\n"
            )
            continue

        if cmd == "/session":
            state = session.session_state
            sys.stdout.write(
                f"Session: {session.session_id}\n"
                f"Workspace: {session.workspace}\n"
                f"Turns: {state.turn_count}\n"
                f"Tokens: {state.total_tokens}\n"
                f"Focus: {state.current_focus or '(none)'}\n"
            )
            if state.key_decisions:
                sys.stdout.write("Decisions:\n")
                for d in state.key_decisions[-5:]:
                    sys.stdout.write(f"  - {d}\n")
            continue

        if cmd == "/sessions":
            # Mark current session inactive BEFORE switching to avoid inconsistency
            old_session_id = session.session_id
            new_session = await _switch_session()
            if new_session is not None:
                await store.update_session(old_session_id, is_active=False)
                session = new_session
            continue

        if cmd == "/new":
            await store.update_session(session.session_id, is_active=False)
            # Use current session's workspace, not the original startup workspace
            session = await _new_session(session.workspace or workspace)
            sys.stdout.write(f"\033[2mNew session: {session.session_id}\033[0m\n")
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


@cli.command()
@click.option(
    "--workspace", "-w",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Workspace to search for local process definitions.",
)
@click.pass_context
def processes(ctx: click.Context, workspace: Path | None) -> None:
    """List available process definitions."""
    from loom.processes.schema import ProcessLoader

    config = ctx.obj["config"]
    ws = (workspace or Path.cwd()).resolve()
    extra = [Path(p) for p in config.process.search_paths]
    loader = ProcessLoader(workspace=ws, extra_search_paths=extra)
    available = loader.list_available()

    if not available:
        click.echo("No process definitions found.")
        click.echo("  Built-in: src/loom/processes/builtin/")
        click.echo("  User:     ~/.loom/processes/")
        click.echo("  Local:    ./loom-processes/")
        return

    click.echo("Available processes:\n")
    for proc in available:
        name = proc["name"]
        ver = proc["version"]
        desc = proc.get("description", "")
        # Truncate description to one line
        if desc:
            desc = desc.strip().split("\n")[0][:60]
        click.echo(f"  {name:30s} v{ver:6s} {desc}")
    click.echo(
        f"\n{len(available)} process(es) found. "
        f"Use --process <name> with 'run' or 'cowork'.",
    )


@cli.command(name="install")
@click.argument("source")
@click.option(
    "--workspace", "-w", "install_workspace",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Install to <workspace>/loom-processes/ instead of global ~/.loom/processes/.",
)
@click.option(
    "--skip-deps", is_flag=True, default=False,
    help="Skip installing Python dependencies.",
)
@click.option(
    "--yes", "-y", is_flag=True, default=False,
    help="Skip interactive review and approve automatically.",
)
@click.pass_context
def install(
    ctx: click.Context,
    source: str,
    install_workspace: Path | None,
    skip_deps: bool,
    yes: bool,
) -> None:
    """Install a process package from a GitHub repo or local path.

    SOURCE can be:

    \b
      - A GitHub URL: https://github.com/user/loom-my-process
      - A shorthand:  user/loom-my-process
      - A local path:  /path/to/my-process/

    The package must contain a process.yaml at its root. Python dependencies
    listed in the 'dependencies' field of process.yaml are automatically
    installed (use --skip-deps to disable).

    Before installation, you'll see a full security review of the package
    contents (dependencies, bundled code) and must confirm. Use -y to skip
    this review (not recommended for untrusted sources).

    Examples:

    \b
      loom install https://github.com/acme/loom-google-analytics
      loom install acme/loom-google-analytics
      loom install ./my-local-process
      loom install ./my-local-process -w /path/to/project
    """
    from loom.processes.installer import (
        InstallError,
        format_review_for_terminal,
        install_process,
    )

    if install_workspace:
        target_dir = install_workspace.resolve() / "loom-processes"
    else:
        target_dir = Path.home() / ".loom" / "processes"

    click.echo(f"Resolving source: {source}")

    def _review_and_prompt(review) -> bool:
        """Display review and ask user for confirmation."""
        click.echo(format_review_for_terminal(review))
        if yes:
            click.echo("  --yes flag set: auto-approving.")
            return True
        return click.confirm("  Proceed with installation?", default=False)

    try:
        dest = install_process(
            source,
            target_dir=target_dir,
            skip_deps=skip_deps,
            review_callback=_review_and_prompt,
        )
        click.echo(f"Installed to: {dest}")
        click.echo("Done. Use --process <name> with 'run' or 'cowork'.")
    except InstallError as e:
        click.echo(f"Install failed: {e}", err=True)
        sys.exit(1)


@cli.command(name="uninstall")
@click.argument("name")
@click.option(
    "--workspace", "-w", "uninstall_workspace",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Also search <workspace>/loom-processes/.",
)
@click.confirmation_option(prompt="Are you sure you want to remove this process?")
@click.pass_context
def uninstall(
    ctx: click.Context, name: str, uninstall_workspace: Path | None,
) -> None:
    """Remove an installed process package by name.

    Only removes user-installed processes. Built-in processes cannot be
    removed.
    """
    from loom.processes.installer import UninstallError, uninstall_process

    search_dirs = [Path.home() / ".loom" / "processes"]
    if uninstall_workspace:
        search_dirs.append(uninstall_workspace.resolve() / "loom-processes")

    try:
        removed = uninstall_process(name, search_dirs=search_dirs)
        click.echo(f"Removed: {removed}")
    except UninstallError as e:
        click.echo(f"Uninstall failed: {e}", err=True)
        sys.exit(1)


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
