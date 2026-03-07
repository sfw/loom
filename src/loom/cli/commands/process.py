"""Process-management CLI commands."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import click

from loom.cli.context import _effective_config


@click.command(name="processes")
@click.option(
    "--workspace",
    "-w",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Workspace to search for local process definitions.",
)
@click.pass_context
def processes(ctx: click.Context, workspace: Path | None) -> None:
    """List available process definitions."""
    from loom.processes.schema import ProcessLoader

    config = _effective_config(ctx, workspace)
    ws = (workspace or Path.cwd()).resolve()
    extra = [Path(p) for p in config.process.search_paths]
    loader = ProcessLoader(
        workspace=ws,
        extra_search_paths=extra,
        require_rule_scope_metadata=bool(
            getattr(config.process, "require_rule_scope_metadata", False),
        ),
        require_v2_contract=bool(
            getattr(config.process, "require_v2_contract", False),
        ),
    )
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


@click.group()
def process() -> None:
    """Process subcommands."""


@process.command(name="test")
@click.argument("name_or_path")
@click.option(
    "--workspace",
    "-w",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help=(
        "Workspace for process execution and local process discovery. "
        "Defaults to current directory for discovery and a temporary directory "
        "for execution."
    ),
)
@click.option(
    "--live",
    is_flag=True,
    default=False,
    help="Include live test cases from process.yaml (requires configured models).",
)
@click.option(
    "--case",
    "case_id",
    default=None,
    help="Run a single process test case by ID.",
)
@click.pass_context
def process_test(
    ctx: click.Context,
    name_or_path: str,
    workspace: Path | None,
    live: bool,
    case_id: str | None,
) -> None:
    """Run declared (or default) process test cases.

    NAME_OR_PATH can be either a process name from discovery or a direct
    path to a process YAML/package directory.
    """
    from loom.processes.schema import ProcessLoader
    from loom.processes.testing import run_process_tests

    discovery_ws = (workspace or Path.cwd()).resolve()
    execution_ws = (
        workspace.resolve()
        if workspace is not None
        else Path(tempfile.mkdtemp(prefix="loom-process-test-")).resolve()
    )
    config = _effective_config(ctx, discovery_ws)
    extra = [Path(p) for p in config.process.search_paths]
    loader = ProcessLoader(
        workspace=discovery_ws,
        extra_search_paths=extra,
        require_rule_scope_metadata=bool(
            getattr(config.process, "require_rule_scope_metadata", False),
        ),
        require_v2_contract=bool(
            getattr(config.process, "require_v2_contract", False),
        ),
    )

    try:
        process_def = loader.load(name_or_path)
    except Exception as e:
        click.echo(f"Failed to load process {name_or_path!r}: {e}", err=True)
        sys.exit(1)

    click.echo(f"Running process tests for {process_def.name} v{process_def.version}")
    if workspace is None:
        click.echo(f"Using temporary test workspace: {execution_ws}")

    try:
        results = asyncio.run(
            run_process_tests(
                process_def,
                config=config,
                workspace=execution_ws,
                include_live=live,
                case_id=case_id,
            )
        )
    except ValueError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    if not results:
        click.echo("No matching process test cases selected.")
        sys.exit(1)

    failed = 0
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        click.echo(
            f"[{status}] case={result.case_id} mode={result.mode} "
            f"task_status={result.task_status or 'n/a'} "
            f"duration={result.duration_seconds:.2f}s"
        )
        if result.message:
            click.echo(f"  {result.message}")
        for detail in result.details:
            click.echo(f"  - {detail}")
        if not result.passed:
            failed += 1

    click.echo(f"\n{len(results) - failed}/{len(results)} case(s) passed.")
    if failed:
        sys.exit(1)


@click.command(name="install")
@click.argument("source")
@click.option(
    "--workspace",
    "-w",
    "install_workspace",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Install to <workspace>/loom-processes/ instead of global ~/.loom/processes/.",
)
@click.option(
    "--skip-deps",
    is_flag=True,
    default=False,
    help="Skip installing Python dependencies.",
)
@click.option(
    "--isolated-deps",
    is_flag=True,
    default=False,
    help=(
        "Install package dependencies into <target>/.deps/<process-name>/ "
        "instead of the current Python environment."
    ),
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip interactive review and approve automatically.",
)
@click.pass_context
def install(
    ctx: click.Context,
    source: str,
    install_workspace: Path | None,
    skip_deps: bool,
    isolated_deps: bool,
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
      loom install ./my-local-process --isolated-deps
    """
    from loom.processes.installer import (
        InstallError,
        format_review_for_terminal,
        install_process,
    )

    del ctx  # Included for signature parity with other commands.

    if install_workspace:
        target_dir = install_workspace.resolve() / "loom-processes"
    else:
        target_dir = Path.home() / ".loom" / "processes"

    click.echo(f"Resolving source: {source}")
    if isolated_deps and not skip_deps:
        click.echo(
            "Dependency mode: isolated "
            "(per-process env under <target>/.deps/...)"
        )
    elif isolated_deps and skip_deps:
        click.echo("Note: --isolated-deps has no effect when --skip-deps is set.")

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
            isolated_deps=isolated_deps,
            review_callback=_review_and_prompt,
        )
        click.echo(f"Installed to: {dest}")
        click.echo("Done. Use --process <name> with 'run' or 'cowork'.")
    except InstallError as e:
        click.echo(f"Install failed: {e}", err=True)
        sys.exit(1)


@click.command(name="uninstall")
@click.argument("name")
@click.option(
    "--workspace",
    "-w",
    "uninstall_workspace",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Also search <workspace>/loom-processes/.",
)
@click.confirmation_option(prompt="Are you sure you want to remove this process?")
@click.pass_context
def uninstall(
    ctx: click.Context,
    name: str,
    uninstall_workspace: Path | None,
) -> None:
    """Remove an installed process package by name.

    Only removes user-installed processes. Built-in processes cannot be
    removed.
    """
    from loom.processes.installer import UninstallError, uninstall_process

    del ctx  # Included for signature parity with other commands.

    search_dirs = [Path.home() / ".loom" / "processes"]
    if uninstall_workspace:
        search_dirs.append(uninstall_workspace.resolve() / "loom-processes")

    try:
        removed = uninstall_process(name, search_dirs=search_dirs)
        click.echo(f"Removed: {removed}")
    except UninstallError as e:
        click.echo(f"Uninstall failed: {e}", err=True)
        sys.exit(1)


def register_process_commands(cli_group: click.Group) -> None:
    cli_group.add_command(processes)
    cli_group.add_command(process)
    cli_group.add_command(install)
    cli_group.add_command(uninstall)
