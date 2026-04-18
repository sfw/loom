"""`loomd` sidecar entrypoint."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import click
import uvicorn

from loom.api.server import create_app
from loom.cli.context import _resolve_config_path
from loom.config import Config, ConfigError, load_config
from loom.daemon.desktop_ownership import (
    clear_sidecar_state,
    monitor_desktop_lease,
    write_sidecar_state,
)


def _with_runtime_overrides(
    config: Config,
    *,
    host: str | None,
    port: int | None,
    database_path: Path | None,
    scratch_dir: Path | None,
    workspace_default_path: Path | None,
) -> Config:
    updated = config
    if host is not None or port is not None:
        updated = replace(
            updated,
            server=replace(
                updated.server,
                host=host if host is not None else updated.server.host,
                port=int(port if port is not None else updated.server.port),
            ),
        )
    if database_path is not None:
        updated = replace(
            updated,
            memory=replace(updated.memory, database_path=str(database_path.expanduser())),
        )
    if scratch_dir is not None or workspace_default_path is not None:
        updated = replace(
            updated,
            workspace=replace(
                updated.workspace,
                scratch_dir=(
                    str(scratch_dir.expanduser())
                    if scratch_dir is not None
                    else updated.workspace.scratch_dir
                ),
                default_path=(
                    str(workspace_default_path.expanduser())
                    if workspace_default_path is not None
                    else updated.workspace.default_path
                ),
            ),
        )
    return updated


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to loom.toml configuration file.",
)
@click.option("--host", default=None, help="Override bind host.")
@click.option("--port", default=None, type=int, help="Override bind port.")
@click.option(
    "--database-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Explicit SQLite database path for the sidecar runtime.",
)
@click.option(
    "--scratch-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Explicit scratch/state directory for the sidecar runtime.",
)
@click.option(
    "--workspace-default-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Default workspace root used by desktop workspace pickers.",
)
@click.option(
    "--desktop-instance-token",
    type=str,
    default=None,
    help="Opaque desktop ownership token used to verify loomd ownership.",
)
@click.option(
    "--desktop-lease-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to the desktop lease heartbeat file.",
)
@click.option(
    "--desktop-sidecar-state-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path where loomd publishes its verified sidecar state.",
)
def main(
    config_path: Path | None,
    host: str | None,
    port: int | None,
    database_path: Path | None,
    scratch_dir: Path | None,
    workspace_default_path: Path | None,
    desktop_instance_token: str | None,
    desktop_lease_path: Path | None,
    desktop_sidecar_state_path: Path | None,
) -> None:
    """Start the Loom desktop sidecar daemon."""
    resolved_config_path = _resolve_config_path(config_path)
    try:
        config = load_config(resolved_config_path)
    except ConfigError as e:
        raise SystemExit(f"Configuration error: {e}") from e

    config = _with_runtime_overrides(
        config,
        host=host,
        port=port,
        database_path=database_path,
        scratch_dir=scratch_dir,
        workspace_default_path=workspace_default_path,
    )

    app = create_app(config, runtime_role="loomd")

    if desktop_instance_token and desktop_lease_path and desktop_sidecar_state_path:
        async def _serve_with_desktop_lease() -> None:
            server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    host=config.server.host,
                    port=int(config.server.port),
                    log_level="info",
                ),
            )
            write_sidecar_state(
                path=desktop_sidecar_state_path,
                instance_id=desktop_instance_token,
                host=config.server.host,
                port=int(config.server.port),
                database_path=Path(config.memory.database_path).expanduser(),
                lease_path=desktop_lease_path.expanduser(),
            )
            watcher = asyncio.create_task(
                monitor_desktop_lease(
                    app=app,
                    server=server,
                    expected_instance_id=desktop_instance_token,
                    lease_path=desktop_lease_path.expanduser(),
                ),
            )
            try:
                await server.serve()
            finally:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
                clear_sidecar_state(
                    desktop_sidecar_state_path.expanduser(),
                    expected_instance_id=desktop_instance_token,
                )

        asyncio.run(_serve_with_desktop_lease())
        return

    uvicorn.run(
        app,
        host=config.server.host,
        port=int(config.server.port),
        log_level="info",
    )


if __name__ == "__main__":
    main()
