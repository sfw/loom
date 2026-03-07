"""CLI entry point compatibility facade for `python -m loom`."""

from loom.cli import (
    PersistenceInitError,
    _cancel_task,
    _check_status,
    _init_persistence,
    cli,
    main,
)

__all__ = [
    "cli",
    "main",
    "PersistenceInitError",
    "_init_persistence",
    "_check_status",
    "_cancel_task",
]

if __name__ == "__main__":
    main()
