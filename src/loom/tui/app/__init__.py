"""Public compatibility facade for the Loom TUI app package."""

from __future__ import annotations

import sys
import types

from . import core as _core
from .core import LoomApp
from .models import ProcessRunLaunchRequest, ProcessRunState, SteeringDirective
from .widgets import ProcessRunList, ProcessRunPane

__all__ = [
    "LoomApp",
    "ProcessRunLaunchRequest",
    "ProcessRunList",
    "ProcessRunPane",
    "ProcessRunState",
    "SteeringDirective",
]


def __getattr__(name: str):
    if hasattr(_core, name):
        return getattr(_core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(dir(_core)))


class _FacadeModule(types.ModuleType):
    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if hasattr(_core, name):
            setattr(_core, name, value)

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)
        if hasattr(_core, name):
            delattr(_core, name)


sys.modules[__name__].__class__ = _FacadeModule
