"""Schema models for runtime-editable Loom config entries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

APPLICATION_LIVE = "live"
APPLICATION_NEXT_CALL = "next_call"
APPLICATION_NEXT_RUN = "next_run"
APPLICATION_RESTART_REQUIRED = "restart_required"

ApplicationClass = Literal[
    "live",
    "next_call",
    "next_run",
    "restart_required",
]
ConfigValueKind = Literal["int", "bool", "enum", "string"]
ExposureLevel = Literal["basic", "advanced"]


@dataclass(frozen=True)
class ParsedConfigValue:
    """Normalized user-provided config value."""

    value: Any
    display_value: str
    warning_code: str = ""


@dataclass(frozen=True)
class ConfigRuntimeEntry:
    """Metadata for one operator-visible config path."""

    path: str
    section: str
    field: str
    kind: ConfigValueKind
    description: str
    default: Any
    supports_runtime: bool
    supports_persist: bool
    application_class: ApplicationClass
    requires_restart: bool = False
    redact_in_output: bool = False
    aliases: tuple[str, ...] = ()
    enum_values: tuple[str, ...] = ()
    minimum: int | None = None
    maximum: int | None = None
    parser: Callable[[object], ParsedConfigValue] | None = None
    search_terms: tuple[str, ...] = field(default_factory=tuple)
    exposure_level: ExposureLevel = "advanced"

    def parse(self, raw_value: object) -> ParsedConfigValue:
        if self.parser is not None:
            return self.parser(raw_value)
        raise ValueError(f"No parser registered for config path {self.path!r}.")

    @property
    def section_title(self) -> str:
        return self.section.replace("_", " ")
