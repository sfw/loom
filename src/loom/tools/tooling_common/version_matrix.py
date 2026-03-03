"""Compatibility matrix for external coding-agent CLIs.

This module centralizes provider identifiers, binary names, and supported
mode combinations used by provider-specific coding-agent tools.
"""

from __future__ import annotations

from dataclasses import dataclass

CANONICAL_SANDBOX_MODES = ("read_only", "workspace_write", "unrestricted")
CANONICAL_APPROVAL_MODES = ("untrusted", "on_failure", "on_request", "never")
CANONICAL_NETWORK_MODES = ("off", "on")
CANONICAL_OUTPUT_MODES = ("text", "json", "stream")


@dataclass(frozen=True)
class ProviderSpec:
    """Static provider-level compatibility contract."""

    provider: str
    binary: str
    version_args: tuple[str, ...]
    run_base_args: tuple[str, ...]
    default_sandbox_mode: str
    default_approval_mode: str
    supports_sandbox_modes: tuple[str, ...]
    supports_approval_modes: tuple[str, ...]
    supports_network_modes: tuple[str, ...]
    supports_output_modes: tuple[str, ...]
    supports_off_network_enforcement: bool = False
    min_supported_version: tuple[int, int, int] | None = None


PROVIDER_SPECS: dict[str, ProviderSpec] = {
    "codex": ProviderSpec(
        provider="codex",
        binary="codex",
        version_args=("--version",),
        run_base_args=("exec",),
        default_sandbox_mode="workspace_write",
        default_approval_mode="on_request",
        supports_sandbox_modes=("read_only", "workspace_write", "unrestricted"),
        supports_approval_modes=("untrusted", "on_failure", "on_request", "never"),
        supports_network_modes=("on",),
        supports_output_modes=("text", "json", "stream"),
        supports_off_network_enforcement=False,
        min_supported_version=(0, 0, 0),
    ),
    "claude_code": ProviderSpec(
        provider="claude_code",
        binary="claude",
        version_args=("--version",),
        run_base_args=("-p",),
        default_sandbox_mode="workspace_write",
        default_approval_mode="on_request",
        supports_sandbox_modes=("workspace_write", "unrestricted"),
        supports_approval_modes=("on_request", "never"),
        supports_network_modes=("on",),
        supports_output_modes=("text", "json", "stream"),
        supports_off_network_enforcement=False,
        min_supported_version=(0, 0, 0),
    ),
    "opencode": ProviderSpec(
        provider="opencode",
        binary="opencode",
        version_args=("--version",),
        run_base_args=("run",),
        default_sandbox_mode="workspace_write",
        default_approval_mode="on_request",
        supports_sandbox_modes=("workspace_write",),
        supports_approval_modes=("on_request",),
        supports_network_modes=("on",),
        supports_output_modes=("text", "json", "stream"),
        supports_off_network_enforcement=False,
        min_supported_version=(0, 0, 0),
    ),
}


def normalize_provider(value: object) -> str:
    """Normalize loose provider aliases to canonical ids."""
    raw = str(value or "").strip().lower()
    if raw in PROVIDER_SPECS:
        return raw
    if raw in {"claude", "claude-code", "claude_code"}:
        return "claude_code"
    return raw


def parse_semver_tuple(version_text: str) -> tuple[int, int, int] | None:
    """Best-effort parse of MAJOR.MINOR.PATCH tuple from version output."""
    text = str(version_text or "")
    digits: list[str] = []
    current = ""
    for ch in text:
        if ch.isdigit():
            current += ch
            continue
        if current:
            digits.append(current)
            current = ""
        if len(digits) >= 3:
            break
    if current and len(digits) < 3:
        digits.append(current)
    if not digits:
        return None
    while len(digits) < 3:
        digits.append("0")
    try:
        return (int(digits[0]), int(digits[1]), int(digits[2]))
    except ValueError:
        return None


def version_supported(
    parsed_version: tuple[int, int, int] | None,
    minimum: tuple[int, int, int] | None,
) -> bool:
    """Return True when parsed version meets minimum gate or no gate exists."""
    if minimum is None:
        return True
    if parsed_version is None:
        return False
    return parsed_version >= minimum
