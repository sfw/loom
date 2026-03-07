"""Slash command registry helpers."""

from __future__ import annotations

from collections.abc import Iterable

from ..models import SlashCommandSpec


def slash_spec_sort_key(spec: SlashCommandSpec, *, priority: dict[str, int]) -> tuple[int, str]:
    """Return deterministic display/completion ordering key for slash specs."""
    canonical = spec.canonical.lower()
    return priority.get(canonical, 999), canonical


def ordered_slash_specs(
    specs: Iterable[SlashCommandSpec],
    *,
    priority: dict[str, int],
) -> list[SlashCommandSpec]:
    """Return built-in slash specs in deterministic UX priority order."""
    return sorted(specs, key=lambda spec: slash_spec_sort_key(spec, priority=priority))


def slash_match_keys(spec: SlashCommandSpec) -> tuple[str, ...]:
    """Return normalized command tokens used for prefix matching."""
    return (spec.canonical.lower(), *(alias.lower() for alias in spec.aliases))


def matching_slash_commands(
    raw_input: str,
    *,
    ordered_specs: Iterable[SlashCommandSpec],
    process_command_map: dict[str, str],
) -> tuple[str, list[tuple[str, str]]]:
    """Return current slash token and matching commands."""
    text = raw_input.strip()
    if not text.startswith("/"):
        return "", []
    token = text.split()[0].lower()

    exact_matches: list[tuple[str, str]] = []
    prefix_matches: list[tuple[str, str]] = []
    fallback_matches: list[tuple[str, str]] = []
    for spec in ordered_specs:
        keys = slash_match_keys(spec)
        label = spec.canonical
        if spec.usage:
            label = f"{label} {spec.usage}"
        desc = spec.description
        if spec.aliases:
            desc = f"{desc} ({', '.join(spec.aliases)})"
        entry = (label, desc)
        if any(key == token for key in keys):
            exact_matches.append(entry)
        elif any(key.startswith(token) for key in keys):
            prefix_matches.append(entry)
        elif any(token in key for key in keys):
            fallback_matches.append(entry)

    for dynamic_token, process_name in sorted(process_command_map.items()):
        entry = (
            f"{dynamic_token} <goal>",
            f"run goal via process '{process_name}'",
        )
        if dynamic_token == token:
            exact_matches.append(entry)
        elif dynamic_token.startswith(token):
            prefix_matches.append(entry)
        elif token in dynamic_token:
            fallback_matches.append(entry)

    if exact_matches or prefix_matches:
        return token, [*exact_matches, *prefix_matches]
    return token, fallback_matches


def slash_completion_candidates(
    token: str,
    *,
    ordered_specs: Iterable[SlashCommandSpec],
    process_command_map: dict[str, str],
) -> list[str]:
    """Return slash command completions for a token prefix."""
    if token == "/":
        builtins = [spec.canonical for spec in ordered_specs]
        dynamic = sorted(process_command_map)
        return builtins + dynamic

    candidates: list[str] = []
    seen: set[str] = set()
    for spec in ordered_specs:
        for key in (spec.canonical, *spec.aliases):
            if key.startswith(token) and key not in seen:
                candidates.append(key)
                seen.add(key)
    for key in sorted(process_command_map):
        if key.startswith(token) and key not in seen:
            candidates.append(key)
            seen.add(key)
    return candidates
