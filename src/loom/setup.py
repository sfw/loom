"""Interactive first-run setup wizard for Loom.

Walks users through configuring their first model provider and writes
~/.loom/loom.toml.  Re-enterable via ``loom setup`` at any time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import httpx

# Provider presets: (display_name, provider_key, needs_api_key, default_base_url)
PROVIDERS = [
    ("Anthropic (Claude API)", "anthropic", True, "https://api.anthropic.com"),
    ("OpenAI-compatible server", "openai_compatible", False, "http://localhost:1234/v1"),
    ("Ollama", "ollama", False, "http://localhost:11434"),
]

ROLE_PRESETS = {
    "all": ["planner", "executor", "extractor", "verifier", "compactor"],
    "primary": ["planner", "executor"],
    "utility": ["extractor", "verifier", "compactor"],
}

CONFIG_DIR = Path.home() / ".loom"
CONFIG_PATH = CONFIG_DIR / "loom.toml"


def _unique_models(names: list[str]) -> list[str]:
    """Return non-empty model names with order-preserving dedupe."""
    result = []
    seen: set[str] = set()
    for name in names:
        if not isinstance(name, str):
            continue
        normalized = name.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _extract_discovered_models(provider_key: str, payload: object) -> list[str]:
    """Parse provider-specific model-list payloads into model names."""
    if provider_key == "ollama" and isinstance(payload, dict):
        models = payload.get("models", [])
        if isinstance(models, list):
            names = []
            for entry in models:
                if isinstance(entry, dict):
                    names.append(
                        entry.get("name", "") or entry.get("model", "")
                    )
            return _unique_models(names)
        return []

    rows: list[object] = []
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            rows = data
        elif isinstance(payload.get("models"), list):
            rows = payload["models"]
    elif isinstance(payload, list):
        rows = payload

    names: list[str] = []
    for entry in rows:
        if isinstance(entry, dict):
            names.append(
                entry.get("id", "")
                or entry.get("name", "")
                or entry.get("model", "")
            )
        elif isinstance(entry, str):
            names.append(entry)
    return _unique_models(names)


def discover_models(
    provider_key: str,
    base_url: str,
    api_key: str = "",
    *,
    timeout: float = 5.0,
) -> list[str]:
    """Discover available models from a provider endpoint.

    Returns an empty list on connectivity/auth/parsing errors.
    """
    normalized_base = base_url.strip().rstrip("/")
    if not normalized_base:
        return []

    headers: dict[str, str] = {}
    endpoints: list[str]
    if provider_key == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
        if api_key:
            headers["x-api-key"] = api_key
        endpoints = ["/v1/models"]
    elif provider_key == "openai_compatible":
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        endpoints = ["/models", "/v1/models"]
    elif provider_key == "ollama":
        endpoints = ["/api/tags"]
    else:
        return []

    with httpx.Client(
        base_url=normalized_base,
        headers=headers,
        timeout=timeout,
        follow_redirects=True,
    ) as client:
        for endpoint in endpoints:
            try:
                response = client.get(endpoint)
            except httpx.HTTPError:
                continue
            if response.status_code != 200:
                continue
            try:
                payload = response.json()
            except ValueError:
                continue
            names = _extract_discovered_models(provider_key, payload)
            if names:
                return names

    return []


def needs_setup() -> bool:
    """Return True if no user config exists yet."""
    if CONFIG_PATH.exists():
        return False
    # Also check cwd for a project-local config
    if (Path.cwd() / "loom.toml").exists():
        return False
    return True


def _prompt_provider() -> tuple[str, str, bool, str]:
    """Ask which provider to use.  Returns (display, key, needs_key, default_url)."""
    click.echo()
    click.echo("Which model provider will you use?")
    click.echo()
    for i, (display, _key, _needs, _url) in enumerate(PROVIDERS, 1):
        click.echo(f"  {i}. {display}")
    click.echo()

    choice = click.prompt(
        "Provider",
        type=click.IntRange(1, len(PROVIDERS)),
        default=1,
    )
    return PROVIDERS[choice - 1]


def _prompt_model_name(
    provider_display: str,
    discovered_models: list[str],
    fallback_prompt: str,
) -> str:
    """Select from discovered models or collect a manual model name."""
    if discovered_models:
        click.echo()
        click.echo(f"Discovered models ({provider_display}):")
        click.echo()
        for i, model in enumerate(discovered_models, 1):
            click.echo(f"  {i}. {model}")
        click.echo()
        choice = click.prompt(
            "Model",
            type=click.IntRange(1, len(discovered_models)),
            default=1,
        )
        return discovered_models[choice - 1]

    click.echo()
    click.echo("Could not auto-discover models from that endpoint.")
    return click.prompt(fallback_prompt).strip()


def _prompt_anthropic_model(default_url: str) -> tuple[str, str, str]:
    """Collect Anthropic-specific settings.  Returns (base_url, model, api_key)."""
    api_key = click.prompt(
        "Anthropic API key",
        hide_input=True,
    ).strip()
    if not api_key:
        click.echo("API key is required for Anthropic.", err=True)
        sys.exit(1)

    base_url = click.prompt(
        "Base URL (press Enter for default)",
        default=default_url,
        show_default=True,
    )

    discovered_models = discover_models("anthropic", base_url, api_key)
    model = _prompt_model_name(
        "Anthropic",
        discovered_models,
        "Model name (e.g. claude-sonnet-4-5-20250929)",
    )
    if not model:
        click.echo("Model name is required.", err=True)
        sys.exit(1)
    return base_url, model, api_key


def _prompt_openai_model(default_url: str) -> tuple[str, str, str]:
    """Collect OpenAI-compatible settings.  Returns (base_url, model, api_key)."""
    click.echo()
    base_url = click.prompt(
        "Server URL",
        default=default_url,
        show_default=True,
    )
    api_key = ""
    if click.confirm("Does this server require an API key?", default=False):
        api_key = click.prompt("API key", hide_input=True).strip()
    discovered_models = discover_models("openai_compatible", base_url, api_key)
    model = _prompt_model_name(
        "OpenAI-compatible",
        discovered_models,
        "Model name (e.g. gpt-4o, mistral-nemo, etc.)",
    )
    if not model:
        click.echo("Model name is required.", err=True)
        sys.exit(1)
    return base_url, model, api_key


def _prompt_ollama_model(default_url: str) -> tuple[str, str, str]:
    """Collect Ollama settings.  Returns (base_url, model, api_key)."""
    click.echo()
    base_url = click.prompt(
        "Ollama URL",
        default=default_url,
        show_default=True,
    )
    discovered_models = discover_models("ollama", base_url, "")
    model = _prompt_model_name(
        "Ollama",
        discovered_models,
        "Model name (e.g. qwen3:14b, llama3:8b, etc.)",
    )
    if not model:
        click.echo("Model name is required.", err=True)
        sys.exit(1)
    return base_url, model, ""


def _prompt_roles() -> list[str]:
    """Ask which roles this model should fill."""
    click.echo()
    click.echo("Which roles should this model handle?")
    click.echo()
    click.echo("  1. All roles (planner, executor, extractor, verifier, compactor)")
    click.echo("  2. Primary (planner, executor)")
    click.echo("  3. Utility (extractor, verifier, compactor)")
    click.echo()
    choice = click.prompt(
        "Roles", type=click.IntRange(1, 3), default=1,
    )
    presets = ["all", "primary", "utility"]
    return ROLE_PRESETS[presets[choice - 1]]


def _generate_toml(models: list[dict]) -> str:
    """Generate a loom.toml string from collected model configs."""
    lines = [
        "# Loom configuration",
        "# Generated by `loom setup` — edit freely.",
        "",
        "[server]",
        'host = "127.0.0.1"',
        "port = 9000",
        "",
    ]

    for m in models:
        lines.append(f"[models.{m['name']}]")
        lines.append(f'provider = "{m["provider"]}"')
        if m.get("base_url"):
            lines.append(f'base_url = "{m["base_url"]}"')
        lines.append(f'model = "{m["model"]}"')
        if m.get("api_key"):
            lines.append(f'api_key = "{m["api_key"]}"')
        lines.append(f"max_tokens = {m.get('max_tokens', 8192)}")
        lines.append(f"temperature = {m.get('temperature', 0.1)}")
        roles_str = ", ".join(f'"{r}"' for r in m["roles"])
        lines.append(f"roles = [{roles_str}]")
        lines.append("")

    lines.extend([
        "[workspace]",
        'default_path = "~/projects"',
        'scratch_dir = "~/.loom/scratch"',
        "",
        "[execution]",
        "max_subtask_retries = 3",
        "max_loop_iterations = 50",
        "delegate_task_timeout_seconds = 3600",
        "auto_approve_confidence_threshold = 0.8",
        "",
        "[verification]",
        "tier1_enabled = true",
        "tier2_enabled = true",
        "tier3_enabled = false",
        "tier3_vote_count = 3",
        "",
        "[limits]",
        "planning_response_max_tokens = 16384",
        "adhoc_repair_source_max_chars = 0",
        "evidence_context_text_max_chars = 8192",
        "",
        "[limits.runner]",
        "max_tool_iterations = 20",
        "max_subtask_wall_clock_seconds = 1200",
        "max_model_context_tokens = 24000",
        "max_state_summary_chars = 640",
        "max_verification_summary_chars = 8000",
        "default_tool_result_output_chars = 2800",
        "heavy_tool_result_output_chars = 3600",
        "compact_tool_result_output_chars = 900",
        "compact_text_output_chars = 1400",
        "minimal_text_output_chars = 260",
        "tool_call_argument_context_chars = 700",
        "compact_tool_call_argument_chars = 1600",
        "enable_filetype_ingest_router = true",
        "enable_artifact_telemetry_events = true",
        "artifact_telemetry_max_metadata_chars = 1200",
        "enable_model_overflow_fallback = true",
        "ingest_artifact_retention_max_age_days = 14",
        "ingest_artifact_retention_max_files_per_scope = 96",
        "ingest_artifact_retention_max_bytes_per_scope = 268435456",
        "",
        "[limits.verifier]",
        "max_tool_args_chars = 360",
        "max_tool_status_chars = 320",
        "max_tool_calls_tokens = 4000",
        "max_verifier_prompt_tokens = 12000",
        "max_result_summary_chars = 7000",
        "compact_result_summary_chars = 2600",
        "max_evidence_section_chars = 4200",
        "max_evidence_section_compact_chars = 2200",
        "max_artifact_section_chars = 4200",
        "max_artifact_section_compact_chars = 2200",
        "max_tool_output_excerpt_chars = 1100",
        "max_artifact_file_excerpt_chars = 800",
        "",
        "[limits.compactor]",
        "max_chunk_chars = 8000",
        "max_chunks_per_round = 10",
        "max_reduction_rounds = 2",
        "min_compact_target_chars = 220",
        "response_tokens_floor = 256",
        "response_tokens_ratio = 0.55",
        "response_tokens_buffer = 256",
        "json_headroom_chars_floor = 128",
        "json_headroom_chars_ratio = 0.30",
        "json_headroom_chars_cap = 1024",
        "chars_per_token_estimate = 2.8",
        "token_headroom = 128",
        "target_chars_ratio = 0.82",
        "",
        "[memory]",
        'database_path = "~/.loom/loom.db"',
        "",
        "[logging]",
        'level = "INFO"',
        'event_log_path = "~/.loom/logs"',
        "",
    ])
    return "\n".join(lines)


def run_setup(*, reconfigure: bool = False) -> Path:
    """Run the interactive setup wizard.

    Args:
        reconfigure: If True, allow overwriting an existing config.

    Returns:
        Path to the written config file.
    """
    click.echo()
    click.secho("  Loom Setup", bold=True)
    click.echo("  " + "─" * 40)
    click.echo()

    if CONFIG_PATH.exists() and not reconfigure:
        click.echo(f"  Config already exists: {CONFIG_PATH}")
        if not click.confirm("  Overwrite with new configuration?", default=False):
            click.echo("  Setup cancelled.")
            sys.exit(0)

    # -- Collect primary model -------------------------------------------------
    click.secho("  Primary model", bold=True)
    click.echo("  This model handles planning and task execution.")

    _display, provider_key, _needs_key, default_url = _prompt_provider()

    if provider_key == "anthropic":
        base_url, model, api_key = _prompt_anthropic_model(default_url)
    elif provider_key == "openai_compatible":
        base_url, model, api_key = _prompt_openai_model(default_url)
    else:
        base_url, model, api_key = _prompt_ollama_model(default_url)

    roles = _prompt_roles()

    models = [{
        "name": "primary",
        "provider": provider_key,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "roles": roles,
        "max_tokens": 8192,
        "temperature": 0.1,
    }]

    # -- Optional utility model ------------------------------------------------
    # Only offer if primary doesn't cover all roles
    missing = set(ROLE_PRESETS["all"]) - set(roles)
    if missing:
        click.echo()
        missing_str = ", ".join(sorted(missing))
        click.echo(f"  Uncovered roles: {missing_str}")
        if click.confirm("  Add a second model for those roles?", default=False):
            click.echo()
            click.secho("  Utility model", bold=True)

            _, u_provider, _, u_default_url = _prompt_provider()

            if u_provider == "anthropic":
                u_base, u_model, u_key = _prompt_anthropic_model(u_default_url)
            elif u_provider == "openai_compatible":
                u_base, u_model, u_key = _prompt_openai_model(u_default_url)
            else:
                u_base, u_model, u_key = _prompt_ollama_model(u_default_url)

            models.append({
                "name": "utility",
                "provider": u_provider,
                "base_url": u_base,
                "model": u_model,
                "api_key": u_key,
                "roles": sorted(missing),
                "max_tokens": 2048,
                "temperature": 0.0,
            })

    # -- Write config ----------------------------------------------------------
    toml_content = _generate_toml(models)

    click.echo()
    click.echo("  Configuration preview:")
    click.echo("  " + "─" * 40)
    for line in toml_content.splitlines():
        click.echo(f"  {line}")
    click.echo("  " + "─" * 40)
    click.echo()

    if not click.confirm(f"  Write to {CONFIG_PATH}?", default=True):
        click.echo("  Setup cancelled.")
        sys.exit(0)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(toml_content)

    # Ensure supporting directories exist
    (CONFIG_DIR / "scratch").mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "processes").mkdir(parents=True, exist_ok=True)

    click.echo()
    click.secho("  Setup complete!", bold=True)
    click.echo(f"  Config written to {CONFIG_PATH}")
    click.echo("  Run `loom` to start, or `loom setup` to reconfigure.")
    click.echo()

    return CONFIG_PATH
