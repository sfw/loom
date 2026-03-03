"""OpenAI Codex tool wrapper."""

from __future__ import annotations

from loom.tools.tooling_common.provider_agent_tool import ProviderAgentTool


class OpenAICodexTool(ProviderAgentTool):
    """Run OpenAI Codex CLI in non-interactive mode."""

    __loom_register__ = True
    name = "openai_codex"
    description = (
        "Run OpenAI Codex CLI in non-interactive mode with sandbox and approval "
        "controls plus normalized output."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Prompt text passed to Codex.",
            },
            "cwd": {
                "type": "string",
                "description": "Optional working directory (relative to workspace).",
            },
            "sandbox_mode": {
                "type": "string",
                "enum": ["read_only", "workspace_write", "unrestricted"],
                "description": "Sandbox mode for Codex.",
            },
            "approval_mode": {
                "type": "string",
                "enum": ["untrusted", "on_failure", "on_request", "never"],
                "description": "Approval mode for Codex.",
            },
            "skip_git_repo_check": {
                "type": "boolean",
                "description": (
                    "Skip Codex git/trust directory preflight check. "
                    "Defaults to true."
                ),
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Command timeout in seconds.",
            },
            "output_mode": {
                "type": "string",
                "enum": ["text", "json", "stream"],
                "description": "Output mode.",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional allowlisted passthrough CLI args.",
            },
            "provider": {
                "type": "string",
                "description": "Compatibility alias; must resolve to codex.",
            },
        },
        "required": ["prompt"],
    }
    _provider = "codex"
