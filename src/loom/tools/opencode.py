"""OpenCode tool wrapper."""

from __future__ import annotations

from loom.tools.tooling_common.provider_agent_tool import ProviderAgentTool


class OpenCodeTool(ProviderAgentTool):
    """Run OpenCode in non-interactive mode."""

    __loom_register__ = True
    name = "opencode"
    description = (
        "Run OpenCode in non-interactive mode with timeout and normalized output."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Prompt text passed to OpenCode.",
            },
            "cwd": {
                "type": "string",
                "description": "Optional working directory (relative to workspace).",
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
                "description": "Compatibility alias; must resolve to opencode.",
            },
        },
        "required": ["prompt"],
    }
    _provider = "opencode"
