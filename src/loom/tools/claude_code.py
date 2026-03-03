"""Claude Code tool wrapper."""

from __future__ import annotations

from loom.tools.tooling_common.provider_agent_tool import ProviderAgentTool


class ClaudeCodeTool(ProviderAgentTool):
    """Run Claude Code in non-interactive mode."""

    __loom_register__ = True
    name = "claude_code"
    description = (
        "Run Claude Code in non-interactive mode with permission-mode mapping, "
        "bounded execution, and normalized output."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Prompt text passed to Claude Code.",
            },
            "cwd": {
                "type": "string",
                "description": "Optional working directory (relative to workspace).",
            },
            "sandbox_mode": {
                "type": "string",
                "enum": ["workspace_write", "unrestricted"],
                "description": "Permission sandbox profile for Claude Code.",
            },
            "approval_mode": {
                "type": "string",
                "enum": ["on_request", "never"],
                "description": "Approval mode mapped to Claude permission-mode.",
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
                "description": "Compatibility alias; must resolve to claude_code.",
            },
        },
        "required": ["prompt"],
    }
    _provider = "claude_code"
