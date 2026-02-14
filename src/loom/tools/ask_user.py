"""AskUser tool: lets the model ask the developer for clarification.

When the model needs input (a decision, clarification, or preference),
it calls this tool.  The cowork CLI will display the question and wait
for the user's response, which becomes the tool result.
"""

from __future__ import annotations

from loom.tools.registry import Tool, ToolContext, ToolResult


class AskUserTool(Tool):
    """Ask the user a question and return their response."""

    name = "ask_user"
    description = (
        "Ask the developer a question when you need clarification, a decision, "
        "or additional information. The question will be displayed and the "
        "developer's response returned. Use this instead of guessing."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the developer.",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of suggested options for the developer to choose from.",
            },
        },
        "required": ["question"],
    }

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Execute is handled specially by the cowork session.

        The cowork CLI intercepts this tool call, displays the question,
        waits for user input, and returns it as the result.  If we get
        here (e.g. in non-interactive mode), return a placeholder.
        """
        question = args.get("question", "")
        options = args.get("options", [])

        # In non-interactive mode, return the question as a prompt
        # The caller is expected to intercept this tool call
        formatted = f"QUESTION: {question}"
        if options:
            formatted += "\nOptions: " + ", ".join(options)

        return ToolResult(
            success=True,
            output=formatted,
            data={"question": question, "options": options, "awaiting_input": True},
        )
