"""Constraint library for local model prompts.

Local models need explicit negative constraints that RLHF'd models
handle implicitly. These are applied across all roles.
"""

COMMON_CONSTRAINTS = """\
- Do NOT hallucinate file contents. Always read files before referencing them.
- Do NOT assume the state of the workspace. Inspect before acting.
- Do NOT produce markdown explanations when tool calls are expected.
- If you're unsure, use a tool to check rather than guessing."""

EXECUTOR_CONSTRAINTS = """\
- Complete ONLY the current subtask. Do NOT move to the next one.
- Do NOT explain reasoning unless explicitly asked.
- If data is missing or unclear, report the gap. Do NOT substitute or fabricate.
- Keep edits minimal and targeted. Do not rewrite files unnecessarily."""

SAFETY_CONSTRAINTS = """\
- Do NOT modify files outside the workspace directory.
- Do NOT execute commands that delete data without explicit instruction.
- Do NOT install system-level packages.
- Do NOT access the network unless the task explicitly requires it."""

PLANNER_CONSTRAINTS = """\
- Create 3-30 subtasks. Let task complexity determine the count.
- The first subtask should always read/inspect the current state.
- Do NOT create subtasks that depend on information you don't have.
- Use depends_on to expose independent subtasks that can run in parallel.
- Each subtask ID must be unique and descriptive."""

VERIFIER_CONSTRAINTS = """\
- Be skeptical. Do not assume success without evidence.
- Check that tool calls actually accomplished the described goal.
- A partially completed subtask is a failure.
- Respond with ONLY the JSON object."""


def get_constraints_for_role(role: str) -> str:
    """Get the combined constraint text for a given role."""
    parts = [COMMON_CONSTRAINTS, SAFETY_CONSTRAINTS]

    role_constraints = {
        "executor": EXECUTOR_CONSTRAINTS,
        "planner": PLANNER_CONSTRAINTS,
        "verifier": VERIFIER_CONSTRAINTS,
    }

    if role in role_constraints:
        parts.append(role_constraints[role])

    return "\n".join(parts)
