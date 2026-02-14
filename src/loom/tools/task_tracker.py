"""Progress tracking tool for multi-step tasks.

Gives the model a structured way to track its own work, show progress
to the user, and manage complex multi-step tasks. State is kept in
memory for the session — no persistence needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from loom.tools.registry import Tool, ToolContext, ToolResult


@dataclass
class _Task:
    id: int
    content: str
    status: str = "pending"  # pending | in_progress | completed

    def to_dict(self) -> dict:
        return {"id": self.id, "content": self.content, "status": self.status}


class TaskTrackerTool(Tool):
    """Track progress on multi-step tasks.

    The model can create, update, and list tasks. This helps organize
    complex work and shows the user what's been done and what's next.
    """

    name = "task_tracker"
    description = (
        "Track progress on multi-step tasks. Actions: "
        "'add' (create a task), 'update' (change status to pending/in_progress/completed), "
        "'list' (show all tasks), 'clear' (remove all tasks). "
        "Use this to organize complex work and communicate progress."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "update", "list", "clear"],
                "description": "Action to perform.",
            },
            "content": {
                "type": "string",
                "description": "Task description (for 'add').",
            },
            "task_id": {
                "type": "integer",
                "description": "Task ID (for 'update').",
            },
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed"],
                "description": "New status (for 'update').",
            },
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        super().__init__()
        self._tasks: list[_Task] = []
        self._next_id = 1

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        action = args.get("action", "")

        if action == "add":
            return self._add(args.get("content", ""))
        elif action == "update":
            return self._update(args.get("task_id"), args.get("status", ""))
        elif action == "list":
            return self._list()
        elif action == "clear":
            return self._clear()
        else:
            return ToolResult.fail(f"Unknown action: {action}")

    def _add(self, content: str) -> ToolResult:
        if not content:
            return ToolResult.fail("Task content is required.")
        task = _Task(id=self._next_id, content=content)
        self._tasks.append(task)
        self._next_id += 1
        return ToolResult.ok(
            f"Task #{task.id} added: {content}",
            data=task.to_dict(),
        )

    def _update(self, task_id: int | None, status: str) -> ToolResult:
        if task_id is None:
            return ToolResult.fail("task_id is required.")
        if status not in ("pending", "in_progress", "completed"):
            return ToolResult.fail(f"Invalid status: {status}")

        for task in self._tasks:
            if task.id == task_id:
                task.status = status
                return ToolResult.ok(
                    f"Task #{task.id} -> {status}: {task.content}",
                    data=task.to_dict(),
                )
        return ToolResult.fail(f"Task #{task_id} not found.")

    def _list(self) -> ToolResult:
        if not self._tasks:
            return ToolResult.ok("No tasks tracked.")

        lines = []
        counts = {"pending": 0, "in_progress": 0, "completed": 0}
        for t in self._tasks:
            icon = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}[t.status]
            lines.append(f"  {icon} #{t.id} {t.content}")
            counts[t.status] += 1

        summary = (
            f"{counts['completed']}/{len(self._tasks)} done"
            f" | {counts['in_progress']} in progress"
            f" | {counts['pending']} pending"
        )
        lines.insert(0, summary)
        return ToolResult.ok(
            "\n".join(lines),
            data={"tasks": [t.to_dict() for t in self._tasks], "counts": counts},
        )

    def _clear(self) -> ToolResult:
        count = len(self._tasks)
        self._tasks.clear()
        self._next_id = 1
        return ToolResult.ok(f"Cleared {count} task(s).")
