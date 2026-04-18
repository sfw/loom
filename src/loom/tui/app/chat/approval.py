"""Tool approval modal callback flow."""

from __future__ import annotations

import asyncio

from loom.cowork.approval import ApprovalDecision
from loom.tui.screens import ToolApprovalScreen
from loom.tui.widgets import ChatLog
from loom.tui.widgets.tool_call import tool_args_preview


async def approval_callback(
    self,
    tool_name: str,
    args: dict,
) -> ApprovalDecision:
    """Show approval modal and wait for result."""
    preview = tool_args_preview(tool_name, args)
    risk_info = args.get("_loom_risk_info")
    if not isinstance(risk_info, dict):
        risk_info = None

    approval_event = asyncio.Event()
    self._approval_event = approval_event
    self._approval_result = ApprovalDecision.DENY
    prompt_id = f"approval:{id(approval_event)}"

    try:
        chat = self.query_one("#chat-log", ChatLog)
    except Exception:
        chat = None
    if chat is not None:
        chat.add_approval_prompt(
            prompt_id,
            tool_name,
            preview,
            risk_info=risk_info,
        )

    def handle_result(result: str) -> None:
        if result == "approve":
            self._approval_result = ApprovalDecision.APPROVE
        elif result == "approve_all":
            self._approval_result = ApprovalDecision.APPROVE_ALL
        else:
            self._approval_result = ApprovalDecision.DENY
        approval_event.set()

    self.push_screen(
        ToolApprovalScreen(tool_name, preview, risk_info=risk_info),
        callback=handle_result,
    )

    await approval_event.wait()
    if chat is not None:
        chat.clear_info_line(prompt_id)
    if self._approval_event is approval_event:
        self._approval_event = None
    return self._approval_result
