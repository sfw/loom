"""Shared compaction control helpers for runner and cowork context assembly."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from loom.models.request_diagnostics import RequestDiagnostics, collect_request_diagnostics


@dataclass(frozen=True)
class BudgetedSelection:
    selected_indices: tuple[int, ...]
    protected_indices: tuple[int, ...]
    preserved_tool_exchange: tuple[int, ...]


@dataclass(frozen=True)
class RequestBudget:
    diagnostics: RequestDiagnostics
    context_budget_tokens: int
    target_budget_tokens: int
    request_est_tokens: int
    usage_ratio: float
    deficit_tokens: int


def preserved_tool_exchange_indices(messages: list[dict]) -> tuple[int, ...]:
    newest_assistant_tool = -1
    newest_tool_result = -1
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        if role == "assistant" and msg.get("tool_calls"):
            newest_assistant_tool = idx
        elif role == "tool":
            newest_tool_result = idx
    return tuple(
        sorted(idx for idx in {newest_assistant_tool, newest_tool_result} if idx >= 0),
    )


def protected_tail_indices(
    messages: list[dict],
    *,
    preserve_recent: int,
    todo_prefix: str = "",
) -> tuple[int, ...]:
    preserve_recent = max(2, int(preserve_recent))
    protected: set[int] = set(preserved_tool_exchange_indices(messages))
    if messages:
        protected.add(0)
    narrative_indices: list[int] = []
    for idx, msg in enumerate(messages):
        if idx == 0 or not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        content = msg.get("content")
        if (
            todo_prefix
            and role == "user"
            and isinstance(content, str)
            and content.startswith(todo_prefix)
        ):
            protected.add(idx)
            continue
        if role == "assistant" and msg.get("tool_calls"):
            continue
        if role in {"assistant", "user", "system"}:
            narrative_indices.append(idx)
    for idx in narrative_indices[-preserve_recent:]:
        protected.add(idx)
    return tuple(sorted(protected))


def select_budgeted_messages(
    messages: list[dict],
    *,
    token_budget: int,
    recent_message_cap: int,
    preserve_recent: int,
    token_estimator: Callable[[dict], int],
    todo_prefix: str = "",
) -> BudgetedSelection:
    if not messages:
        return BudgetedSelection(
            selected_indices=(),
            protected_indices=(),
            preserved_tool_exchange=(),
        )

    budget = max(1, int(token_budget))
    recent_cap = max(1, int(recent_message_cap))
    protected = tuple(
        idx
        for idx in protected_tail_indices(
            messages,
            preserve_recent=preserve_recent,
            todo_prefix=todo_prefix,
        )
        if idx < len(messages)
    )
    protected_set = set(protected)
    preserved_exchange = tuple(
        idx for idx in preserved_tool_exchange_indices(messages) if idx < len(messages)
    )
    selected_desc: list[int] = []
    used = 0

    for idx in range(len(messages) - 1, -1, -1):
        if len(selected_desc) >= recent_cap:
            break
        msg = messages[idx]
        msg_tokens = max(0, int(token_estimator(msg)))
        if msg_tokens > budget:
            continue
        if used + msg_tokens > budget:
            break
        selected_desc.append(idx)
        used += msg_tokens

    selected = sorted(selected_desc)
    selected_set = set(selected)

    for idx in protected:
        if idx in selected_set:
            continue
        msg_tokens = max(0, int(token_estimator(messages[idx])))
        if msg_tokens > budget:
            continue
        while selected:
            if len(selected) < recent_cap and used + msg_tokens <= budget:
                break
            eviction_idx = next(
                (candidate for candidate in selected if candidate not in protected_set),
                None,
            )
            if eviction_idx is None:
                break
            selected.remove(eviction_idx)
            selected_set.discard(eviction_idx)
            used = max(0, used - max(0, int(token_estimator(messages[eviction_idx]))))
        if len(selected) >= recent_cap or used + msg_tokens > budget:
            continue
        selected.append(idx)
        selected.sort()
        selected_set.add(idx)
        used += msg_tokens

    while selected:
        first = selected[0]
        first_msg = messages[first]
        if not isinstance(first_msg, dict):
            selected.pop(0)
            selected_set.discard(first)
            continue
        if str(first_msg.get("role", "")).strip().lower() != "tool":
            break
        selected.pop(0)
        selected_set.discard(first)

    return BudgetedSelection(
        selected_indices=tuple(selected),
        protected_indices=protected,
        preserved_tool_exchange=preserved_exchange,
    )


def compute_request_budget(
    *,
    messages: list[dict],
    context_budget_tokens: int,
    tools: list[dict] | None = None,
    target_ratio: float = 1.0,
    origin: str = "",
) -> RequestBudget:
    safe_budget = max(1, int(context_budget_tokens))
    safe_ratio = max(0.0, float(target_ratio or 0.0))
    target_budget = max(1, min(safe_budget, int(round(safe_budget * safe_ratio))))
    diagnostics = collect_request_diagnostics(
        messages=messages,
        tools=tools or [],
        origin=origin or "compaction_control.request_budget",
    )
    request_est_tokens = int(diagnostics.request_est_tokens)
    usage_ratio = request_est_tokens / max(1, safe_budget)
    deficit_tokens = max(0, request_est_tokens - target_budget)
    return RequestBudget(
        diagnostics=diagnostics,
        context_budget_tokens=safe_budget,
        target_budget_tokens=target_budget,
        request_est_tokens=request_est_tokens,
        usage_ratio=usage_ratio,
        deficit_tokens=deficit_tokens,
    )
