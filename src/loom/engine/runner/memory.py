"""Memory extraction helpers for runner."""

from __future__ import annotations

import asyncio
import json
import logging

from loom.models.base import ModelResponse
from loom.models.request_diagnostics import (
    collect_request_diagnostics,
    collect_response_diagnostics,
)
from loom.models.retry import ModelRetryPolicy, call_with_model_retry
from loom.state.memory import MemoryEntry


def spawn_memory_extraction(
    runner,
    task_id: str,
    subtask_id: str,
    result,
    *,
    logger: logging.Logger,
) -> None:
    """Schedule memory extraction as a background task."""
    remaining_seconds = runner._remaining_subtask_seconds()
    extractor_guard = float(
        getattr(
            runner,
            "_extractor_timeout_guard_seconds",
            runner.EXTRACTOR_TIMEOUT_GUARD_SECONDS,
        ),
    )
    if remaining_seconds <= extractor_guard:
        logger.debug(
            "Memory extraction skipped for %s: timeout guard active "
            "(remaining=%.2fs, guard=%.2fs)",
            subtask_id,
            remaining_seconds,
            extractor_guard,
        )
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(
            runner._extract_memory(task_id, subtask_id, result)
        )
    except RuntimeError:
        logger.debug("Memory extraction skipped: no running event loop")


async def extract_memory(
    runner,
    task_id: str,
    subtask_id: str,
    result,
    *,
    logger: logging.Logger,
) -> None:
    """Extract structured memory entries from subtask execution."""
    try:
        model = runner._router.select(tier=1, role="extractor")
    except Exception as e:
        logger.debug("Memory extraction skipped (no extractor model): %s", e)
        return

    compacted_fields: set[str] = set()
    extractor_tool_args_max = int(
        getattr(
            runner,
            "_extractor_tool_args_max_chars",
            runner.EXTRACTOR_TOOL_ARGS_MAX_CHARS,
        ),
    )
    extractor_tool_trace_max = int(
        getattr(
            runner,
            "_extractor_tool_trace_max_chars",
            runner.EXTRACTOR_TOOL_TRACE_MAX_CHARS,
        ),
    )
    extractor_prompt_max = int(
        getattr(
            runner,
            "_extractor_prompt_max_chars",
            runner.EXTRACTOR_PROMPT_MAX_CHARS,
        ),
    )

    tool_lines = []
    for tc in result.tool_calls:
        status = "OK" if tc.result.success else f"FAILED: {tc.result.error}"
        raw_args_text = json.dumps(tc.args, ensure_ascii=False, default=str)
        compact_args = await runner._summarize_tool_call_arguments(
            tc.args,
            max_chars=extractor_tool_args_max,
            label=f"{tc.tool} extractor args",
        )
        compact_args_text = json.dumps(
            compact_args,
            ensure_ascii=False,
            default=str,
        )
        if compact_args_text != raw_args_text:
            compacted_fields.add("tool_args")
        line = f"- {tc.tool}({compact_args_text}) → {status}"
        if tc.result.content_blocks:
            block_types = [getattr(b, "type", "?") for b in tc.result.content_blocks]
            line += f" [content: {', '.join(block_types)}]"
        tool_lines.append(line)
    tool_calls_formatted = "\n".join(tool_lines) if tool_lines else "No tool calls."
    if len(tool_calls_formatted) > extractor_tool_trace_max:
        tool_calls_formatted = await runner._compact_text(
            tool_calls_formatted,
            max_chars=extractor_tool_trace_max,
            label="memory extractor tool trace",
        )
        compacted_fields.add("tool_trace")

    model_output = str(result.summary or "")
    prompt = runner._prompts.build_extractor_prompt(
        subtask_id=subtask_id,
        tool_calls_formatted=tool_calls_formatted,
        model_output=model_output,
    )
    if len(prompt) > extractor_prompt_max:
        output_budget = max(
            220,
            min(int(extractor_prompt_max * 0.35), len(model_output)),
        )
        if len(model_output) > output_budget:
            model_output = await runner._compact_text(
                model_output,
                max_chars=output_budget,
                label="memory extractor model output",
            )
            compacted_fields.add("model_output")
        prompt = runner._prompts.build_extractor_prompt(
            subtask_id=subtask_id,
            tool_calls_formatted=tool_calls_formatted,
            model_output=model_output,
        )
    if len(prompt) > extractor_prompt_max:
        tightened_trace_budget = max(
            220,
            min(
                int(extractor_prompt_max * 0.45),
                max(220, extractor_tool_trace_max // 2),
            ),
        )
        tool_calls_formatted = await runner._compact_text(
            tool_calls_formatted,
            max_chars=tightened_trace_budget,
            label="memory extractor tool trace strict",
        )
        compacted_fields.add("tool_trace")
        prompt = runner._prompts.build_extractor_prompt(
            subtask_id=subtask_id,
            tool_calls_formatted=tool_calls_formatted,
            model_output=model_output,
        )

    request_messages = [{"role": "user", "content": prompt}]
    extractor_prompt_est_tokens = runner._estimate_message_tokens(request_messages)

    try:
        request_diag = collect_request_diagnostics(
            messages=request_messages,
            origin="runner.extract_memory.complete",
        )
        runner._emit_model_event(
            task_id=task_id,
            subtask_id=subtask_id,
            model_name=model.name,
            phase="start",
            details={
                **request_diag.to_event_payload(),
                "extractor_prompt_chars": len(prompt),
                "extractor_prompt_est_tokens": extractor_prompt_est_tokens,
                "extractor_compacted_fields": sorted(compacted_fields),
            },
        )
        policy = ModelRetryPolicy.from_execution_config(runner._config.execution)
        response = await call_with_model_retry(
            lambda: model.complete(request_messages),
            policy=policy,
        )
        runner._emit_model_event(
            task_id=task_id,
            subtask_id=subtask_id,
            model_name=model.name,
            phase="done",
            details={
                "origin": request_diag.origin,
                "extractor_prompt_chars": len(prompt),
                "extractor_prompt_est_tokens": extractor_prompt_est_tokens,
                "extractor_compacted_fields": sorted(compacted_fields),
                **collect_response_diagnostics(response).to_event_payload(),
            },
        )
        entries = runner._parse_memory_entries(response, task_id, subtask_id)
        if entries:
            await runner._memory.store_many(entries)
    except Exception as e:
        logger.debug("Memory extraction failed for subtask %s: %s", subtask_id, e)


def parse_memory_entries(
    runner,
    response: ModelResponse,
    task_id: str,
    subtask_id: str,
) -> list[MemoryEntry]:
    """Parse extractor model response into MemoryEntry objects."""
    raw_entries: list[dict] = []
    text = (response.text or "").strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                raw_entries = parsed
        except (json.JSONDecodeError, ValueError):
            pass

    if not raw_entries:
        validation = runner._validator.validate_json_response(
            response, expected_keys=["entries"]
        )
        if validation.valid and validation.parsed is not None:
            raw_entries = validation.parsed.get("entries", [])

    entries = []
    for e in raw_entries:
        if not isinstance(e, dict):
            continue
        entry_type = e.get("type", "discovery")
        if entry_type not in (
            "decision", "error", "tool_result", "discovery", "artifact", "context"
        ):
            entry_type = "discovery"
        entries.append(MemoryEntry(
            task_id=task_id,
            subtask_id=subtask_id,
            entry_type=entry_type,
            summary=str(e.get("summary", "")),
            detail=str(e.get("detail", "")),
            tags=str(e.get("tags", "")),
        ))
    return entries
