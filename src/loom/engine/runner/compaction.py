"""Compaction helpers for runner overflow and payload rewriting."""

from __future__ import annotations

import json
import logging
from typing import Any

from loom.utils.tokens import estimate_tokens

from .types import CompactionClass, CompactionPressureTier, _CompactionPlan


def is_model_request_overflow_error(error: BaseException | str) -> bool:
    text = str(error or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in (
        "total message size",
        "exceeds limit",
        "exceeded model token limit",
        "maximum context length",
        "context length exceeded",
        "context_length_exceeded",
        "too many tokens",
    ))


def tool_call_name_index(messages: list[dict]) -> dict[str, str]:
    index: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for item in calls:
            if not isinstance(item, dict):
                continue
            call_id = str(item.get("id", "")).strip()
            fn = item.get("function")
            name = ""
            if isinstance(fn, dict):
                name = str(fn.get("name", "")).strip()
            if call_id and name:
                index[call_id] = name
    return index


def overflow_excerpt(value: str, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 32:
        return text[:max_chars]
    head = max_chars - 20
    return f"{text[:head].rstrip()} ...[excerpt]"


def rewrite_tool_payload_for_overflow(
    runner: Any,
    *,
    content: str,
    tool_name: str,
) -> tuple[str | None, int]:
    content_text = str(content or "")
    if not content_text:
        return None, 0

    min_chars = int(
        getattr(
            runner,
            "_overflow_fallback_tool_message_min_chars",
            runner.OVERFLOW_FALLBACK_TOOL_MESSAGE_MIN_CHARS,
        ),
    )
    excerpt_chars = int(
        getattr(
            runner,
            "_overflow_fallback_tool_output_excerpt_chars",
            runner.OVERFLOW_FALLBACK_TOOL_OUTPUT_EXCERPT_CHARS,
        ),
    )
    if len(content_text) < min_chars:
        return None, 0

    try:
        parsed = json.loads(content_text)
    except (json.JSONDecodeError, TypeError):
        excerpt = overflow_excerpt(content_text, max_chars=excerpt_chars)
        payload = {
            "success": True,
            "output": (
                f"{excerpt}\n\n"
                "[overflow fallback applied: condensed oversized non-JSON tool payload]"
            ),
            "error": None,
            "files_changed": [],
            "data": {
                "overflow_fallback": True,
                "tool_name": tool_name,
                "original_chars": len(content_text),
            },
        }
        compacted = json.dumps(payload, ensure_ascii=False)
        return compacted, len(content_text) - len(compacted)

    if not isinstance(parsed, dict):
        return None, 0

    raw_output = str(parsed.get("output", ""))
    data = parsed.get("data")
    data_dict = data if isinstance(data, dict) else {}
    kind = str(data_dict.get("content_kind", "")).strip().lower()
    artifact_ref = str(data_dict.get("artifact_ref", "")).strip()
    should_rewrite = (
        kind in runner._OVERFLOW_BINARY_CONTENT_KINDS
        or bool(artifact_ref)
        or tool_name in runner._HEAVY_OUTPUT_TOOLS
        or len(raw_output) >= excerpt_chars * 2
        or len(content_text) >= min_chars * 2
    )
    if not should_rewrite:
        return None, 0

    output_excerpt = overflow_excerpt(raw_output, max_chars=excerpt_chars)
    if output_excerpt:
        summary_output = (
            f"{output_excerpt}\n\n"
            "[overflow fallback applied: condensed oversized tool payload]"
        )
    else:
        summary_output = (
            "[overflow fallback applied: tool payload condensed to reduce "
            "request size]"
        )

    compact_payload: dict[str, Any] = {
        "success": bool(parsed.get("success", False)),
        "output": summary_output,
        "error": parsed.get("error"),
        "files_changed": list(parsed.get("files_changed", []))[:5],
    }
    files_changed = parsed.get("files_changed")
    if isinstance(files_changed, list) and len(files_changed) > 5:
        compact_payload["files_changed_count"] = len(files_changed)

    if data_dict:
        keep_keys = (
            "artifact_ref",
            "artifact_path",
            "artifact_workspace_relpath",
            "content_kind",
            "media_type",
            "size_bytes",
            "declared_size_bytes",
            "status_code",
            "source_url",
            "url",
            "truncated",
            "extract_text",
            "handler",
            "extracted_chars",
            "extraction_truncated",
        )
        compact_data = {
            key: data_dict[key]
            for key in keep_keys
            if key in data_dict
        }
        compact_data["overflow_fallback"] = True
        compact_data["tool_name"] = tool_name
        compact_data["original_chars"] = len(content_text)
        compact_payload["data"] = compact_data
    else:
        compact_payload["data"] = {
            "overflow_fallback": True,
            "tool_name": tool_name,
            "original_chars": len(content_text),
        }

    compacted = json.dumps(compact_payload, ensure_ascii=False)
    delta = len(content_text) - len(compacted)
    if delta <= 0:
        return None, 0
    return compacted, delta


def apply_model_overflow_fallback(
    runner: Any,
    messages: list[dict],
) -> tuple[list[dict], dict[str, Any]]:
    if not messages:
        return messages, {
            "overflow_fallback_applied": False,
            "overflow_fallback_rewritten_messages": 0,
            "overflow_fallback_chars_reduced": 0,
            "overflow_fallback_preserved_recent_messages": 0,
            "overflow_fallback_skipped_reason": "empty_history",
        }

    latest_tool_idx = max(
        (
            idx
            for idx, msg in enumerate(messages)
            if isinstance(msg, dict)
            and str(msg.get("role", "")).strip().lower() == "tool"
        ),
        default=-1,
    )
    call_name_index = tool_call_name_index(messages)

    rewritten_messages = list(messages)
    rewritten_count = 0
    chars_reduced = 0
    candidate_count = 0
    preserved_recent_messages = 1 if latest_tool_idx >= 0 else 0

    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip().lower()
        if role != "tool":
            continue
        if idx == latest_tool_idx:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        candidate_count += 1
        call_id = str(message.get("tool_call_id", "")).strip()
        tool_name = call_name_index.get(call_id, "")
        rewritten_content, delta = rewrite_tool_payload_for_overflow(
            runner,
            content=content,
            tool_name=tool_name,
        )
        if not rewritten_content or delta <= 0:
            continue
        new_message = dict(message)
        new_message["content"] = rewritten_content
        rewritten_messages[idx] = new_message
        rewritten_count += 1
        chars_reduced += delta

    if rewritten_count <= 0:
        return messages, {
            "overflow_fallback_applied": False,
            "overflow_fallback_rewritten_messages": 0,
            "overflow_fallback_chars_reduced": 0,
            "overflow_fallback_candidate_messages": candidate_count,
            "overflow_fallback_preserved_recent_messages": preserved_recent_messages,
            "overflow_fallback_skipped_reason": "no_eligible_tool_payloads",
        }

    return rewritten_messages, {
        "overflow_fallback_applied": True,
        "overflow_fallback_rewritten_messages": rewritten_count,
        "overflow_fallback_chars_reduced": chars_reduced,
        "overflow_fallback_candidate_messages": candidate_count,
        "overflow_fallback_preserved_recent_messages": preserved_recent_messages,
        "overflow_fallback_skipped_reason": "",
    }


def set_compaction_diagnostics(runner: Any, payload: dict[str, Any]) -> None:
    runner._last_compaction_diagnostics = payload


def estimate_message_tokens(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        try:
            total += estimate_tokens(json.dumps(message))
        except (TypeError, ValueError):
            total += estimate_tokens(str(message))
    return total


def compute_compaction_pressure_tier(
    runner: Any,
    usage_ratio: float,
) -> CompactionPressureTier:
    soft = float(
        getattr(
            runner,
            "_compaction_pressure_ratio_soft",
            runner.COMPACTION_PRESSURE_RATIO_SOFT,
        ),
    )
    hard = float(
        getattr(
            runner,
            "_compaction_pressure_ratio_hard",
            runner.COMPACTION_PRESSURE_RATIO_HARD,
        ),
    )
    hard = max(soft + 0.01, hard)
    if usage_ratio <= soft:
        return CompactionPressureTier.NORMAL
    if usage_ratio <= hard:
        return CompactionPressureTier.PRESSURE
    return CompactionPressureTier.CRITICAL


def critical_message_indices(runner: Any, messages: list[dict]) -> tuple[int, ...]:
    preserve_recent = int(
        getattr(
            runner,
            "_preserve_recent_critical_messages",
            runner.PRESERVE_RECENT_CRITICAL_MESSAGES,
        ),
    )
    preserve_recent = max(2, preserve_recent)
    critical: set[int] = {0}
    narrative_indices: list[int] = []
    for idx, msg in enumerate(messages):
        if idx == 0 or not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        content = msg.get("content")
        if (
            role == "user"
            and isinstance(content, str)
            and content.startswith(runner._TODO_REMINDER_PREFIX)
        ):
            critical.add(idx)
            continue
        if role == "assistant" and msg.get("tool_calls"):
            continue
        if role in {"assistant", "user", "system"}:
            narrative_indices.append(idx)
    for idx in narrative_indices[-preserve_recent:]:
        critical.add(idx)
    return tuple(sorted(critical))


def classify_message_for_compaction(
    message: dict,
    *,
    index: int,
    total: int,
    critical_indices: set[int],
) -> CompactionClass:
    del total
    if not isinstance(message, dict):
        return CompactionClass.HISTORICAL_CONTEXT
    role = str(message.get("role", "")).strip().lower()
    if role == "tool":
        return CompactionClass.TOOL_TRACE
    if role == "assistant" and message.get("tool_calls"):
        return CompactionClass.TOOL_TRACE
    if index in critical_indices:
        return CompactionClass.CRITICAL
    return CompactionClass.HISTORICAL_CONTEXT


def build_compaction_plan(
    runner: Any,
    messages: list[dict],
    *,
    tier: CompactionPressureTier,
) -> _CompactionPlan:
    critical_indices = set(critical_message_indices(runner, messages))
    total = len(messages)
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

    preserve_tool_exchange = {
        idx for idx in {newest_assistant_tool, newest_tool_result} if idx >= 0
    }
    stage1_tool_args: list[int] = []
    stage2_tool_output: list[int] = []
    stage3_historical: list[int] = []
    for idx, msg in enumerate(messages):
        if idx == 0 or not isinstance(msg, dict):
            continue
        message_class = classify_message_for_compaction(
            msg,
            index=idx,
            total=total,
            critical_indices=critical_indices,
        )
        role = str(msg.get("role", "")).strip().lower()
        if message_class == CompactionClass.TOOL_TRACE:
            if idx in preserve_tool_exchange:
                continue
            if role == "assistant" and msg.get("tool_calls"):
                stage1_tool_args.append(idx)
            elif role == "tool":
                stage2_tool_output.append(idx)
            continue
        if message_class == CompactionClass.HISTORICAL_CONTEXT:
            stage3_historical.append(idx)

    stage4_merge: list[int] = []
    if tier == CompactionPressureTier.CRITICAL and total > 4:
        preserve_recent = int(
            getattr(
                runner,
                "_preserve_recent_critical_messages",
                runner.PRESERVE_RECENT_CRITICAL_MESSAGES,
            ),
        )
        merge_limit = max(1, total - max(3, preserve_recent + 1))
        for idx in range(1, merge_limit):
            if idx in preserve_tool_exchange:
                continue
            stage4_merge.append(idx)

    return _CompactionPlan(
        critical_indices=tuple(sorted(critical_indices)),
        stage1_tool_args=tuple(stage1_tool_args),
        stage2_tool_output=tuple(stage2_tool_output),
        stage3_historical=tuple(stage3_historical),
        stage4_merge=tuple(stage4_merge),
    )


async def compact_text(
    runner: Any,
    text: str,
    *,
    max_chars: int,
    label: str,
    logger: logging.Logger | None = None,
) -> str:
    value = str(text or "")
    if runner._runner_compaction_mode() == "off":
        runner._record_compaction_skip("policy_disabled")
        return value
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value

    runner._ensure_runner_compaction_state()
    key = runner._compaction_cache_key(value, max_chars=max_chars, label=label)
    cached = runner._runner_compaction_cache.get(key)
    if cached is not None:
        runner._record_compaction_skip("cache_hit")
        return cached

    if key in runner._runner_compaction_overshoot:
        runner._record_compaction_skip("no_gain")
        return value

    no_gain_attempt_limit = int(
        getattr(
            runner,
            "_compaction_no_gain_attempt_limit",
            runner.COMPACTION_NO_GAIN_ATTEMPT_LIMIT,
        ),
    )
    no_gain_attempts = int(runner._runner_compaction_no_gain.get(key, 0))
    if no_gain_attempts >= no_gain_attempt_limit:
        runner._record_compaction_skip("no_gain")
        return value

    compacted = await runner._compactor.compact(
        value,
        max_chars=max_chars,
        label=label,
    )
    runner._record_compactor_call()
    runner._runner_compaction_cache[key] = compacted
    runner._trim_compaction_cache(runner._runner_compaction_cache)

    min_delta = int(
        getattr(
            runner,
            "_compaction_no_gain_min_delta_chars",
            runner.COMPACTION_NO_GAIN_MIN_DELTA_CHARS,
        ),
    )
    reduction_delta = max(0, len(value) - len(compacted))
    if reduction_delta < max(1, min_delta):
        runner._runner_compaction_no_gain[key] = no_gain_attempts + 1
        runner._trim_compaction_cache(runner._runner_compaction_no_gain)
    else:
        runner._runner_compaction_no_gain.pop(key, None)

    if len(compacted) > max_chars:
        active_logger = logger or logging.getLogger(__name__)
        active_logger.warning(
            "Compaction exceeded budget for %s: got %d chars (limit %d)",
            label,
            len(compacted),
            max_chars,
        )
        runner._runner_compaction_overshoot.add(key)
        if len(runner._runner_compaction_overshoot) > 512:
            overflow = len(runner._runner_compaction_overshoot) - 512
            for stale in list(runner._runner_compaction_overshoot)[:overflow]:
                runner._runner_compaction_overshoot.discard(stale)
    return compacted


async def summarize_tool_data(runner: Any, data: dict | None) -> dict | None:
    if not isinstance(data, dict) or not data:
        return None
    compact_tool_arg_chars = int(
        getattr(
            runner,
            "_compact_tool_call_argument_chars",
            runner.COMPACT_TOOL_CALL_ARGUMENT_CHARS,
        ),
    )
    compact_tool_data_text_chars = max(
        80,
        int(round(compact_tool_arg_chars * 0.75)),
    )
    if len(data) > 12:
        packed = json.dumps(data, ensure_ascii=False, default=str)
        summary_text = await compact_text(
            runner,
            packed,
            max_chars=900,
            label="tool data payload",
        )
        return {
            "summary": summary_text,
            "key_count": len(data),
        }
    summary: dict = {}
    for key, value in data.items():
        if isinstance(value, str):
            summary[key] = await compact_text(
                runner,
                value,
                max_chars=compact_tool_data_text_chars,
                label=f"tool data {key}",
            )
        elif isinstance(value, (int, float, bool)) or value is None:
            summary[key] = value
        elif isinstance(value, dict):
            packed = json.dumps(value, ensure_ascii=False, default=str)
            summary[key] = await compact_text(
                runner,
                packed,
                max_chars=compact_tool_arg_chars,
                label=f"tool data object {key}",
            )
        elif isinstance(value, list):
            packed = json.dumps(value, ensure_ascii=False, default=str)
            summary[key] = await compact_text(
                runner,
                packed,
                max_chars=compact_tool_arg_chars,
                label=f"tool data list {key}",
            )
        else:
            summary[key] = str(type(value).__name__)
    return summary or None


async def summarize_tool_call_arguments(
    runner: Any,
    args: object,
    *,
    max_chars: int,
    label: str,
) -> dict:
    if isinstance(args, dict):
        summary = await summarize_tool_data(runner, args) or {}
        packed = json.dumps(summary, ensure_ascii=False, default=str)
        if len(packed) <= max_chars:
            return summary
        compacted = await compact_text(
            runner,
            packed,
            max_chars=max_chars,
            label=label,
        )
        return {
            "summary": compacted,
            "key_count": len(args),
        }

    packed = json.dumps(args, ensure_ascii=False, default=str)
    compacted = await compact_text(
        runner,
        packed,
        max_chars=max_chars,
        label=label,
    )
    return {"summary": compacted}


async def compact_assistant_tool_calls(
    runner: Any,
    tool_calls: object,
    *,
    max_chars: int,
) -> list[dict] | None:
    if not isinstance(tool_calls, list):
        return None

    compacted_calls: list[dict] = []
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        compact_item = dict(item)
        function = compact_item.get("function")
        if not isinstance(function, dict):
            compacted_calls.append(compact_item)
            continue

        function_copy = dict(function)
        name = str(function_copy.get("name", "") or "tool")
        raw_args = function_copy.get("arguments", "{}")
        parsed_args: object
        if isinstance(raw_args, dict):
            parsed_args = raw_args
        elif isinstance(raw_args, str):
            try:
                decoded = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                parsed_args = {"raw": raw_args}
            else:
                parsed_args = decoded
        else:
            parsed_args = {"raw": str(raw_args)}

        compact_args = await summarize_tool_call_arguments(
            runner,
            parsed_args,
            max_chars=max_chars,
            label=f"{name} assistant tool-call args",
        )
        function_copy["arguments"] = json.dumps(
            compact_args,
            ensure_ascii=False,
            default=str,
        )
        compact_item["function"] = function_copy
        compacted_calls.append(compact_item)

    return compacted_calls or None


async def compact_tool_message_content(
    runner: Any,
    content: str,
    *,
    max_output_chars: int,
) -> str:
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return await compact_text(
            runner,
            str(content),
            max_chars=max_output_chars,
            label="tool message content",
        )

    if not isinstance(parsed, dict):
        return await compact_text(
            runner,
            str(content),
            max_chars=max_output_chars,
            label="tool message payload",
        )

    output = await compact_text(
        runner,
        str(parsed.get("output", "")),
        max_chars=max_output_chars,
        label="tool message output",
    )
    payload: dict = {
        "success": bool(parsed.get("success", False)),
        "output": output,
        "error": parsed.get("error"),
        "files_changed": list(parsed.get("files_changed", [])),
    }
    if len(payload["files_changed"]) > 20:
        files_text = "\n".join(str(item) for item in payload["files_changed"])
        payload["files_changed_summary"] = await compact_text(
            runner,
            files_text,
            max_chars=320,
            label="tool message files changed",
        )
        payload["files_changed_count"] = len(payload["files_changed"])
        payload.pop("files_changed", None)

    data_summary = await summarize_tool_data(runner, parsed.get("data"))
    if data_summary:
        payload["data"] = data_summary

    raw_blocks = parsed.get("content_blocks")
    if isinstance(raw_blocks, list) and raw_blocks:
        compact_blocks: list[dict] = []
        for block in raw_blocks:
            if not isinstance(block, dict):
                continue
            compact = dict(block)
            for key in ("text", "text_fallback", "extracted_text", "thinking"):
                value = compact.get(key)
                if isinstance(value, str):
                    compact[key] = await compact_text(
                        runner,
                        value,
                        max_chars=min(max_output_chars, 400),
                        label=f"tool block {key}",
                    )
            compact_blocks.append(compact)
        if compact_blocks:
            payload["content_blocks"] = compact_blocks
    return json.dumps(payload)


async def compact_messages_for_model_tiered(
    runner: Any,
    messages: list[dict],
    *,
    remaining_seconds: float | None = None,
    logger: logging.Logger,
) -> list[dict]:
    runner._reset_compaction_runtime_stats()
    mode = runner._runner_compaction_mode()
    if len(messages) < 3:
        set_compaction_diagnostics(runner, {
            "compaction_policy_mode": mode,
            "compaction_pressure_tier": CompactionPressureTier.NORMAL.value,
            "compaction_pressure_ratio": 0.0,
            "compaction_stage": "none",
            "compaction_candidate_count": 0,
            "compaction_skipped_reason": "short_history",
            "compaction_compactor_calls": 0,
        })
        return messages

    context_budget = int(
        getattr(runner, "_max_model_context_tokens", runner.MAX_MODEL_CONTEXT_TOKENS),
    )
    soft_ratio = float(
        getattr(
            runner,
            "_compaction_pressure_ratio_soft",
            runner.COMPACTION_PRESSURE_RATIO_SOFT,
        ),
    )
    hard_ratio = float(
        getattr(
            runner,
            "_compaction_pressure_ratio_hard",
            runner.COMPACTION_PRESSURE_RATIO_HARD,
        ),
    )
    hard_ratio = max(soft_ratio + 0.01, hard_ratio)
    compact_tool_call_args = int(
        getattr(
            runner,
            "_compact_tool_call_argument_chars",
            runner.COMPACT_TOOL_CALL_ARGUMENT_CHARS,
        ),
    )
    compact_tool_output = int(
        getattr(
            runner,
            "_compact_tool_result_output_chars",
            runner.COMPACT_TOOL_RESULT_OUTPUT_CHARS,
        ),
    )
    compact_text_chars = int(
        getattr(runner, "_compact_text_output_chars", runner.COMPACT_TEXT_OUTPUT_CHARS),
    )
    minimal_text_chars = int(
        getattr(runner, "_minimal_text_output_chars", runner.MINIMAL_TEXT_OUTPUT_CHARS),
    )

    estimate_before = estimate_message_tokens(messages)
    pressure_ratio = (
        estimate_before / max(1, context_budget)
        if context_budget > 0
        else 0.0
    )
    tier = compute_compaction_pressure_tier(runner, pressure_ratio)
    if tier == CompactionPressureTier.NORMAL:
        set_compaction_diagnostics(runner, {
            "compaction_policy_mode": mode,
            "compaction_pressure_tier": tier.value,
            "compaction_pressure_ratio": round(pressure_ratio, 4),
            "compaction_stage": "none",
            "compaction_candidate_count": 0,
            "compaction_skipped_reason": "no_pressure",
            "compaction_est_tokens_before": estimate_before,
            "compaction_est_tokens_after": estimate_before,
            "compaction_compactor_calls": 0,
        })
        return messages

    compacted: list[dict] = [
        dict(message) if isinstance(message, dict) else message
        for message in messages
    ]
    plan = build_compaction_plan(runner, compacted, tier=tier)
    timeout_guard_active = runner._is_timeout_guard_active(remaining_seconds)
    total_candidates = (
        len(plan.stage1_tool_args)
        + len(plan.stage2_tool_output)
        + len(plan.stage3_historical)
        + len(plan.stage4_merge)
    )
    if total_candidates == 0:
        set_compaction_diagnostics(runner, {
            "compaction_policy_mode": mode,
            "compaction_pressure_tier": tier.value,
            "compaction_pressure_ratio": round(pressure_ratio, 4),
            "compaction_stage": "none",
            "compaction_candidate_count": 0,
            "compaction_skipped_reason": "policy_preserve",
            "compaction_est_tokens_before": estimate_before,
            "compaction_est_tokens_after": estimate_before,
            "compaction_compactor_calls": 0,
        })
        return compacted

    estimate_after = estimate_before
    pressure_after = pressure_ratio
    applied_stages: list[str] = []

    async def _compact_stage_1() -> bool:
        changed = False
        for idx in plan.stage1_tool_args:
            msg = compacted[idx]
            if not isinstance(msg, dict):
                continue
            compact_calls = await compact_assistant_tool_calls(
                runner,
                msg.get("tool_calls"),
                max_chars=compact_tool_call_args,
            )
            if compact_calls is not None and compact_calls != msg.get("tool_calls"):
                msg["tool_calls"] = compact_calls
                changed = True
        return changed

    async def _compact_stage_2() -> bool:
        changed = False
        for idx in plan.stage2_tool_output:
            msg = compacted[idx]
            if not isinstance(msg, dict):
                continue
            prior = msg.get("content", "")
            compacted_content = await compact_tool_message_content(
                runner,
                prior,
                max_output_chars=compact_tool_output,
            )
            if compacted_content != prior:
                msg["content"] = compacted_content
                changed = True
        return changed

    async def _compact_stage_3() -> bool:
        changed = False
        text_budget = (
            compact_text_chars
            if tier == CompactionPressureTier.PRESSURE
            else minimal_text_chars
        )
        for idx in plan.stage3_historical:
            msg = compacted[idx]
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip().lower()
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            if role == "user" and content.startswith(runner._TODO_REMINDER_PREFIX):
                continue
            if len(content) <= text_budget:
                continue
            label = f"{role or 'message'} context"
            compacted_content = await compact_text(
                runner,
                content,
                max_chars=text_budget,
                label=label,
            )
            if compacted_content != content:
                msg["content"] = compacted_content
                changed = True
        return changed

    async def _compact_stage_4() -> bool:
        if tier != CompactionPressureTier.CRITICAL:
            return False
        if pressure_after <= hard_ratio:
            return False
        if not plan.stage4_merge:
            return False
        merge_lines: list[str] = []
        for idx in plan.stage4_merge:
            msg = compacted[idx]
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip().lower() or "unknown"
            if role == "assistant" and msg.get("tool_calls"):
                tool_names = [
                    str(tc.get("function", {}).get("name", "tool"))
                    for tc in list(msg.get("tool_calls", []))
                    if isinstance(tc, dict)
                ]
                merge_lines.append(
                    f"[assistant/tool_call] {', '.join(tool_names) or 'tool call'}",
                )
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            merge_lines.append(f"[{role}] {content}")
        merged_text = "\n".join(merge_lines).strip()
        if not merged_text:
            return False
        merged_summary = await compact_text(
            runner,
            merged_text,
            max_chars=max(480, int(compact_text_chars * 1.5)),
            label="prior conversation context",
        )
        merge_set = set(plan.stage4_merge)
        rebuilt = [compacted[0]]
        rebuilt.append({
            "role": "user",
            "content": f"Prior compacted context:\n{merged_summary}",
        })
        for idx, msg in enumerate(compacted[1:], start=1):
            if idx in merge_set:
                continue
            rebuilt.append(msg)
        compacted[:] = rebuilt
        return True

    async def _run_stage(stage_name: str, apply_fn) -> bool:
        nonlocal estimate_after, pressure_after
        changed = await apply_fn()
        if changed:
            estimate_after = estimate_message_tokens(compacted)
            pressure_after = estimate_after / max(1, context_budget)
            applied_stages.append(stage_name)
        return changed

    await _run_stage("stage_1_tool_args", _compact_stage_1)
    if pressure_after <= soft_ratio:
        pass
    else:
        if timeout_guard_active:
            runner._record_compaction_skip("timeout_guard")
        else:
            await _run_stage("stage_2_tool_outputs", _compact_stage_2)
            if pressure_after > soft_ratio:
                await _run_stage("stage_3_historical", _compact_stage_3)
            if pressure_after > soft_ratio and tier == CompactionPressureTier.CRITICAL:
                await _run_stage("stage_4_merge", _compact_stage_4)

    stats = dict(getattr(runner, "_compaction_runtime_stats", {}))
    compactor_calls = int(stats.get("compactor_calls", 0))
    if compactor_calls > int(
        getattr(
            runner,
            "_compaction_churn_warning_calls",
            runner.COMPACTION_CHURN_WARNING_CALLS,
        ),
    ):
        logger.warning(
            "High compactor churn in runner tiered policy: calls=%d tier=%s "
            "tokens_before=%d tokens_after=%d",
            compactor_calls,
            tier.value,
            estimate_before,
            estimate_after,
        )

    skipped_reason = ""
    if timeout_guard_active and pressure_after > soft_ratio:
        skipped_reason = "timeout_guard"
    elif not applied_stages:
        skip_reasons = stats.get("skip_reasons", {})
        if isinstance(skip_reasons, dict) and skip_reasons:
            skipped_reason = str(
                max(skip_reasons.items(), key=lambda item: item[1])[0],
            )
        else:
            skipped_reason = "no_gain"

    set_compaction_diagnostics(runner, {
        "compaction_policy_mode": mode,
        "compaction_pressure_tier": tier.value,
        "compaction_pressure_ratio": round(pressure_ratio, 4),
        "compaction_pressure_ratio_after": round(pressure_after, 4),
        "compaction_stage": applied_stages[-1] if applied_stages else "none",
        "compaction_applied_stages": applied_stages,
        "compaction_candidate_count": total_candidates,
        "compaction_skipped_reason": skipped_reason,
        "compaction_est_tokens_before": estimate_before,
        "compaction_est_tokens_after": estimate_after,
        "compaction_compactor_calls": compactor_calls,
        "compaction_skip_reasons": stats.get("skip_reasons", {}),
    })
    return compacted


async def compact_messages_for_model_legacy(
    runner: Any,
    messages: list[dict],
) -> list[dict]:
    """Legacy eager compaction path kept for rollout safety."""
    runner._reset_compaction_runtime_stats()
    mode = runner._runner_compaction_mode()
    if len(messages) < 3:
        set_compaction_diagnostics(runner, {
            "compaction_policy_mode": mode,
            "compaction_stage": "none",
            "compaction_candidate_count": 0,
            "compaction_skipped_reason": "short_history",
            "compaction_compactor_calls": 0,
        })
        return messages

    context_budget = int(
        getattr(runner, "_max_model_context_tokens", runner.MAX_MODEL_CONTEXT_TOKENS),
    )
    compact_tool_call_args = int(
        getattr(
            runner,
            "_compact_tool_call_argument_chars",
            runner.COMPACT_TOOL_CALL_ARGUMENT_CHARS,
        ),
    )
    compact_tool_output = int(
        getattr(
            runner,
            "_compact_tool_result_output_chars",
            runner.COMPACT_TOOL_RESULT_OUTPUT_CHARS,
        ),
    )
    compact_text_chars = int(
        getattr(runner, "_compact_text_output_chars", runner.COMPACT_TEXT_OUTPUT_CHARS),
    )
    minimal_text_chars = int(
        getattr(runner, "_minimal_text_output_chars", runner.MINIMAL_TEXT_OUTPUT_CHARS),
    )

    estimate_before = estimate_message_tokens(messages)
    if estimate_before <= context_budget:
        set_compaction_diagnostics(runner, {
            "compaction_policy_mode": mode,
            "compaction_stage": "none",
            "compaction_candidate_count": 0,
            "compaction_skipped_reason": "no_pressure",
            "compaction_est_tokens_before": estimate_before,
            "compaction_est_tokens_after": estimate_before,
            "compaction_compactor_calls": 0,
        })
        return messages

    compacted: list[dict] = [
        dict(message) if isinstance(message, dict) else message
        for message in messages
    ]
    stage_name = "stage_1_tool_args"
    candidate_count = 0

    for msg in compacted:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        if role != "assistant" or not msg.get("tool_calls"):
            continue
        candidate_count += 1
        compact_calls = await compact_assistant_tool_calls(
            runner,
            msg.get("tool_calls"),
            max_chars=compact_tool_call_args,
        )
        if compact_calls is not None:
            msg["tool_calls"] = compact_calls

    async def _apply_pass(
        *,
        preserve_recent: int,
        tool_chars: int,
        text_chars: int,
        stage: str,
    ) -> None:
        nonlocal stage_name, candidate_count
        preserve_from = max(1, len(compacted) - preserve_recent)
        stage_name = stage
        for idx, msg in enumerate(compacted):
            if idx == 0 or idx >= preserve_from or not isinstance(msg, dict):
                continue
            candidate_count += 1
            role = str(msg.get("role", "")).strip().lower()
            if role == "tool":
                msg["content"] = await compact_tool_message_content(
                    runner,
                    msg.get("content", ""),
                    max_output_chars=tool_chars,
                )
            elif role == "assistant":
                if msg.get("tool_calls"):
                    msg["content"] = runner.TOOL_CALL_CONTEXT_PLACEHOLDER
                    compact_calls = await compact_assistant_tool_calls(
                        runner,
                        msg.get("tool_calls"),
                        max_chars=max(80, min(text_chars, compact_tool_call_args)),
                    )
                    if compact_calls is not None:
                        msg["tool_calls"] = compact_calls
                else:
                    content = msg.get("content")
                    if isinstance(content, str) and len(content) > text_chars:
                        msg["content"] = await compact_text(
                            runner,
                            content,
                            max_chars=text_chars,
                            label="assistant context",
                        )
            elif role == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    if content.startswith(runner._TODO_REMINDER_PREFIX):
                        msg["content"] = (
                            "Continue current subtask only. "
                            "Do NOT move to the next subtask."
                        )
                    elif len(content) > text_chars:
                        msg["content"] = await compact_text(
                            runner,
                            content,
                            max_chars=text_chars,
                            label="user context",
                        )
            elif role == "system":
                content = msg.get("content")
                if isinstance(content, str) and len(content) > text_chars:
                    msg["content"] = await compact_text(
                        runner,
                        content,
                        max_chars=text_chars,
                        label="system context",
                    )

    await _apply_pass(
        preserve_recent=8,
        tool_chars=compact_tool_output,
        text_chars=compact_text_chars,
        stage="stage_2_general",
    )
    if estimate_message_tokens(compacted) > context_budget:
        await _apply_pass(
            preserve_recent=4,
            tool_chars=120,
            text_chars=minimal_text_chars,
            stage="stage_3_minimal",
        )

    if estimate_message_tokens(compacted) > context_budget:
        preserve_from = max(1, len(compacted) - 3)
        old_context = compacted[1:preserve_from]
        recent = compacted[preserve_from:]
        while (
            recent
            and isinstance(recent[0], dict)
            and recent[0].get("role") == "tool"
        ):
            recent = recent[1:]

        if old_context:
            merged_lines: list[str] = []
            for msg in old_context:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "")).strip().lower() or "unknown"
                content = msg.get("content", "")
                if role == "assistant" and msg.get("tool_calls"):
                    tool_names = [
                        str(tc.get("function", {}).get("name", "tool"))
                        for tc in list(msg.get("tool_calls", []))
                        if isinstance(tc, dict)
                    ]
                    merged_lines.append(
                        f"[assistant/tool_call] {', '.join(tool_names) or 'tool call'}",
                    )
                    continue
                if not isinstance(content, str):
                    content = str(content)
                merged_lines.append(f"[{role}] {content}")

            merged_text = "\n".join(merged_lines).strip()
            if merged_text:
                stage_name = "stage_4_merge"
                merged_summary = await compact_text(
                    runner,
                    merged_text,
                    max_chars=700,
                    label="prior conversation context",
                )
                compacted = [
                    compacted[0],
                    {
                        "role": "user",
                        "content": f"Prior compacted context:\n{merged_summary}",
                    },
                    *recent,
                ]

    estimate_after = estimate_message_tokens(compacted)
    stats = dict(getattr(runner, "_compaction_runtime_stats", {}))
    set_compaction_diagnostics(runner, {
        "compaction_policy_mode": mode,
        "compaction_stage": stage_name,
        "compaction_candidate_count": candidate_count,
        "compaction_skipped_reason": "",
        "compaction_est_tokens_before": estimate_before,
        "compaction_est_tokens_after": estimate_after,
        "compaction_compactor_calls": int(stats.get("compactor_calls", 0)),
        "compaction_skip_reasons": stats.get("skip_reasons", {}),
    })
    return compacted
