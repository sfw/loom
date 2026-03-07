"""Pipeline orchestration helpers for semantic compactor."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from loom.models.base import ModelProvider


async def compact_with_model(
    *,
    model: ModelProvider,
    text: str,
    max_chars: int,
    label: str,
    max_chunk_chars: int,
    max_reduction_rounds: int,
    compact_once: Callable[..., Awaitable[tuple[str, int]]],
    chunked_compaction: Callable[..., Awaitable[tuple[str, int]]],
) -> tuple[str, int]:
    current = text
    total_retry_count = 0
    if len(current) > max_chunk_chars:
        current, retries_used = await chunked_compaction(
            model=model,
            text=current,
            max_chars=max_chars,
            label=label,
        )
        total_retry_count += retries_used

    rounds = 0
    while len(current) > max_chars and rounds < max_reduction_rounds:
        # Always aim at the caller's requested budget instead of inflating
        # per-round targets from the current text length.
        candidate, retries_used = await compact_once(
            model=model,
            text=current,
            max_chars=max_chars,
            label=label,
            strict=(rounds >= 1),
        )
        total_retry_count += retries_used
        if not candidate:
            break
        if len(candidate) >= len(current):
            break
        current = candidate
        rounds += 1

    return current.strip() or text, total_retry_count


async def chunked_compaction(
    *,
    model: ModelProvider,
    text: str,
    max_chars: int,
    label: str,
    max_chunk_chars: int,
    max_chunks_per_round: int,
    min_compact_target_chars: int,
    split_text: Callable[[str, int], list[str]],
    compact_once: Callable[..., Awaitable[tuple[str, int]]],
) -> tuple[str, int]:
    chunks = split_text(text, max_chunk_chars)
    if not chunks:
        return text, 0

    total_retry_count = 0

    # Hierarchical pre-merge when there are too many chunks.
    while len(chunks) > max_chunks_per_round:
        merged: list[str] = []
        for idx in range(0, len(chunks), 3):
            block = "\n\n".join(chunks[idx:idx + 3])
            merged_text, retries_used = await compact_once(
                model=model,
                text=block,
                max_chars=max(
                    min_compact_target_chars,
                    int(max_chars),
                ),
                label=f"{label} pre-merge",
                strict=False,
            )
            total_retry_count += retries_used
            merged.append(merged_text or block)
        chunks = merged

    map_target = max(
        min_compact_target_chars,
        min(int(max_chars), 2_400),
    )
    mapped: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        if len(chunk) <= map_target:
            mapped.append(chunk)
            continue
        mapped_text, retries_used = await compact_once(
            model=model,
            text=chunk,
            max_chars=map_target,
            label=f"{label} chunk {idx}/{len(chunks)}",
            strict=False,
        )
        total_retry_count += retries_used
        mapped.append(mapped_text or chunk)

    return (
        "\n".join(part.strip() for part in mapped if part.strip()).strip(),
        total_retry_count,
    )


def build_compactor_user_prompt(
    *,
    label: str,
    source_text: str,
    target_chars: int,
    hard_limit: int,
    retry_hint: str = "",
) -> str:
    retry_block = ""
    if retry_hint:
        retry_block = f"Retry guidance:\n{retry_hint}\n\n"
    return (
        f"Compact the following {label}.\n"
        f"Target character budget: <= {target_chars}.\n"
        f"Hard character limit: <= {hard_limit}.\n"
        "You MUST keep the compressed_text value at or below this hard "
        "character limit.\n\n"
        f"{retry_block}"
        "Preservation requirements:\n"
        "- Keep all facts, numbers, dates, file paths, URLs, IDs, commands, "
        "errors, acceptance criteria, and deliverable names.\n"
        "- Keep dependency/ordering constraints.\n"
        "- Keep meaning and intent intact.\n"
        "- Remove repetition and filler.\n"
        "- Output ONLY transformed source content.\n"
        "- Do NOT include meta commentary about the task, user intent, "
        "or your process (for example, avoid phrases like "
        "\"The user wants\" or \"I need to\").\n"
        "- Do NOT add preambles, explanations, or analysis.\n"
        "- Respond with exactly one JSON object with one key: "
        "{\"compressed_text\":\"<compacted text>\"}.\n"
        "- Do not include markdown fences.\n\n"
        f"Source text:\n{source_text}"
    )


def next_validation_retry(
    *,
    current_source_text: str,
    current_target_chars: int,
    current_max_tokens: int | None,
    hard_limit: int,
    token_ceiling: int,
    validation_error: str,
    parsed_text: str | None,
    response_text: str,
    response_finish_reason: str,
    min_compact_target_chars: int,
) -> tuple[str, int, int | None, str, str]:
    reason = str(validation_error or "validation_failed").strip() or "validation_failed"
    finish_reason = str(response_finish_reason or "").strip().lower()
    parsed = str(parsed_text or "").strip()
    raw = str(response_text or "").strip()
    next_source_text = parsed or raw or current_source_text
    next_target_chars = int(current_target_chars)
    next_max_tokens = current_max_tokens
    retry_hint = (
        "Previous attempt failed validation. Repair the draft below and return "
        "valid JSON only."
    )

    if reason == "invalid_json":
        retry_hint = (
            "Previous attempt produced invalid JSON. Repair the draft below and "
            "return one valid JSON object with a single compressed_text key."
        )
        if finish_reason == "length" and isinstance(next_max_tokens, int):
            grown = max(next_max_tokens + 64, int(round(next_max_tokens * 1.35)))
            next_max_tokens = min(token_ceiling, grown)
            retry_hint = (
                "Previous attempt was truncated before valid JSON was completed. "
                "Repair the draft below into valid JSON and finish the object."
            )
    elif reason == "output_exceeds_target":
        target_floor = min(min_compact_target_chars, hard_limit)
        reduced_target = int(round(next_target_chars * 0.9))
        next_target_chars = max(1, min(hard_limit, max(target_floor, reduced_target)))
        retry_hint = (
            "Previous draft exceeded the hard character limit. Re-compress the "
            "draft below more aggressively while preserving concrete details."
        )
    elif reason.startswith("meta_commentary"):
        retry_hint = (
            "Previous draft included meta commentary. Remove commentary and return "
            "only transformed source content in valid JSON."
        )

    return (
        next_source_text,
        next_target_chars,
        next_max_tokens,
        reason,
        retry_hint,
    )


def split_text(text: str, chunk_chars: int) -> list[str]:
    chunks: list[str] = []
    remaining = text.strip()
    while remaining:
        if len(remaining) <= chunk_chars:
            chunks.append(remaining)
            break

        split_at = remaining.rfind("\n\n", 0, chunk_chars)
        if split_at < int(chunk_chars * 0.55):
            split_at = remaining.rfind("\n", 0, chunk_chars)
        if split_at < int(chunk_chars * 0.55):
            split_at = remaining.rfind(". ", 0, chunk_chars)
        if split_at < int(chunk_chars * 0.55):
            split_at = chunk_chars

        chunk = remaining[:split_at].strip()
        if not chunk:
            break
        chunks.append(chunk)
        remaining = remaining[split_at:].strip()
    return chunks


async def compact_once(
    *,
    model: ModelProvider,
    text: str,
    max_chars: int,
    label: str,
    max_validation_retries: int,
    min_compact_target_chars: int,
    configured_temperature: float | None,
    response_format: dict[str, Any] | None,
    compactor_hard_limit_chars: Callable[[int, ModelProvider], int],
    compactor_target_chars: Callable[[int], int],
    compactor_response_max_tokens: Callable[[int, ModelProvider], int | None],
    compactor_token_ceiling_for_hard_limit: Callable[[int, ModelProvider], int],
    invoke_compactor_model: Callable[..., Awaitable[Any]],
    extract_compacted_text: Callable[[str], tuple[str | None, str]],
    extract_partial_compressed_text: Callable[[str], str | None],
    validate_compacted_text: Callable[[str, int], str],
    emit_model_event: Callable[..., None],
    logger: logging.Logger,
) -> tuple[str, int]:
    requested_max_chars = max(1, int(max_chars))
    hard_limit = compactor_hard_limit_chars(requested_max_chars, model)
    target_chars = compactor_target_chars(hard_limit)
    max_tokens = compactor_response_max_tokens(hard_limit, model)
    system = (
        "You are a semantic context compactor for LLM pipelines. "
        "Preserve concrete details and required instructions while removing "
        "redundancy. Never invent facts. "
        "Return strict JSON only."
    )
    source_text = str(text or "")
    current_target_chars = int(target_chars)
    current_max_tokens = max_tokens
    retry_reason = ""
    retry_hint = ""
    token_ceiling = compactor_token_ceiling_for_hard_limit(hard_limit, model)

    for validation_attempt in range(1, max_validation_retries + 2):
        # Compactor length compliance must be strict on every attempt.
        attempt_strict = True
        user = build_compactor_user_prompt(
            label=label,
            source_text=source_text,
            target_chars=current_target_chars,
            hard_limit=hard_limit,
            retry_hint=retry_hint,
        )
        response = await invoke_compactor_model(
            model=model,
            system=system,
            user=user,
            requested_max_chars=requested_max_chars,
            target_chars=current_target_chars,
            hard_limit=hard_limit,
            max_tokens=current_max_tokens,
            temperature=configured_temperature,
            label=label,
            strict=attempt_strict,
            response_format=response_format,
            validation_attempt=validation_attempt,
        )
        if isinstance(response, Exception):
            raise response

        response_text = str(response.text or "")
        response_finish_reason = str(getattr(response, "finish_reason", "") or "").strip()
        parsed_text, parse_error = extract_compacted_text(response_text)
        recovered_partial_json = False
        if parse_error == "invalid_json" and response_finish_reason.lower() == "length":
            partial_text = extract_partial_compressed_text(response_text)
            if partial_text:
                parsed_text = partial_text
                parse_error = ""
                recovered_partial_json = True
        validation_error = ""
        target_delta_chars = 0
        warning_reason = ""
        warning_delta_chars = 0
        warning_kept_output = False
        if parse_error:
            validation_error = parse_error
        elif parsed_text is not None:
            validation_error = validate_compacted_text(parsed_text, hard_limit)
            target_delta_chars = len(parsed_text) - hard_limit

        if (
            validation_error == "output_exceeds_target"
            and parsed_text is not None
            and validation_attempt > max_validation_retries
        ):
            warning_reason = "output_exceeds_target"
            warning_delta_chars = max(0, len(parsed_text) - hard_limit)
            warning_kept_output = True
            validation_error = ""
            logger.warning(
                "Semantic compactor kept over-target output for %s: "
                "target=%d received=%d delta=%d",
                label,
                hard_limit,
                len(parsed_text),
                warning_delta_chars,
            )

        is_valid = not validation_error and parsed_text is not None
        output_chars = len(parsed_text or "")
        emit_model_event(
            model_name=model.name,
            phase="validation",
            details={
                "operation": "complete",
                "origin": "semantic_compactor.complete",
                "compactor_label": label,
                "compactor_requested_max_chars": requested_max_chars,
                "compactor_target_chars": current_target_chars,
                "compactor_hard_limit_chars": hard_limit,
                "compactor_limit_chars": hard_limit,
                "compactor_token_budget_chars": hard_limit,
                "compactor_max_tokens": current_max_tokens,
                "compactor_strict": bool(attempt_strict),
                "compactor_validation_attempt": validation_attempt,
                "compactor_retry_count": validation_attempt - 1,
                "compactor_retry_reason": retry_reason,
                "compactor_retry_source_chars": len(source_text),
                "compactor_response_chars": len(response_text),
                "compactor_response_finish_reason": response_finish_reason,
                "compactor_partial_json_recovered": bool(recovered_partial_json),
                "compactor_output_hard_capped": False,
                "compactor_target_delta_chars": int(target_delta_chars),
                "compactor_warning": bool(warning_kept_output),
                "compactor_warning_reason": warning_reason,
                "compactor_warning_delta_chars": int(warning_delta_chars),
                "compactor_warning_target_chars": int(hard_limit),
                "compactor_warning_received_chars": int(output_chars),
                "compactor_output_chars": output_chars,
                "compactor_output_valid": bool(is_valid),
                "compactor_invalid_reason": validation_error,
            },
        )
        if is_valid:
            return parsed_text, validation_attempt - 1

        if validation_attempt > max_validation_retries:
            raise RuntimeError(
                "Semantic compactor produced invalid structured output: "
                f"{validation_error}",
            )

        (
            source_text,
            current_target_chars,
            current_max_tokens,
            retry_reason,
            retry_hint,
        ) = next_validation_retry(
            current_source_text=source_text,
            current_target_chars=current_target_chars,
            current_max_tokens=current_max_tokens,
            hard_limit=hard_limit,
            token_ceiling=token_ceiling,
            validation_error=validation_error,
            parsed_text=parsed_text,
            response_text=response_text,
            response_finish_reason=response_finish_reason,
            min_compact_target_chars=min_compact_target_chars,
        )

    raise RuntimeError("Semantic compactor exceeded validation retry budget")
