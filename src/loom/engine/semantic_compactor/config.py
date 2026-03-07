"""Budget and boundary helpers for semantic compactor configuration."""

from __future__ import annotations

from typing import Any


def compactor_response_max_tokens(
    limit_chars: int,
    model: Any,
    *,
    response_tokens_floor: int,
    response_tokens_ratio: float,
    response_tokens_buffer: int,
    json_headroom_chars_floor: int,
    json_headroom_chars_ratio: float,
    json_headroom_chars_cap: int,
    chars_per_token_estimate: float,
    token_headroom: int,
) -> int | None:
    """Estimate max_tokens tightly around the validated hard character limit."""
    limit = max(1, int(limit_chars))
    token_ceiling = compactor_token_ceiling_for_hard_limit(
        limit,
        model,
        json_headroom_chars_floor=json_headroom_chars_floor,
        json_headroom_chars_ratio=json_headroom_chars_ratio,
        json_headroom_chars_cap=json_headroom_chars_cap,
        chars_per_token_estimate=chars_per_token_estimate,
        token_headroom=token_headroom,
    )
    if (
        response_tokens_floor <= 0
        and response_tokens_ratio <= 0.0
        and response_tokens_buffer <= 0
    ):
        return token_ceiling

    estimated = int(limit * response_tokens_ratio) + response_tokens_buffer
    budget = max(response_tokens_floor, estimated)
    budget = max(1, budget)
    budget = min(budget, token_ceiling)
    return max(1, budget)


def compactor_json_headroom_chars(
    hard_limit_chars: int,
    *,
    json_headroom_chars_floor: int,
    json_headroom_chars_ratio: float,
    json_headroom_chars_cap: int,
) -> int:
    limit = max(1, int(hard_limit_chars))
    estimated = int(round(limit * json_headroom_chars_ratio))
    estimated = max(json_headroom_chars_floor, estimated)
    return max(1, min(json_headroom_chars_cap, estimated))


def compactor_token_ceiling_for_hard_limit(
    hard_limit_chars: int,
    model: Any,
    *,
    json_headroom_chars_floor: int,
    json_headroom_chars_ratio: float,
    json_headroom_chars_cap: int,
    chars_per_token_estimate: float,
    token_headroom: int,
) -> int:
    limit = max(1, int(hard_limit_chars))
    char_budget = limit + compactor_json_headroom_chars(
        limit,
        json_headroom_chars_floor=json_headroom_chars_floor,
        json_headroom_chars_ratio=json_headroom_chars_ratio,
        json_headroom_chars_cap=json_headroom_chars_cap,
    )
    token_budget = int(round(char_budget / chars_per_token_estimate))
    token_budget += token_headroom
    token_budget = max(1, token_budget)
    configured_max = getattr(model, "configured_max_tokens", None)
    if isinstance(configured_max, int) and configured_max > 0:
        token_budget = min(token_budget, configured_max)
    return max(1, token_budget)


def compactor_hard_limit_chars(
    requested_chars: int,
    model: Any,
    *,
    response_tokens_ratio: float,
    response_tokens_buffer: int,
) -> int:
    """Bound requested output chars by model token capacity when known."""
    limit = max(1, int(requested_chars))
    configured_max = getattr(model, "configured_max_tokens", None)
    if (
        not isinstance(configured_max, int)
        or configured_max <= 0
        or response_tokens_ratio <= 0.0
    ):
        return limit

    available_tokens = configured_max - response_tokens_buffer
    if available_tokens <= 0:
        return 1

    ceiling = int(available_tokens / response_tokens_ratio)
    if ceiling <= 0:
        return 1
    return min(limit, ceiling)


def compactor_target_chars(hard_limit: int, *, target_chars_ratio: float) -> int:
    """Choose a conservative target below hard limit for JSON envelope room."""
    limit = max(1, int(hard_limit))
    target = int(round(limit * target_chars_ratio))
    return max(1, min(limit, target))
