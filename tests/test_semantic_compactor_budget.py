"""Budget math parity tests for semantic compactor helpers."""

from __future__ import annotations

from loom.engine.semantic_compactor import SemanticCompactor
from loom.engine.semantic_compactor import config as compactor_config


class _BudgetModel:
    def __init__(self, configured_max_tokens: int | None) -> None:
        self.configured_max_tokens = configured_max_tokens


def test_compactor_target_chars_small_and_near_budget_bounds() -> None:
    assert compactor_config.compactor_target_chars(0, target_chars_ratio=0.82) == 1
    assert compactor_config.compactor_target_chars(200, target_chars_ratio=0.82) == 164
    assert compactor_config.compactor_target_chars(200, target_chars_ratio=1.25) == 200


def test_compactor_hard_limit_chars_returns_one_when_buffer_exhausts_capacity() -> None:
    model = _BudgetModel(configured_max_tokens=64)

    hard_limit = compactor_config.compactor_hard_limit_chars(
        400,
        model,
        response_tokens_ratio=0.5,
        response_tokens_buffer=96,
    )

    assert hard_limit == 1


def test_compactor_hard_limit_chars_keeps_requested_when_ceiling_not_binding() -> None:
    model = _BudgetModel(configured_max_tokens=4096)

    hard_limit = compactor_config.compactor_hard_limit_chars(
        720,
        model,
        response_tokens_ratio=0.55,
        response_tokens_buffer=256,
    )

    assert hard_limit == 720


def test_compactor_response_max_tokens_clamped_to_token_ceiling() -> None:
    model = _BudgetModel(configured_max_tokens=320)

    budget = compactor_config.compactor_response_max_tokens(
        1200,
        model,
        response_tokens_floor=300,
        response_tokens_ratio=0.9,
        response_tokens_buffer=128,
        json_headroom_chars_floor=128,
        json_headroom_chars_ratio=0.30,
        json_headroom_chars_cap=1024,
        chars_per_token_estimate=2.8,
        token_headroom=128,
    )

    assert budget == 320


def test_semantic_compactor_budget_wrappers_match_config_helpers() -> None:
    model = _BudgetModel(configured_max_tokens=1000)
    compactor = SemanticCompactor(
        model=model,  # type: ignore[arg-type]
        response_tokens_floor=0,
        response_tokens_ratio=1.0,
        response_tokens_buffer=0,
        target_chars_ratio=0.82,
    )

    hard_limit = compactor._compactor_hard_limit_chars(1600, model)  # noqa: SLF001
    expected_hard_limit = compactor_config.compactor_hard_limit_chars(
        1600,
        model,
        response_tokens_ratio=1.0,
        response_tokens_buffer=0,
    )
    assert hard_limit == expected_hard_limit

    expected_budget = compactor_config.compactor_response_max_tokens(
        hard_limit,
        model,
        response_tokens_floor=0,
        response_tokens_ratio=1.0,
        response_tokens_buffer=0,
        json_headroom_chars_floor=128,
        json_headroom_chars_ratio=0.30,
        json_headroom_chars_cap=1024,
        chars_per_token_estimate=2.8,
        token_headroom=128,
    )
    assert compactor._compactor_response_max_tokens(hard_limit, model) == expected_budget  # noqa: SLF001
