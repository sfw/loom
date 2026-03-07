"""Semantic text compaction for model-facing context windows.

This module avoids naive character slicing by using an LLM to compress text
while preserving concrete details (facts, identifiers, filenames, paths,
commands, URLs, errors, and numeric values).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loom.models.base import ModelProvider
from loom.models.router import ModelRouter

from . import cache as compactor_cache
from . import config as compactor_config
from . import model as compactor_model
from . import parse as compactor_parse
from . import pipeline as compactor_pipeline

logger = logging.getLogger(__name__)

__all__ = ["SemanticCompactor"]


@dataclass(frozen=True)
class _CacheKey:
    digest: str
    max_chars: int


class SemanticCompactor:
    """LLM-backed semantic compactor with chunked map/reduce behavior."""

    _MAX_CHUNK_CHARS = 9_000
    _MAX_CHUNKS_PER_ROUND = 12
    _MAX_REDUCTION_ROUNDS = 4
    _MIN_COMPACT_TARGET_CHARS = 140
    _RESPONSE_TOKENS_FLOOR = 256
    _RESPONSE_TOKENS_RATIO = 0.75
    _RESPONSE_TOKENS_BUFFER = 256
    _JSON_HEADROOM_CHARS_FLOOR = 48
    _JSON_HEADROOM_CHARS_RATIO = 0.08
    _JSON_HEADROOM_CHARS_CAP = 320
    _CHARS_PER_TOKEN_ESTIMATE = 3.6
    _TOKEN_HEADROOM = 24
    _TARGET_CHARS_RATIO = 0.75
    _MAX_VALIDATION_RETRIES = 1
    _NEAR_BUDGET_RATIO = 1.20
    _TINY_TARGET_SKIP_THRESHOLD = 220
    _META_COMMENTARY_PREFIXES = (
        "the user wants",
        "user wants",
        "i need to",
        "i will",
        "let me",
        "we need to",
        "this text",
    )
    _META_COMMENTARY_PHRASES = (
        "the user wants me",
        "i need to compact",
        "let me analyze",
        "character target:",
    )

    def __init__(
        self,
        model_router: ModelRouter | None = None,
        *,
        model: ModelProvider | None = None,
        model_event_hook: Callable[[dict[str, Any]], None] | None = None,
        role: str = "compactor",
        tier: int = 1,
        allow_role_fallback: bool = False,
        max_chunk_chars: int | None = None,
        max_chunks_per_round: int | None = None,
        max_reduction_rounds: int | None = None,
        min_compact_target_chars: int | None = None,
        response_tokens_floor: int | None = None,
        response_tokens_ratio: float | None = None,
        response_tokens_buffer: int | None = None,
        json_headroom_chars_floor: int | None = None,
        json_headroom_chars_ratio: float | None = None,
        json_headroom_chars_cap: int | None = None,
        chars_per_token_estimate: float | None = None,
        token_headroom: int | None = None,
        target_chars_ratio: float | None = None,
    ):
        self._router = model_router
        self._model = model
        self._model_event_hook = model_event_hook
        self._role = role
        self._tier = tier
        self._allow_role_fallback = bool(allow_role_fallback)
        self._max_chunk_chars = max(
            300,
            int(max_chunk_chars or self._MAX_CHUNK_CHARS),
        )
        self._max_chunks_per_round = max(
            1,
            int(max_chunks_per_round or self._MAX_CHUNKS_PER_ROUND),
        )
        self._max_reduction_rounds = max(
            1,
            int(max_reduction_rounds or self._MAX_REDUCTION_ROUNDS),
        )
        self._min_compact_target_chars = max(
            20,
            int(min_compact_target_chars or self._MIN_COMPACT_TARGET_CHARS),
        )
        self._response_tokens_floor = max(
            0,
            int(
                self._RESPONSE_TOKENS_FLOOR
                if response_tokens_floor is None
                else response_tokens_floor
            ),
        )
        self._response_tokens_ratio = max(
            0.0,
            float(
                self._RESPONSE_TOKENS_RATIO
                if response_tokens_ratio is None
                else response_tokens_ratio
            ),
        )
        self._response_tokens_buffer = max(
            0,
            int(
                self._RESPONSE_TOKENS_BUFFER
                if response_tokens_buffer is None
                else response_tokens_buffer
            ),
        )
        self._json_headroom_chars_floor = max(
            0,
            int(
                self._JSON_HEADROOM_CHARS_FLOOR
                if json_headroom_chars_floor is None
                else json_headroom_chars_floor
            ),
        )
        self._json_headroom_chars_ratio = max(
            0.0,
            float(
                self._JSON_HEADROOM_CHARS_RATIO
                if json_headroom_chars_ratio is None
                else json_headroom_chars_ratio
            ),
        )
        self._json_headroom_chars_cap = max(
            self._json_headroom_chars_floor,
            int(
                self._JSON_HEADROOM_CHARS_CAP
                if json_headroom_chars_cap is None
                else json_headroom_chars_cap
            ),
        )
        self._chars_per_token_estimate = max(
            0.1,
            float(
                self._CHARS_PER_TOKEN_ESTIMATE
                if chars_per_token_estimate is None
                else chars_per_token_estimate
            ),
        )
        self._token_headroom = max(
            0,
            int(
                self._TOKEN_HEADROOM
                if token_headroom is None
                else token_headroom
            ),
        )
        self._target_chars_ratio = min(
            1.0,
            max(
                0.01,
                float(
                    self._TARGET_CHARS_RATIO
                    if target_chars_ratio is None
                    else target_chars_ratio
                ),
            ),
        )
        self._cache: dict[_CacheKey, str] = {}
        self._inflight: dict[_CacheKey, asyncio.Future[str]] = {}
        self._cache_lock = asyncio.Lock()

    async def compact(self, text: str, *, max_chars: int, label: str = "context") -> str:
        """Semantically compact text to approximately ``max_chars`` characters.

        This never performs hard character slicing. If no compactor model is
        available, it falls back to whitespace-only compaction (lossless).
        """
        value = str(text or "").strip()
        if max_chars <= 0:
            return ""
        if not value or len(value) <= max_chars:
            return value

        key = _CacheKey(
            digest=hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest(),
            max_chars=max_chars,
        )
        cached, inflight, owner = await compactor_cache.lookup_cache_or_inflight(
            cache=self._cache,
            inflight=self._inflight,
            lock=self._cache_lock,
            key=key,
        )
        if cached is not None:
            self._emit_model_event(
                model_name="semantic_compactor",
                phase="cache",
                details={
                    "operation": "compact",
                    "origin": "semantic_compactor.complete",
                    "compactor_label": label,
                    "compactor_target_chars": int(max_chars),
                    "compactor_input_chars": len(value),
                    "compactor_output_chars": len(cached),
                    "compactor_cache_hit": True,
                    "compactor_inflight_join": False,
                    "compactor_bypass_reason": "cache_hit",
                    "compactor_retry_count": 0,
                },
            )
            return cached

        if not owner:
            assert inflight is not None
            compacted = await inflight
            self._emit_model_event(
                model_name="semantic_compactor",
                phase="cache",
                details={
                    "operation": "compact",
                    "origin": "semantic_compactor.complete",
                    "compactor_label": label,
                    "compactor_target_chars": int(max_chars),
                    "compactor_input_chars": len(value),
                    "compactor_output_chars": len(compacted),
                    "compactor_cache_hit": False,
                    "compactor_inflight_join": True,
                    "compactor_bypass_reason": "inflight_join",
                    "compactor_retry_count": 0,
                },
            )
            return compacted

        compacted = value
        model_name = "semantic_compactor"
        bypass_reason = ""
        retry_count = 0
        hard_cap_applied = False
        output_raw_chars = len(compacted)
        try:
            fast_path = self._fast_path_compaction(value, max_chars=max_chars)
            if fast_path is not None:
                compacted = fast_path
                bypass_reason = "fast_path"
            else:
                model = self._select_model()
                if model is None:
                    compacted = self._normalize_whitespace(value)
                    bypass_reason = "model_unavailable"
                else:
                    model_name = model.name
                    try:
                        compacted, retry_count = await self._compact_with_model(
                            model=model,
                            text=value,
                            max_chars=max_chars,
                            label=label,
                        )
                    except Exception as exc:
                        logger.warning("Semantic compaction failed for %s: %s", label, exc)
                        compacted = self._normalize_whitespace(value)
                        bypass_reason = "model_error"
        finally:
            output_raw_chars = len(compacted)
            await compactor_cache.store_cache_and_release_inflight(
                cache=self._cache,
                inflight=self._inflight,
                lock=self._cache_lock,
                key=key,
                value=compacted,
            )

        self._emit_model_event(
            model_name=model_name,
            phase="compact_result",
            details={
                "operation": "compact",
                "origin": "semantic_compactor.complete",
                "compactor_label": label,
                "compactor_target_chars": int(max_chars),
                "compactor_input_chars": len(value),
                "compactor_output_chars": len(compacted),
                "compactor_output_raw_chars": int(output_raw_chars),
                "compactor_hard_cap_applied": bool(hard_cap_applied),
                "compactor_hard_cap_limit_chars": int(max_chars),
                "compactor_cache_hit": False,
                "compactor_inflight_join": False,
                "compactor_bypass_reason": bypass_reason,
                "compactor_retry_count": int(retry_count),
            },
        )
        return compacted

    def _select_model(self) -> ModelProvider | None:
        return compactor_model.select_model(
            explicit_model=self._model,
            router=self._router,
            role=self._role,
            tier=self._tier,
            allow_role_fallback=self._allow_role_fallback,
            on_role_fallback=lambda expected, selected: logger.debug(
                "Semantic compactor falling back from role %s to %s",
                expected,
                selected,
            ),
        )

    async def _compact_with_model(
        self,
        *,
        model: ModelProvider,
        text: str,
        max_chars: int,
        label: str,
    ) -> tuple[str, int]:
        return await compactor_pipeline.compact_with_model(
            model=model,
            text=text,
            max_chars=max_chars,
            label=label,
            max_chunk_chars=self._max_chunk_chars,
            max_reduction_rounds=self._max_reduction_rounds,
            compact_once=self._compact_once,
            chunked_compaction=self._chunked_compaction,
        )

    async def _chunked_compaction(
        self,
        *,
        model: ModelProvider,
        text: str,
        max_chars: int,
        label: str,
    ) -> tuple[str, int]:
        return await compactor_pipeline.chunked_compaction(
            model=model,
            text=text,
            max_chars=max_chars,
            label=label,
            max_chunk_chars=self._max_chunk_chars,
            max_chunks_per_round=self._max_chunks_per_round,
            min_compact_target_chars=self._min_compact_target_chars,
            split_text=self._split_text,
            compact_once=self._compact_once,
        )

    async def _compact_once(
        self,
        *,
        model: ModelProvider,
        text: str,
        max_chars: int,
        label: str,
        strict: bool,
    ) -> tuple[str, int]:
        del strict
        return await compactor_pipeline.compact_once(
            model=model,
            text=text,
            max_chars=max_chars,
            label=label,
            max_validation_retries=self._MAX_VALIDATION_RETRIES,
            min_compact_target_chars=self._min_compact_target_chars,
            configured_temperature=self._configured_model_temperature(model),
            response_format=self._compactor_response_format(model),
            compactor_hard_limit_chars=self._compactor_hard_limit_chars,
            compactor_target_chars=self._compactor_target_chars,
            compactor_response_max_tokens=self._compactor_response_max_tokens,
            compactor_token_ceiling_for_hard_limit=self._compactor_token_ceiling_for_hard_limit,
            invoke_compactor_model=self._invoke_compactor_model,
            extract_compacted_text=self._extract_compacted_text,
            extract_partial_compressed_text=self._extract_partial_compressed_text,
            validate_compacted_text=lambda value, limit: self._validate_compacted_text(
                value,
                limit=limit,
            ),
            emit_model_event=self._emit_model_event,
            logger=logger,
        )

    async def _invoke_compactor_model(
        self,
        *,
        model: ModelProvider,
        system: str,
        user: str,
        requested_max_chars: int,
        target_chars: int,
        hard_limit: int,
        max_tokens: int | None,
        temperature: float | None,
        label: str,
        strict: bool,
        response_format: dict[str, Any] | None = None,
        validation_attempt: int = 1,
    ):
        return await compactor_model.invoke_compactor_model(
            model=model,
            system=system,
            user=user,
            requested_max_chars=requested_max_chars,
            target_chars=target_chars,
            hard_limit=hard_limit,
            max_tokens=max_tokens,
            temperature=temperature,
            label=label,
            strict=strict,
            response_format=response_format,
            validation_attempt=validation_attempt,
            emit_model_event=self._emit_model_event,
            should_retry=self._should_retry_compaction_error,
        )

    def _emit_model_event(
        self,
        *,
        model_name: str,
        phase: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        if not callable(self._model_event_hook):
            return
        payload: dict[str, Any] = {"model": model_name, "phase": phase}
        if isinstance(details, dict) and details:
            payload.update(details)
        try:
            self._model_event_hook(payload)
        except Exception as exc:  # pragma: no cover - defensive only
            logger.debug("Semantic compactor model event hook failed: %s", exc)

    @staticmethod
    def _is_temperature_one_only_error(value: object) -> bool:
        return compactor_model.is_temperature_one_only_error(value)

    @staticmethod
    def _configured_model_temperature(model: ModelProvider) -> float | None:
        return compactor_model.configured_model_temperature(model)

    def _compactor_response_max_tokens(
        self,
        limit_chars: int,
        model: ModelProvider,
    ) -> int | None:
        return compactor_config.compactor_response_max_tokens(
            limit_chars,
            model,
            response_tokens_floor=self._response_tokens_floor,
            response_tokens_ratio=self._response_tokens_ratio,
            response_tokens_buffer=self._response_tokens_buffer,
            json_headroom_chars_floor=self._json_headroom_chars_floor,
            json_headroom_chars_ratio=self._json_headroom_chars_ratio,
            json_headroom_chars_cap=self._json_headroom_chars_cap,
            chars_per_token_estimate=self._chars_per_token_estimate,
            token_headroom=self._token_headroom,
        )

    def _compactor_json_headroom_chars(self, hard_limit_chars: int) -> int:
        return compactor_config.compactor_json_headroom_chars(
            hard_limit_chars,
            json_headroom_chars_floor=self._json_headroom_chars_floor,
            json_headroom_chars_ratio=self._json_headroom_chars_ratio,
            json_headroom_chars_cap=self._json_headroom_chars_cap,
        )

    def _compactor_token_ceiling_for_hard_limit(
        self,
        hard_limit_chars: int,
        model: ModelProvider,
    ) -> int:
        return compactor_config.compactor_token_ceiling_for_hard_limit(
            hard_limit_chars,
            model,
            json_headroom_chars_floor=self._json_headroom_chars_floor,
            json_headroom_chars_ratio=self._json_headroom_chars_ratio,
            json_headroom_chars_cap=self._json_headroom_chars_cap,
            chars_per_token_estimate=self._chars_per_token_estimate,
            token_headroom=self._token_headroom,
        )

    def _compactor_hard_limit_chars(
        self,
        requested_chars: int,
        model: ModelProvider,
    ) -> int:
        return compactor_config.compactor_hard_limit_chars(
            requested_chars,
            model,
            response_tokens_ratio=self._response_tokens_ratio,
            response_tokens_buffer=self._response_tokens_buffer,
        )

    def _compactor_target_chars(self, hard_limit: int) -> int:
        return compactor_config.compactor_target_chars(
            hard_limit,
            target_chars_ratio=self._target_chars_ratio,
        )

    @classmethod
    def _should_retry_compaction_error(cls, error: BaseException) -> bool:
        return compactor_model.should_retry_compaction_error(error)

    def _fast_path_compaction(self, text: str, *, max_chars: int) -> str | None:
        normalized = self._normalize_whitespace(text)
        near_budget_limit = int(max_chars * self._NEAR_BUDGET_RATIO)
        if (
            len(text) <= near_budget_limit
            and len(normalized) <= max_chars
        ):
            return normalized
        if (
            max_chars <= self._TINY_TARGET_SKIP_THRESHOLD
            and len(normalized) <= max_chars
        ):
            return normalized
        return None

    def _compactor_response_format(
        self,
        model: ModelProvider,
    ) -> dict[str, Any] | None:
        return compactor_model.compactor_response_format(model)

    @classmethod
    def _build_compactor_user_prompt(
        cls,
        *,
        label: str,
        source_text: str,
        target_chars: int,
        hard_limit: int,
        retry_hint: str = "",
    ) -> str:
        del cls
        return compactor_pipeline.build_compactor_user_prompt(
            label=label,
            source_text=source_text,
            target_chars=target_chars,
            hard_limit=hard_limit,
            retry_hint=retry_hint,
        )

    def _next_validation_retry(
        self,
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
    ) -> tuple[str, int, int | None, str, str]:
        return compactor_pipeline.next_validation_retry(
            current_source_text=current_source_text,
            current_target_chars=current_target_chars,
            current_max_tokens=current_max_tokens,
            hard_limit=hard_limit,
            token_ceiling=token_ceiling,
            validation_error=validation_error,
            parsed_text=parsed_text,
            response_text=response_text,
            response_finish_reason=response_finish_reason,
            min_compact_target_chars=self._min_compact_target_chars,
        )

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        return compactor_parse.strip_markdown_fences(text)

    @classmethod
    def _extract_compacted_text(cls, raw_text: str) -> tuple[str | None, str]:
        return compactor_parse.extract_compacted_text(raw_text)

    @staticmethod
    def _extract_first_json_object(text: str) -> dict[str, Any] | None:
        return compactor_parse.extract_first_json_object(text)

    @classmethod
    def _extract_partial_compressed_text(cls, raw_text: str) -> str | None:
        return compactor_parse.extract_partial_compressed_text(raw_text)

    @classmethod
    def _validate_compacted_text(cls, text: str, *, limit: int) -> str:
        return compactor_parse.validate_compacted_text(
            text,
            limit=limit,
            meta_commentary_prefixes=cls._META_COMMENTARY_PREFIXES,
            meta_commentary_phrases=cls._META_COMMENTARY_PHRASES,
        )

    @staticmethod
    def _split_text(text: str, chunk_chars: int) -> list[str]:
        return compactor_pipeline.split_text(text, chunk_chars)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return compactor_parse.normalize_whitespace(text)
