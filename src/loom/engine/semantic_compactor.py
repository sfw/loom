"""Semantic text compaction for model-facing context windows.

This module avoids naive character slicing by using an LLM to compress text
while preserving concrete details (facts, identifiers, filenames, paths,
commands, URLs, errors, and numeric values).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loom.models.base import ModelProvider
from loom.models.request_diagnostics import (
    collect_request_diagnostics,
    collect_response_diagnostics,
)
from loom.models.retry import ModelRetryPolicy, call_with_model_retry
from loom.models.router import ModelRouter

logger = logging.getLogger(__name__)


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
        async with self._cache_lock:
            cached = self._cache.get(key)
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

            inflight = self._inflight.get(key)
            if inflight is None:
                inflight = asyncio.get_running_loop().create_future()
                self._inflight[key] = inflight
                owner = True
            else:
                owner = False

        if not owner:
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
            async with self._cache_lock:
                self._cache[key] = compacted
                waiting = self._inflight.pop(key, None)
                if waiting is not None and not waiting.done():
                    waiting.set_result(compacted)

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
        """Resolve the configured compactor model, if available."""
        if self._model is not None:
            return self._model

        if self._router is None:
            return None

        if self._allow_role_fallback:
            raw_roles = (self._role, "verifier", "extractor", "executor")
            candidate_roles = tuple(dict.fromkeys(raw_roles))
        else:
            candidate_roles = (self._role,)
        for role in candidate_roles:
            try:
                model = self._router.select(tier=self._tier, role=role)
            except Exception:
                continue

            roles = getattr(model, "roles", None)
            if isinstance(roles, list) and role not in roles:
                continue
            if role != self._role:
                logger.debug(
                    "Semantic compactor falling back from role %s to %s",
                    self._role,
                    role,
                )
            return model
        return None

    async def _compact_with_model(
        self,
        *,
        model: ModelProvider,
        text: str,
        max_chars: int,
        label: str,
    ) -> tuple[str, int]:
        current = text
        total_retry_count = 0
        if len(current) > self._max_chunk_chars:
            current, retries_used = await self._chunked_compaction(
                model=model,
                text=current,
                max_chars=max_chars,
                label=label,
            )
            total_retry_count += retries_used

        rounds = 0
        while len(current) > max_chars and rounds < self._max_reduction_rounds:
            # Always aim at the caller's requested budget instead of inflating
            # per-round targets from the current text length.
            candidate, retries_used = await self._compact_once(
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

    async def _chunked_compaction(
        self,
        *,
        model: ModelProvider,
        text: str,
        max_chars: int,
        label: str,
    ) -> tuple[str, int]:
        chunks = self._split_text(text, self._max_chunk_chars)
        if not chunks:
            return text, 0

        total_retry_count = 0

        # Hierarchical pre-merge when there are too many chunks.
        while len(chunks) > self._max_chunks_per_round:
            merged: list[str] = []
            for idx in range(0, len(chunks), 3):
                block = "\n\n".join(chunks[idx:idx + 3])
                merged_text, retries_used = await self._compact_once(
                    model=model,
                    text=block,
                    max_chars=max(
                        self._min_compact_target_chars,
                        int(max_chars),
                    ),
                    label=f"{label} pre-merge",
                    strict=False,
                )
                total_retry_count += retries_used
                merged.append(merged_text or block)
            chunks = merged

        map_target = max(
            self._min_compact_target_chars,
            min(int(max_chars), 2_400),
        )
        mapped: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            if len(chunk) <= map_target:
                mapped.append(chunk)
                continue
            mapped_text, retries_used = await self._compact_once(
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

    async def _compact_once(
        self,
        *,
        model: ModelProvider,
        text: str,
        max_chars: int,
        label: str,
        strict: bool,
    ) -> tuple[str, int]:
        requested_max_chars = max(1, int(max_chars))
        hard_limit = self._compactor_hard_limit_chars(requested_max_chars, model)
        target_chars = self._compactor_target_chars(hard_limit)
        max_tokens = self._compactor_response_max_tokens(hard_limit, model)
        system = (
            "You are a semantic context compactor for LLM pipelines. "
            "Preserve concrete details and required instructions while removing "
            "redundancy. Never invent facts. "
            "Return strict JSON only."
        )
        configured_temperature = self._configured_model_temperature(model)
        response_format = self._compactor_response_format(model)
        source_text = str(text or "")
        current_target_chars = int(target_chars)
        current_max_tokens = max_tokens
        retry_reason = ""
        retry_hint = ""
        token_ceiling = self._compactor_token_ceiling_for_hard_limit(hard_limit, model)

        for validation_attempt in range(1, self._MAX_VALIDATION_RETRIES + 2):
            # Compactor length compliance must be strict on every attempt.
            attempt_strict = True
            user = self._build_compactor_user_prompt(
                label=label,
                source_text=source_text,
                target_chars=current_target_chars,
                hard_limit=hard_limit,
                retry_hint=retry_hint,
            )
            response = await self._invoke_compactor_model(
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
            parsed_text, parse_error = self._extract_compacted_text(response_text)
            recovered_partial_json = False
            if parse_error == "invalid_json" and response_finish_reason.lower() == "length":
                partial_text = self._extract_partial_compressed_text(response_text)
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
                validation_error = self._validate_compacted_text(
                    parsed_text,
                    limit=hard_limit,
                )
                target_delta_chars = len(parsed_text) - hard_limit

            if (
                validation_error == "output_exceeds_target"
                and parsed_text is not None
                and validation_attempt > self._MAX_VALIDATION_RETRIES
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
            self._emit_model_event(
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

            if validation_attempt > self._MAX_VALIDATION_RETRIES:
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
            ) = self._next_validation_retry(
                current_source_text=source_text,
                current_target_chars=current_target_chars,
                current_max_tokens=current_max_tokens,
                hard_limit=hard_limit,
                token_ceiling=token_ceiling,
                validation_error=validation_error,
                parsed_text=parsed_text,
                response_text=response_text,
                response_finish_reason=response_finish_reason,
            )

        raise RuntimeError("Semantic compactor exceeded validation retry budget")

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
        """Call compactor model and return response or terminal exception."""
        request_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload: dict[str, Any] = {
            "messages": request_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        request_diag = collect_request_diagnostics(
            messages=request_messages,
            payload=payload,
            origin="semantic_compactor.complete",
        )
        policy = ModelRetryPolicy()
        invocation_attempt = 0
        attempt_started_at = 0.0

        async def _invoke_model():
            nonlocal invocation_attempt, attempt_started_at
            invocation_attempt += 1
            attempt_started_at = time.monotonic()
            self._emit_model_event(
                model_name=model.name,
                phase="start",
                details={
                    "operation": "complete",
                    "origin": request_diag.origin,
                    "invocation_attempt": invocation_attempt,
                    "invocation_max_attempts": policy.max_attempts,
                    "compactor_label": label,
                    "compactor_requested_max_chars": requested_max_chars,
                    "compactor_target_chars": target_chars,
                    "compactor_hard_limit_chars": hard_limit,
                    "compactor_limit_chars": hard_limit,
                    "compactor_token_budget_chars": hard_limit,
                    "compactor_max_tokens": max_tokens,
                    "compactor_strict": bool(strict),
                    "compactor_validation_attempt": int(validation_attempt),
                    "compactor_retry_count": max(0, int(validation_attempt) - 1),
                    **request_diag.to_event_payload(),
                },
            )
            return await model.complete(
                request_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )

        def _on_failure(
            attempt: int,
            max_attempts: int,
            error: BaseException,
            remaining: int,
        ) -> None:
            elapsed = (
                time.monotonic() - attempt_started_at if attempt_started_at > 0 else 0.0
            )
            self._emit_model_event(
                model_name=model.name,
                phase="done",
                details={
                    "operation": "complete",
                    "origin": request_diag.origin,
                    "invocation_attempt": attempt,
                    "invocation_max_attempts": max_attempts,
                    "retry_queue_remaining": remaining,
                    "compactor_label": label,
                    "compactor_requested_max_chars": requested_max_chars,
                    "compactor_target_chars": target_chars,
                    "compactor_hard_limit_chars": hard_limit,
                    "compactor_limit_chars": hard_limit,
                    "compactor_token_budget_chars": hard_limit,
                    "compactor_max_tokens": max_tokens,
                    "compactor_strict": bool(strict),
                    "compactor_validation_attempt": int(validation_attempt),
                    "compactor_retry_count": max(0, int(validation_attempt) - 1),
                    "duration_seconds": round(elapsed, 6),
                    "compactor_response_chars": 0,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )

        try:
            response = await call_with_model_retry(
                _invoke_model,
                policy=policy,
                should_retry=self._should_retry_compaction_error,
                on_failure=_on_failure,
            )
            elapsed = (
                time.monotonic() - attempt_started_at if attempt_started_at > 0 else 0.0
            )
            response_diag = collect_response_diagnostics(response)
            self._emit_model_event(
                model_name=model.name,
                phase="done",
                details={
                    "operation": "complete",
                    "origin": request_diag.origin,
                    "invocation_attempt": invocation_attempt,
                    "invocation_max_attempts": policy.max_attempts,
                    "compactor_label": label,
                    "compactor_requested_max_chars": requested_max_chars,
                    "compactor_target_chars": target_chars,
                    "compactor_hard_limit_chars": hard_limit,
                    "compactor_limit_chars": hard_limit,
                    "compactor_token_budget_chars": hard_limit,
                    "compactor_max_tokens": max_tokens,
                    "compactor_strict": bool(strict),
                    "compactor_validation_attempt": int(validation_attempt),
                    "compactor_retry_count": max(0, int(validation_attempt) - 1),
                    "duration_seconds": round(elapsed, 6),
                    "compactor_response_chars": response_diag.response_chars,
                    **response_diag.to_event_payload(),
                },
            )
            return response
        except Exception as exc:  # pragma: no cover - covered by callers/tests
            return exc

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
        text = str(value or "").lower()
        return "invalid temperature" in text and "only 1 is allowed" in text

    @staticmethod
    def _configured_model_temperature(model: ModelProvider) -> float | None:
        value = getattr(model, "configured_temperature", None)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _compactor_response_max_tokens(
        self,
        limit_chars: int,
        model: ModelProvider,
    ) -> int | None:
        """Estimate max_tokens tightly around the validated hard character limit."""
        limit = max(1, int(limit_chars))
        token_ceiling = self._compactor_token_ceiling_for_hard_limit(limit, model)
        if (
            self._response_tokens_floor <= 0
            and self._response_tokens_ratio <= 0.0
            and self._response_tokens_buffer <= 0
        ):
            return token_ceiling

        estimated = int(limit * self._response_tokens_ratio) + self._response_tokens_buffer
        budget = max(self._response_tokens_floor, estimated)
        budget = max(1, budget)
        budget = min(budget, token_ceiling)
        return max(1, budget)

    def _compactor_json_headroom_chars(self, hard_limit_chars: int) -> int:
        limit = max(1, int(hard_limit_chars))
        estimated = int(round(limit * self._json_headroom_chars_ratio))
        estimated = max(self._json_headroom_chars_floor, estimated)
        return max(1, min(self._json_headroom_chars_cap, estimated))

    def _compactor_token_ceiling_for_hard_limit(
        self,
        hard_limit_chars: int,
        model: ModelProvider,
    ) -> int:
        limit = max(1, int(hard_limit_chars))
        char_budget = limit + self._compactor_json_headroom_chars(limit)
        token_budget = int(round(char_budget / self._chars_per_token_estimate))
        token_budget += self._token_headroom
        token_budget = max(1, token_budget)
        configured_max = getattr(model, "configured_max_tokens", None)
        if isinstance(configured_max, int) and configured_max > 0:
            token_budget = min(token_budget, configured_max)
        return max(1, token_budget)

    def _compactor_hard_limit_chars(
        self,
        requested_chars: int,
        model: ModelProvider,
    ) -> int:
        """Bound requested output chars by model token capacity when known."""
        limit = max(1, int(requested_chars))
        configured_max = getattr(model, "configured_max_tokens", None)
        if (
            not isinstance(configured_max, int)
            or configured_max <= 0
            or self._response_tokens_ratio <= 0.0
        ):
            return limit

        available_tokens = configured_max - self._response_tokens_buffer
        if available_tokens <= 0:
            return 1

        ceiling = int(available_tokens / self._response_tokens_ratio)
        if ceiling <= 0:
            return 1
        return min(limit, ceiling)

    def _compactor_target_chars(self, hard_limit: int) -> int:
        """Choose a conservative target below hard limit for JSON envelope room."""
        limit = max(1, int(hard_limit))
        target = int(round(limit * self._target_chars_ratio))
        return max(1, min(limit, target))

    @classmethod
    def _should_retry_compaction_error(cls, error: BaseException) -> bool:
        # Avoid exponential retries for deterministic provider validation errors.
        return not cls._is_temperature_one_only_error(error)

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
        module_name = model.__class__.__module__.lower()
        if "openai_provider" in module_name:
            return {"type": "json_object"}
        if "ollama_provider" in module_name:
            return {"type": "json"}
        return None

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
            target_floor = min(self._min_compact_target_chars, hard_limit)
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

    @staticmethod
    def _hard_cap_text(text: str, *, max_chars: int) -> str:
        value = str(text or "")
        if max_chars <= 0:
            return ""
        if len(value) <= max_chars:
            return value
        if max_chars <= 40:
            return value[:max_chars]

        marker = "...[truncated]..."
        remaining = max_chars - len(marker)
        if remaining <= 0:
            return value[:max_chars]

        head = max(16, int(remaining * 0.65))
        tail = max(8, remaining - head)
        if head + tail > remaining:
            tail = max(0, remaining - head)
        compacted = f"{value[:head]}{marker}{value[-tail:] if tail else ''}"
        return compacted[:max_chars]

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        if not text.startswith("```"):
            return text
        lines = text.split("\n")
        content = "\n".join(lines[1:])
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()

    @classmethod
    def _extract_compacted_text(cls, raw_text: str) -> tuple[str | None, str]:
        payload_text = cls._strip_markdown_fences(str(raw_text or "").strip())
        if not payload_text:
            return None, "empty_response"

        parsed: Any
        try:
            parsed = json.loads(payload_text)
        except json.JSONDecodeError:
            parsed = cls._extract_first_json_object(payload_text)
            if parsed is None:
                return None, "invalid_json"

        if not isinstance(parsed, dict):
            return None, "json_not_object"
        if "compressed_text" not in parsed:
            return None, "missing_compressed_text"

        compacted = parsed.get("compressed_text")
        if not isinstance(compacted, str):
            return None, "compressed_text_not_string"
        return compacted.strip(), ""

    @staticmethod
    def _extract_first_json_object(text: str) -> dict[str, Any] | None:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            start = match.start()
            try:
                candidate, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                return candidate
        return None

    @classmethod
    def _extract_partial_compressed_text(cls, raw_text: str) -> str | None:
        payload_text = cls._strip_markdown_fences(str(raw_text or "").strip())
        if not payload_text:
            return None

        key_match = re.search(r'"compressed_text"\s*:\s*"', payload_text)
        if key_match is None:
            return None

        idx = key_match.end()
        chars: list[str] = []
        escaping = False
        hex_chars = "0123456789abcdefABCDEF"

        while idx < len(payload_text):
            char = payload_text[idx]
            if escaping:
                if char == "n":
                    chars.append("\n")
                elif char == "r":
                    chars.append("\r")
                elif char == "t":
                    chars.append("\t")
                elif char == "b":
                    chars.append("\b")
                elif char == "f":
                    chars.append("\f")
                elif char in {'"', "\\", "/"}:
                    chars.append(char)
                elif char == "u":
                    digits = payload_text[idx + 1:idx + 5]
                    if len(digits) < 4 or not all(c in hex_chars for c in digits):
                        break
                    chars.append(chr(int(digits, 16)))
                    idx += 4
                else:
                    chars.append(char)
                escaping = False
                idx += 1
                continue

            if char == "\\":
                escaping = True
                idx += 1
                continue
            if char == '"':
                break
            chars.append(char)
            idx += 1

        recovered = "".join(chars).strip()
        return recovered or None

    @classmethod
    def _validate_compacted_text(cls, text: str, *, limit: int) -> str:
        content = str(text or "").strip()
        if not content:
            return "empty_compacted_text"
        if len(content) > limit:
            return "output_exceeds_target"

        lower_head = content[:220].lower()
        for prefix in cls._META_COMMENTARY_PREFIXES:
            if lower_head.startswith(prefix):
                return "meta_commentary_prefix"
        for phrase in cls._META_COMMENTARY_PHRASES:
            if phrase in lower_head:
                return "meta_commentary_phrase"
        return ""

    @staticmethod
    def _split_text(text: str, chunk_chars: int) -> list[str]:
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

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Lossless fallback compaction: normalize repeated whitespace."""
        if not text:
            return text
        lines = [line.rstrip() for line in text.splitlines()]
        collapsed: list[str] = []
        blank = False
        for line in lines:
            if line:
                collapsed.append(" ".join(line.split()))
                blank = False
            elif not blank:
                collapsed.append("")
                blank = True
        return "\n".join(collapsed).strip()
