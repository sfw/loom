"""Model selection and invocation helpers for semantic compactor."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from loom.models.base import ModelProvider
from loom.models.request_diagnostics import (
    collect_request_diagnostics,
    collect_response_diagnostics,
)
from loom.models.retry import ModelRetryPolicy, call_with_model_retry
from loom.models.router import ModelRouter


def select_model(
    *,
    explicit_model: ModelProvider | None,
    router: ModelRouter | None,
    role: str,
    tier: int,
    allow_role_fallback: bool,
    on_role_fallback: Callable[[str, str], None] | None = None,
) -> ModelProvider | None:
    """Resolve the configured compactor model, if available."""
    if explicit_model is not None:
        return explicit_model

    if router is None:
        return None

    if allow_role_fallback:
        raw_roles = (role, "verifier", "extractor", "executor")
        candidate_roles = tuple(dict.fromkeys(raw_roles))
    else:
        candidate_roles = (role,)

    for candidate_role in candidate_roles:
        try:
            model = router.select(tier=tier, role=candidate_role)
        except Exception:
            continue

        roles = getattr(model, "roles", None)
        if isinstance(roles, list) and candidate_role not in roles:
            continue
        if candidate_role != role and callable(on_role_fallback):
            on_role_fallback(role, candidate_role)
        return model
    return None


def is_temperature_one_only_error(value: object) -> bool:
    text = str(value or "").lower()
    return "invalid temperature" in text and "only 1 is allowed" in text


def configured_model_temperature(model: ModelProvider) -> float | None:
    value = getattr(model, "configured_temperature", None)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def should_retry_compaction_error(error: BaseException) -> bool:
    # Avoid exponential retries for deterministic provider validation errors.
    return not is_temperature_one_only_error(error)


def compactor_response_format(model: ModelProvider) -> dict[str, Any] | None:
    module_name = model.__class__.__module__.lower()
    if "openai_provider" in module_name:
        return {"type": "json_object"}
    if "ollama_provider" in module_name:
        return {"type": "json"}
    return None


async def invoke_compactor_model(
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
    response_format: dict[str, Any] | None,
    validation_attempt: int,
    emit_model_event: Callable[..., None],
    should_retry: Callable[[BaseException], bool],
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
        emit_model_event(
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
        emit_model_event(
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
            should_retry=should_retry,
            on_failure=_on_failure,
        )
        elapsed = (
            time.monotonic() - attempt_started_at if attempt_started_at > 0 else 0.0
        )
        response_diag = collect_response_diagnostics(response)
        emit_model_event(
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
