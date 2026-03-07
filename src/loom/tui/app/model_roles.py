"""Role-based helper-model selection and invocation."""

from __future__ import annotations

import logging

from loom.models.base import ModelProvider
from loom.models.retry import call_with_model_retry

logger = logging.getLogger(__name__)


def configured_model_temperature(model: ModelProvider | None) -> float | None:
    """Return selected model's configured temperature from config."""
    if model is None:
        return None
    value = getattr(model, "configured_temperature", None)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def configured_model_max_tokens(model: ModelProvider | None) -> int | None:
    """Return selected model's configured max_tokens from config."""
    if model is None:
        return None
    value = getattr(model, "configured_max_tokens", None)
    if isinstance(value, int) and value > 0:
        return value
    return None


def planning_response_max_tokens_limit(self) -> int | None:
    """Return planner-only max_tokens override from limits config."""
    value = getattr(
        getattr(self._config, "limits", None),
        "planning_response_max_tokens",
        None,
    )
    if isinstance(value, int) and value > 0:
        return value
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def has_configured_role_model(self, role: str) -> bool:
    """Return True when config declares at least one model for the role.

    If no model config is loaded (tests/ephemeral contexts), fall back to
    whether an active session model exists.
    """
    models = getattr(getattr(self, "_config", None), "models", None)
    if isinstance(models, dict) and models:
        return any(
            role in list(getattr(model_cfg, "roles", []) or [])
            for model_cfg in models.values()
        )
    return self._model is not None


def cowork_compactor_model(self) -> ModelProvider | None:
    """Best-effort compactor model for cowork context compaction."""
    config = getattr(self, "_config", None)
    models = getattr(config, "models", None)
    if not isinstance(models, dict) or not models:
        return None
    try:
        from loom.models.router import ModelRouter

        router = ModelRouter.from_config(config)
        return router.select(tier=1, role="compactor")
    except Exception as e:
        logger.debug("No cowork compactor role model configured: %s", e)
        return None


def cowork_memory_indexer_model(self) -> tuple[ModelProvider | None, str]:
    """Select memory-index helper model with role preference ordering."""
    if self._model is None:
        return None, ""
    config = getattr(self, "_config", None)
    models = getattr(config, "models", None)
    if not isinstance(models, dict) or not models:
        return self._model, "active"
    try:
        from loom.models.router import ModelRouter

        router = ModelRouter.from_config(config)
    except Exception as e:
        logger.debug("Failed building router for cowork memory indexer: %s", e)
        return self._model, "active"

    for role in ("compactor", "extractor"):
        try:
            return router.select(tier=1, role=role), role
        except Exception:
            continue
    return self._model, "active"


def select_helper_model_for_role(
    self,
    *,
    role: str,
    tier: int,
) -> tuple[ModelProvider | None, object | None]:
    """Select a helper model for a role and return (model, router_or_none)."""
    models = getattr(getattr(self, "_config", None), "models", None)
    if isinstance(models, dict) and models:
        from loom.models.router import ModelRouter

        try:
            router = ModelRouter.from_config(self._config)
        except Exception as e:
            logger.debug("Failed to build model router for role %s: %s", role, e)
            return None, None
        try:
            model = router.select(tier=tier, role=role)
        except Exception as e:
            logger.debug("No helper model configured for role %s: %s", role, e)
            return None, router
        return model, router
    return self._model, None


async def invoke_helper_role_completion(
    self,
    *,
    role: str,
    tier: int,
    prompt: str,
    max_tokens: int | None,
    temperature: float | None = None,
) -> tuple[object, str, float | None, int | None]:
    """Invoke a role-routed helper completion and close temporary routers."""
    model, router = self._select_helper_model_for_role(role=role, tier=tier)
    try:
        if model is None:
            raise RuntimeError(f"No model configured for role: {role}")
        resolved_temperature = (
            temperature
            if temperature is not None
            else self._configured_model_temperature(model)
        )
        planner_max_tokens = (
            self._planning_response_max_tokens_limit()
            if role == "planner"
            else None
        )
        resolved_max_tokens = (
            max_tokens
            if isinstance(max_tokens, int) and max_tokens > 0
            else planner_max_tokens
            if isinstance(planner_max_tokens, int) and planner_max_tokens > 0
            else self._configured_model_max_tokens(model)
        )
        response = await call_with_model_retry(
            lambda: model.complete(
                [{"role": "user", "content": prompt}],
                temperature=resolved_temperature,
                max_tokens=resolved_max_tokens,
            ),
            policy=self._model_retry_policy(),
        )
        return (
            response,
            str(getattr(model, "name", "") or ""),
            resolved_temperature,
            resolved_max_tokens,
        )
    finally:
        if router is not None:
            close = getattr(router, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception as e:
                    logger.debug("Failed closing helper role router: %s", e)
