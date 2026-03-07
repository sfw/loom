"""Model metadata formatting and normalization helpers."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from loom.models.base import ModelProvider


def configured_models(self) -> dict[str, object]:
    """Return configured model aliases from loaded config."""
    models = getattr(getattr(self, "_config", None), "models", None)
    if not isinstance(models, dict):
        return {}
    return {
        str(alias): cfg
        for alias, cfg in models.items()
    }


def normalize_provider_name(provider: object | None) -> str:
    """Normalize provider naming aliases for consistent display/matching."""
    value = str(provider or "").strip().lower()
    if value == "openai":
        return "openai_compatible"
    return value


def protocol_for_provider(provider: object | None) -> str:
    """Map provider type to user-facing protocol label."""
    normalized = normalize_provider_name(provider)
    if normalized == "anthropic":
        return "anthropic-messages"
    if normalized == "ollama":
        return "ollama-chat"
    if normalized == "openai_compatible":
        return "openai-chat-completions"
    if normalized:
        return "unknown"
    return "-"


def sanitize_endpoint_url(raw_url: object | None) -> str:
    """Render an endpoint URL with credentials/query/fragment stripped."""
    value = str(raw_url or "").strip()
    if not value:
        return "-"
    try:
        parsed = urlsplit(value)
    except Exception:
        return "(invalid-configured-url)"
    scheme = str(parsed.scheme or "").strip().lower()
    host = str(parsed.hostname or "").strip()
    if not scheme or not host:
        return "(invalid-configured-url)"
    try:
        port = parsed.port
    except ValueError:
        return "(invalid-configured-url)"
    safe_host = host
    if ":" in host and not host.startswith("["):
        safe_host = f"[{host}]"
    netloc = f"{safe_host}:{port}" if port is not None else safe_host
    safe = urlunsplit((scheme, netloc, parsed.path or "", "", ""))
    return safe or "(invalid-configured-url)"


def endpoint_for_config(self, provider: object | None, base_url: object | None) -> str:
    """Resolve and sanitize displayed endpoint from config values."""
    normalized = self._normalize_provider_name(provider)
    value = str(base_url or "").strip()
    if not value and normalized == "anthropic":
        value = "https://api.anthropic.com"
    if not value:
        return "-"
    return self._sanitize_endpoint_url(value)


def runtime_model_provider(self, model: ModelProvider | None) -> str:
    """Best-effort provider key for a runtime model instance."""
    if model is None:
        return ""
    cfg = getattr(model, "_config", None)
    normalized = self._normalize_provider_name(getattr(cfg, "provider", ""))
    if normalized:
        return normalized
    cls_name = type(model).__name__.lower()
    if "anthropic" in cls_name:
        return "anthropic"
    if "ollama" in cls_name:
        return "ollama"
    if "openai" in cls_name:
        return "openai_compatible"
    return ""


def runtime_model_id(model: ModelProvider | None) -> str:
    """Best-effort model identifier for a runtime model instance."""
    if model is None:
        return ""
    for attr in ("model", "_model"):
        value = getattr(model, attr, "")
        text = str(value or "").strip()
        if text:
            return text
    return ""


def runtime_model_roles(model: ModelProvider | None) -> list[str]:
    """Best-effort roles for a runtime model instance."""
    if model is None:
        return []
    value = getattr(model, "roles", [])
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def runtime_model_endpoint(self, model: ModelProvider | None) -> str:
    """Best-effort endpoint for a runtime model instance."""
    if model is None:
        return "-"
    cfg = getattr(model, "_config", None)
    base_url = getattr(cfg, "base_url", "")
    if not str(base_url or "").strip():
        base_url = getattr(model, "_base_url", "")
    return self._endpoint_for_config(self._runtime_model_provider(model), base_url)


def runtime_model_tier(model: ModelProvider | None) -> int | None:
    """Best-effort runtime tier for active model."""
    if model is None:
        return None
    value = getattr(model, "tier", None)
    if isinstance(value, int):
        return value if value > 0 else None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def infer_tier_from_config(
    self,
    provider: object | None,
    model_id: object | None,
) -> int | None:
    """Infer a tier label from provider + model name when not explicit."""
    normalized = self._normalize_provider_name(provider)
    value = str(model_id or "").strip().lower()
    if not value:
        return None
    if normalized == "openai_compatible":
        if any(token in value for token in ("70b", "72b", "m2.1", "large")):
            return 3
        if any(token in value for token in ("14b", "32b", "medium")):
            return 2
        return 1
    if normalized == "ollama":
        if any(token in value for token in ("70b", "72b", "large")):
            return 3
        if any(token in value for token in ("14b", "32b", "medium")):
            return 2
        return 1
    if normalized == "anthropic":
        if "opus" in value:
            return 3
        if "sonnet" in value:
            return 2
        if "haiku" in value:
            return 1
        return 2
    return None


def format_tier_label(explicit_tier: object | None, inferred_tier: int | None) -> str:
    """Render tier label, preferring explicit value then inferred."""
    if isinstance(explicit_tier, int) and explicit_tier > 0:
        return str(explicit_tier)
    try:
        parsed = int(explicit_tier)
    except (TypeError, ValueError):
        parsed = 0
    if parsed > 0:
        return str(parsed)
    if isinstance(inferred_tier, int) and inferred_tier > 0:
        return f"{inferred_tier} (inferred)"
    return "auto"


def format_temperature(value: object | None) -> str:
    """Render temperature with compact formatting."""
    if isinstance(value, (int, float)):
        return f"{float(value):g}"
    return "-"


def format_max_tokens(value: object | None) -> str:
    """Render max token limit when set."""
    if isinstance(value, int) and value > 0:
        return str(value)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return "-"
    return str(parsed) if parsed > 0 else "-"


def format_model_roles(roles: object | None) -> str:
    """Render roles in deterministic order."""
    if isinstance(roles, (list, tuple, set)):
        cleaned = {
            str(item).strip()
            for item in roles
            if str(item).strip()
        }
        if cleaned:
            return ", ".join(sorted(cleaned, key=str.casefold))
    return "-"


def format_capabilities(capabilities: object | None) -> str:
    """Render capabilities as compact yes/no flags."""
    if capabilities is None:
        return "-"
    fields = (
        "vision",
        "native_pdf",
        "thinking",
        "citations",
        "audio_input",
        "audio_output",
    )
    rendered: list[str] = []
    for field_name in fields:
        rendered.append(
            f"{field_name}={'yes' if bool(getattr(capabilities, field_name, False)) else 'no'}",
        )
    return ", ".join(rendered)
