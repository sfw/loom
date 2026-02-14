"""Model router: selects providers by role and tier.

Selection logic:
1. Filter models by role compatibility
2. Filter by minimum tier requirement
3. Select the lowest-tier model that meets requirements (prefer cheap)
4. Fall back to higher tier if lower tier is unavailable
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from loom.config import Config, ModelConfig
from loom.models.anthropic_provider import AnthropicProvider
from loom.models.base import ModelNotAvailableError, ModelProvider, ModelResponse
from loom.models.ollama_provider import OllamaProvider
from loom.models.openai_provider import OpenAICompatibleProvider


class ModelRouter:
    """Routes model requests to the appropriate provider based on role and tier."""

    def __init__(self, providers: dict[str, ModelProvider] | None = None):
        self._providers: dict[str, ModelProvider] = providers or {}
        self._role_map: dict[str, list[str]] = {}
        if self._providers:
            self._build_role_map()

    @classmethod
    def from_config(cls, config: Config) -> ModelRouter:
        """Create a router from configuration, instantiating all providers."""
        providers: dict[str, ModelProvider] = {}
        for name, model_config in config.models.items():
            provider = _create_provider(name, model_config)
            providers[name] = provider
        return cls(providers)

    def _build_role_map(self) -> None:
        """Build the role -> provider name mapping."""
        self._role_map.clear()
        for name, provider in self._providers.items():
            for role in provider.roles:
                if role not in self._role_map:
                    self._role_map[role] = []
                self._role_map[role].append(name)

    def add_provider(self, name: str, provider: ModelProvider) -> None:
        """Register a provider at runtime."""
        self._providers[name] = provider
        self._build_role_map()

    def select(self, tier: int = 1, role: str = "executor") -> ModelProvider:
        """Select the best model for the given role and minimum tier.

        Priority:
        1. Exact tier match with role capability
        2. Next tier up with role capability
        3. Any model with role capability (lowest tier)
        4. Raise ModelNotAvailableError
        """
        candidates = self._role_map.get(role, [])
        if not candidates:
            raise ModelNotAvailableError(f"No model configured for role: {role}")

        # Sort by distance from target tier, preferring models at or above the tier
        def sort_key(name: str) -> tuple[int, int]:
            provider_tier = self._providers[name].tier
            if provider_tier >= tier:
                return (0, provider_tier)  # At or above: prefer lowest
            return (1, -provider_tier)  # Below: deprioritize, prefer highest

        sorted_candidates = sorted(candidates, key=sort_key)
        return self._providers[sorted_candidates[0]]

    def list_providers(self) -> list[dict]:
        """List all registered providers with their metadata."""
        result = []
        for name, provider in self._providers.items():
            result.append({
                "name": name,
                "model": provider.name,
                "tier": provider.tier,
                "roles": provider.roles,
            })
        return result

    async def health(self) -> dict[str, bool]:
        """Check health of all configured models."""
        results = {}
        for name, provider in self._providers.items():
            try:
                results[name] = await provider.health_check()
            except Exception:
                results[name] = False
        return results

    async def close(self) -> None:
        """Close all provider HTTP clients."""
        for provider in self._providers.values():
            if hasattr(provider, "close"):
                await provider.close()


class ResponseValidator:
    """Validates model responses before processing.

    Catches malformed JSON, invalid tool calls, and hallucinated tool names.
    """

    def validate_tool_calls(
        self,
        response: ModelResponse,
        available_tools: list[dict],
    ) -> ValidationResult:
        """Validate that tool calls reference real tools with valid args."""
        if not response.tool_calls:
            return ValidationResult(valid=True)

        valid_names = {t["name"] for t in available_tools}
        issues = []

        for tc in response.tool_calls:
            if tc.name not in valid_names:
                issues.append(f"Unknown tool: {tc.name}")
            if not isinstance(tc.arguments, dict):
                issues.append(f"Invalid arguments for {tc.name}: expected dict")

        if issues:
            return ValidationResult(
                valid=False,
                error="; ".join(issues),
                suggestion="Use only the tools listed in AVAILABLE TOOLS.",
            )
        return ValidationResult(valid=True)

    def validate_json_response(
        self,
        response: ModelResponse,
        expected_keys: list[str] | None = None,
    ) -> ValidationResult:
        """Validate that the response text is valid JSON."""
        text = response.text.strip()
        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            return ValidationResult(
                valid=False,
                error=f"Invalid JSON: {e}",
                suggestion="Respond with ONLY valid JSON, no markdown or explanation.",
                parsed=None,
            )

        if expected_keys:
            missing = [k for k in expected_keys if k not in parsed]
            if missing:
                return ValidationResult(
                    valid=False,
                    error=f"Missing required keys: {missing}",
                    suggestion=f"Response must include: {expected_keys}",
                    parsed=parsed,
                )

        return ValidationResult(valid=True, parsed=parsed)


@dataclass
class ValidationResult:
    valid: bool
    error: str = ""
    suggestion: str = ""
    parsed: dict | list | None = None


def _create_provider(name: str, config: ModelConfig) -> ModelProvider:
    """Create a model provider from configuration."""
    if config.provider == "ollama":
        return OllamaProvider(config, provider_name=name)
    if config.provider in ("openai_compatible", "openai"):
        return OpenAICompatibleProvider(config, provider_name=name)
    if config.provider == "anthropic":
        return AnthropicProvider(
            name=name,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url or "https://api.anthropic.com",
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            tier=config.tier,
            roles=config.roles,
        )
    raise ValueError(f"Unknown provider type: {config.provider}")
