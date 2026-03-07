"""Model helper parity tests for semantic compactor."""

from __future__ import annotations

from loom.engine.semantic_compactor import model as compactor_model
from loom.models.base import ModelConnectionError


def test_should_retry_compaction_error_detects_temperature_only_validation() -> None:
    error = ModelConnectionError(
        (
            "Model server returned HTTP 400: "
            '{"error":{"message":"invalid temperature: only 1 is allowed '
            'for this model"}}'
        ),
    )

    assert compactor_model.should_retry_compaction_error(error) is False


def test_should_retry_compaction_error_retries_generic_errors() -> None:
    error = RuntimeError("temporary upstream failure")

    assert compactor_model.should_retry_compaction_error(error) is True


def test_compactor_response_format_detects_known_provider_modules() -> None:
    OpenAIProviderLike = type(  # noqa: N806
        "OpenAIProviderLike",
        (),
        {"__module__": "loom.models.openai_provider"},
    )
    OllamaProviderLike = type(  # noqa: N806
        "OllamaProviderLike",
        (),
        {"__module__": "loom.models.ollama_provider"},
    )

    assert compactor_model.compactor_response_format(OpenAIProviderLike()) == {
        "type": "json_object",
    }
    assert compactor_model.compactor_response_format(OllamaProviderLike()) == {
        "type": "json",
    }
