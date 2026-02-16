"""Tests for multimodal content integration across the pipeline.

Covers: ToolResult serialization/deserialization round-trips,
OpenAI provider message building, and runner tool event emission.
"""

from __future__ import annotations

import json

from loom.content import (
    DocumentBlock,
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    serialize_block,
)
from loom.tools.registry import ToolResult

# ---------------------------------------------------------------------------
# ToolResult round-trip tests
# ---------------------------------------------------------------------------

class TestToolResultRoundTrip:
    """ToolResult.to_json() → ToolResult.from_json() preserves content_blocks."""

    def test_round_trip_no_blocks(self):
        original = ToolResult.ok("hello world", files_changed=["a.py"])
        json_str = original.to_json()
        restored = ToolResult.from_json(json_str)

        assert restored.success is True
        assert restored.output == "hello world"
        assert restored.files_changed == ["a.py"]
        assert restored.content_blocks is None

    def test_round_trip_with_image_block(self):
        block = ImageBlock(
            source_path="/tmp/img.png",
            media_type="image/png",
            width=800,
            height=600,
            size_bytes=12345,
            text_fallback="Image: img.png (800x600)",
        )
        original = ToolResult.multimodal("Image: img.png", [block])
        json_str = original.to_json()
        restored = ToolResult.from_json(json_str)

        assert restored.success is True
        assert restored.content_blocks is not None
        assert len(restored.content_blocks) == 1
        rb = restored.content_blocks[0]
        assert isinstance(rb, ImageBlock)
        assert rb.source_path == "/tmp/img.png"
        assert rb.width == 800
        assert rb.height == 600
        assert rb.media_type == "image/png"
        assert rb.text_fallback == "Image: img.png (800x600)"

    def test_round_trip_with_document_block(self):
        block = DocumentBlock(
            source_path="/tmp/doc.pdf",
            page_count=10,
            size_bytes=50000,
            page_range=(0, 5),
            text_fallback="PDF: doc.pdf (10 pages)",
        )
        original = ToolResult.multimodal("PDF: doc.pdf", [block])
        json_str = original.to_json()
        restored = ToolResult.from_json(json_str)

        assert restored.content_blocks is not None
        assert len(restored.content_blocks) == 1
        rb = restored.content_blocks[0]
        assert isinstance(rb, DocumentBlock)
        assert rb.source_path == "/tmp/doc.pdf"
        assert rb.page_count == 10
        assert rb.page_range == (0, 5)

    def test_round_trip_with_thinking_block(self):
        block = ThinkingBlock(thinking="Let me think...", signature="sig123")
        original = ToolResult.multimodal("output", [block])
        json_str = original.to_json()
        restored = ToolResult.from_json(json_str)

        assert restored.content_blocks is not None
        rb = restored.content_blocks[0]
        assert isinstance(rb, ThinkingBlock)
        assert rb.thinking == "Let me think..."
        assert rb.signature == "sig123"

    def test_round_trip_multiple_blocks(self):
        blocks = [
            TextBlock(text="Some text"),
            ImageBlock(source_path="/tmp/a.png", width=100, height=100),
            DocumentBlock(source_path="/tmp/b.pdf", page_count=3),
        ]
        original = ToolResult.multimodal("mixed content", blocks)
        json_str = original.to_json()
        restored = ToolResult.from_json(json_str)

        assert restored.content_blocks is not None
        assert len(restored.content_blocks) == 3
        assert isinstance(restored.content_blocks[0], TextBlock)
        assert isinstance(restored.content_blocks[1], ImageBlock)
        assert isinstance(restored.content_blocks[2], DocumentBlock)

    def test_round_trip_failed_result(self):
        original = ToolResult.fail("something went wrong")
        json_str = original.to_json()
        restored = ToolResult.from_json(json_str)

        assert restored.success is False
        assert restored.error == "something went wrong"
        assert restored.content_blocks is None

    def test_from_json_invalid(self):
        restored = ToolResult.from_json("not json")
        assert restored.success is False
        assert restored.error == "Invalid JSON"

    def test_from_json_plain_text(self):
        """Plain string content (not JSON) should be handled gracefully."""
        restored = ToolResult.from_json('"just a string"')
        # json.loads returns a string, not a dict — from_json should handle
        assert restored.success is False


# ---------------------------------------------------------------------------
# OpenAI provider message building tests
# ---------------------------------------------------------------------------

class TestOpenAIMessageBuilding:
    """Test _build_openai_messages and _extract_multimodal_parts."""

    def _make_provider(self, vision: bool = True):
        """Create a minimal OpenAI provider for testing."""
        from unittest.mock import MagicMock

        from loom.config import ModelCapabilities, ModelConfig

        config = MagicMock(spec=ModelConfig)
        config.model = "test-model"
        config.base_url = "http://localhost:8080"
        config.max_tokens = 4096
        config.temperature = 0.7
        config.roles = ["executor"]
        config.resolved_capabilities = ModelCapabilities(vision=vision)

        # Patch httpx client creation
        import httpx

        from loom.models.openai_provider import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
        provider._config = config
        provider._model = config.model
        provider._max_tokens = config.max_tokens
        provider._temperature = config.temperature
        provider._provider_name = "test"
        provider._roles = ["executor"]
        provider._tier = 1
        provider._capabilities = config.resolved_capabilities
        provider._client = MagicMock(spec=httpx.AsyncClient)
        return provider

    def test_passthrough_without_vision(self):
        provider = self._make_provider(vision=False)
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "tool", "tool_call_id": "1", "content": '{"success": true, "output": "ok"}'},
        ]
        result = provider._build_openai_messages(messages)
        assert result == messages  # unchanged

    def test_tool_result_text_extracted(self):
        provider = self._make_provider(vision=True)
        messages = [
            {"role": "tool", "tool_call_id": "1", "content": json.dumps({
                "success": True,
                "output": "File contents here",
            })},
        ]
        result = provider._build_openai_messages(messages)
        assert result[0]["content"] == "File contents here"

    def test_multimodal_parts_injected(self):
        provider = self._make_provider(vision=True)
        block = serialize_block(ImageBlock(
            source_path="/tmp/test.png",
            media_type="image/png",
            width=100,
            height=100,
            text_fallback="Image fallback",
        ))
        messages = [
            {"role": "tool", "tool_call_id": "1", "content": json.dumps({
                "success": True,
                "output": "Image read",
                "content_blocks": [block],
            })},
        ]
        result = provider._build_openai_messages(messages)
        # Tool result should be text, followed by user message with image parts
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "Image read"
        # The image will fail to encode (file doesn't exist) so fallback text is used
        assert len(result) == 2
        assert result[1]["role"] == "user"

    def test_extract_multimodal_parts_no_blocks(self):
        provider = self._make_provider(vision=True)
        result = provider._extract_multimodal_parts('{"success": true, "output": "ok"}')
        assert result is None

    def test_extract_multimodal_parts_invalid_json(self):
        provider = self._make_provider(vision=True)
        result = provider._extract_multimodal_parts("not json")
        assert result is None

    def test_extract_multimodal_parts_text_block(self):
        provider = self._make_provider(vision=True)
        content = json.dumps({
            "success": True,
            "output": "ok",
            "content_blocks": [{"type": "text", "text": "hello"}],
        })
        parts = provider._extract_multimodal_parts(content)
        assert parts is not None
        assert len(parts) == 1
        assert parts[0]["type"] == "text"
        assert parts[0]["text"] == "hello"


# ---------------------------------------------------------------------------
# Runner tool event emission tests
# ---------------------------------------------------------------------------

class TestRunnerToolEvents:
    """Test that runner emits tool call events with content_blocks."""

    def test_emit_tool_event_no_bus(self):
        """Runner should not crash when no event bus is set."""
        from loom.engine.runner import SubtaskRunner
        runner = SubtaskRunner.__new__(SubtaskRunner)
        runner._event_bus = None
        # Should not raise
        runner._emit_tool_event(
            "tool_call_started", "task1", "sub1", "read_file", {"path": "a.py"},
        )

    def test_emit_tool_event_with_result(self):
        """Runner emits tool events with content_blocks data."""
        from unittest.mock import MagicMock

        from loom.engine.runner import SubtaskRunner

        bus = MagicMock()
        runner = SubtaskRunner.__new__(SubtaskRunner)
        runner._event_bus = bus

        block = ImageBlock(
            source_path="/tmp/img.png", width=100, height=100,
            text_fallback="Image",
        )
        result = ToolResult.multimodal("Image read", [block])

        runner._emit_tool_event(
            "tool_call_completed", "task1", "sub1",
            "read_file", {"path": "/tmp/img.png"},
            result=result,
        )

        bus.emit.assert_called_once()
        event = bus.emit.call_args[0][0]
        assert event.event_type == "tool_call_completed"
        assert event.task_id == "task1"
        assert event.data["success"] is True
        assert "content_blocks" in event.data
        assert len(event.data["content_blocks"]) == 1
        assert event.data["content_blocks"][0]["type"] == "image"

    def test_emit_tool_event_without_content_blocks(self):
        """Tool events without content_blocks don't include the field."""
        from unittest.mock import MagicMock

        from loom.engine.runner import SubtaskRunner

        bus = MagicMock()
        runner = SubtaskRunner.__new__(SubtaskRunner)
        runner._event_bus = bus

        result = ToolResult.ok("File contents")
        runner._emit_tool_event(
            "tool_call_completed", "task1", "sub1",
            "read_file", {"path": "a.py"},
            result=result,
        )

        event = bus.emit.call_args[0][0]
        assert "content_blocks" not in event.data
        assert event.data["success"] is True


# ---------------------------------------------------------------------------
# Memory extraction content_blocks annotation tests
# ---------------------------------------------------------------------------

class TestExtractorAnnotation:
    """Test that tool calls with content_blocks get annotated in extractor formatting."""

    def test_tool_line_includes_content_annotation(self):
        block = ImageBlock(source_path="/tmp/img.png", text_fallback="Image")
        result = ToolResult.multimodal("Image", [block])

        from loom.engine.runner import ToolCallRecord
        record = ToolCallRecord(tool="read_file", args={"path": "/tmp/img.png"}, result=result)

        # Simulate the formatting logic from runner._extract_memory
        status = "OK" if record.result.success else f"FAILED: {record.result.error}"
        line = f"- {record.tool}({json.dumps(record.args)}) → {status}"
        if record.result.content_blocks:
            block_types = [getattr(b, "type", "?") for b in record.result.content_blocks]
            line += f" [content: {', '.join(block_types)}]"

        assert "[content: image]" in line
        assert "read_file" in line

    def test_tool_line_no_blocks(self):
        result = ToolResult.ok("ok")
        from loom.engine.runner import ToolCallRecord
        record = ToolCallRecord(tool="shell_execute", args={"command": "ls"}, result=result)

        status = "OK" if record.result.success else f"FAILED: {record.result.error}"
        line = f"- {record.tool}({json.dumps(record.args)}) → {status}"
        if record.result.content_blocks:
            block_types = [getattr(b, "type", "?") for b in record.result.content_blocks]
            line += f" [content: {', '.join(block_types)}]"

        assert "[content:" not in line
