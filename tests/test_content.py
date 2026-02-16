"""Tests for the content block type system."""

from __future__ import annotations

import json

from loom.content import (
    ContentBlock,
    DocumentBlock,
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    deserialize_block,
    deserialize_blocks,
    serialize_block,
    serialize_blocks,
)


class TestTextBlock:
    def test_text_fallback_auto(self):
        b = TextBlock(text="hello")
        assert b.text_fallback == "hello"

    def test_text_fallback_explicit(self):
        b = TextBlock(text="hello", text_fallback="override")
        assert b.text_fallback == "override"

    def test_frozen(self):
        b = TextBlock(text="hello")
        import pytest
        with pytest.raises(AttributeError):
            b.text = "changed"  # type: ignore[misc]


class TestImageBlock:
    def test_defaults(self):
        b = ImageBlock(
            source_path="/tmp/img.png",
            media_type="image/png",
            width=640,
            height=480,
            size_bytes=12345,
            text_fallback="[Image: img.png]",
        )
        assert b.type == "image"
        assert b.source_path == "/tmp/img.png"
        assert b.width == 640

    def test_is_content_block(self):
        b = ImageBlock()
        assert isinstance(b, ContentBlock)


class TestDocumentBlock:
    def test_defaults(self):
        b = DocumentBlock(
            source_path="/tmp/doc.pdf",
            page_count=10,
            size_bytes=50000,
            extracted_text="some text",
            page_range=(0, 10),
            text_fallback="[PDF: doc.pdf]",
        )
        assert b.type == "document"
        assert b.page_count == 10
        assert b.page_range == (0, 10)

    def test_page_range_none(self):
        b = DocumentBlock()
        assert b.page_range is None


class TestThinkingBlock:
    def test_defaults(self):
        b = ThinkingBlock(thinking="Let me think...", signature="abc123")
        assert b.type == "thinking"
        assert b.text_fallback == ""

    def test_text_fallback_empty(self):
        b = ThinkingBlock()
        assert b.text_fallback == ""


class TestSerializeBlock:
    def test_text_block(self):
        b = TextBlock(text="hello")
        d = serialize_block(b)
        assert d == {"type": "text", "text": "hello"}

    def test_image_block(self):
        b = ImageBlock(
            source_path="/img.png",
            media_type="image/png",
            width=100,
            height=200,
            size_bytes=5000,
            text_fallback="[Image]",
        )
        d = serialize_block(b)
        assert d["type"] == "image"
        assert d["source_path"] == "/img.png"
        assert d["width"] == 100
        assert d["text_fallback"] == "[Image]"

    def test_document_block(self):
        b = DocumentBlock(
            source_path="/doc.pdf",
            page_count=5,
            size_bytes=10000,
            page_range=(0, 5),
            text_fallback="[PDF]",
        )
        d = serialize_block(b)
        assert d["type"] == "document"
        assert d["page_range"] == [0, 5]

    def test_document_block_no_page_range(self):
        b = DocumentBlock(source_path="/doc.pdf")
        d = serialize_block(b)
        assert d["page_range"] is None

    def test_thinking_block(self):
        b = ThinkingBlock(thinking="deep thought", signature="sig123")
        d = serialize_block(b)
        assert d == {"type": "thinking", "thinking": "deep thought", "signature": "sig123"}

    def test_generic_content_block(self):
        b = ContentBlock(type="unknown", text_fallback="fallback text")
        d = serialize_block(b)
        assert d == {"type": "unknown", "text_fallback": "fallback text"}


class TestDeserializeBlock:
    def test_text_block(self):
        b = deserialize_block({"type": "text", "text": "hello"})
        assert isinstance(b, TextBlock)
        assert b.text == "hello"

    def test_image_block(self):
        b = deserialize_block({
            "type": "image",
            "source_path": "/img.png",
            "media_type": "image/jpeg",
            "width": 640,
            "height": 480,
            "size_bytes": 12345,
            "text_fallback": "[Image]",
        })
        assert isinstance(b, ImageBlock)
        assert b.source_path == "/img.png"
        assert b.media_type == "image/jpeg"
        assert b.width == 640

    def test_document_block(self):
        b = deserialize_block({
            "type": "document",
            "source_path": "/doc.pdf",
            "page_count": 5,
            "size_bytes": 10000,
            "page_range": [0, 5],
            "text_fallback": "[PDF]",
        })
        assert isinstance(b, DocumentBlock)
        assert b.page_range == (0, 5)

    def test_document_block_no_page_range(self):
        b = deserialize_block({"type": "document", "source_path": "/doc.pdf"})
        assert isinstance(b, DocumentBlock)
        assert b.page_range is None

    def test_thinking_block(self):
        b = deserialize_block({
            "type": "thinking",
            "thinking": "hmm",
            "signature": "sig",
        })
        assert isinstance(b, ThinkingBlock)
        assert b.thinking == "hmm"

    def test_unknown_type_returns_text(self):
        b = deserialize_block({"type": "audio", "text_fallback": "fallback"})
        assert isinstance(b, TextBlock)
        assert b.text == "fallback"

    def test_missing_type_returns_text(self):
        b = deserialize_block({"text": "hello"})
        assert isinstance(b, TextBlock)


class TestRoundTrip:
    def test_text_round_trip(self):
        original = TextBlock(text="hello world")
        restored = deserialize_block(serialize_block(original))
        assert isinstance(restored, TextBlock)
        assert restored.text == original.text

    def test_image_round_trip(self):
        original = ImageBlock(
            source_path="/path/to/image.png",
            media_type="image/png",
            width=800,
            height=600,
            size_bytes=50000,
            text_fallback="[Image: image.png, 800x600]",
        )
        restored = deserialize_block(serialize_block(original))
        assert isinstance(restored, ImageBlock)
        assert restored.source_path == original.source_path
        assert restored.width == original.width
        assert restored.text_fallback == original.text_fallback

    def test_document_round_trip(self):
        original = DocumentBlock(
            source_path="/doc.pdf",
            page_count=10,
            size_bytes=100000,
            page_range=(5, 10),
            text_fallback="[PDF pages 6-10]",
        )
        restored = deserialize_block(serialize_block(original))
        assert isinstance(restored, DocumentBlock)
        assert restored.page_range == original.page_range
        assert restored.page_count == original.page_count

    def test_thinking_round_trip(self):
        original = ThinkingBlock(thinking="analysis", signature="abc")
        restored = deserialize_block(serialize_block(original))
        assert isinstance(restored, ThinkingBlock)
        assert restored.thinking == original.thinking
        assert restored.signature == original.signature


class TestSerializeBlocks:
    def test_serialize_list(self):
        blocks = [TextBlock(text="hello"), ImageBlock(source_path="/img.png")]
        result = serialize_blocks(blocks)
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["type"] == "text"
        assert parsed[1]["type"] == "image"

    def test_deserialize_list(self):
        data = json.dumps([
            {"type": "text", "text": "hello"},
            {"type": "image", "source_path": "/img.png"},
        ])
        blocks = deserialize_blocks(data)
        assert len(blocks) == 2
        assert isinstance(blocks[0], TextBlock)
        assert isinstance(blocks[1], ImageBlock)

    def test_deserialize_invalid_json(self):
        assert deserialize_blocks("not json") == []

    def test_deserialize_not_a_list(self):
        assert deserialize_blocks(json.dumps({"type": "text"})) == []

    def test_deserialize_skips_non_dicts(self):
        data = json.dumps([{"type": "text", "text": "ok"}, 42, "bad"])
        blocks = deserialize_blocks(data)
        assert len(blocks) == 1


class TestExtractedTextSerialization:
    """Verify DocumentBlock.extracted_text survives round-trips."""

    def test_extracted_text_serialized(self):
        b = DocumentBlock(
            source_path="/doc.pdf",
            extracted_text="Page 1 content here",
            text_fallback="[PDF]",
        )
        d = serialize_block(b)
        assert d["extracted_text"] == "Page 1 content here"

    def test_extracted_text_deserialized(self):
        b = deserialize_block({
            "type": "document",
            "source_path": "/doc.pdf",
            "extracted_text": "Hello from PDF",
        })
        assert isinstance(b, DocumentBlock)
        assert b.extracted_text == "Hello from PDF"

    def test_extracted_text_round_trip(self):
        original = DocumentBlock(
            source_path="/doc.pdf",
            page_count=3,
            extracted_text="Full text content",
            text_fallback="[PDF]",
        )
        restored = deserialize_block(serialize_block(original))
        assert isinstance(restored, DocumentBlock)
        assert restored.extracted_text == "Full text content"

    def test_empty_extracted_text_not_serialized(self):
        b = DocumentBlock(source_path="/doc.pdf")
        d = serialize_block(b)
        assert "extracted_text" not in d


class TestPageRangeValidation:
    """Verify page_range is validated during deserialization."""

    def test_valid_page_range(self):
        b = deserialize_block({
            "type": "document",
            "page_range": [0, 20],
        })
        assert b.page_range == (0, 20)

    def test_invalid_page_range_reversed(self):
        b = deserialize_block({
            "type": "document",
            "page_range": [10, 5],
        })
        assert b.page_range is None

    def test_invalid_page_range_negative(self):
        b = deserialize_block({
            "type": "document",
            "page_range": [-5, 10],
        })
        assert b.page_range is None

    def test_invalid_page_range_wrong_length(self):
        b = deserialize_block({
            "type": "document",
            "page_range": [1],
        })
        assert b.page_range is None

    def test_invalid_page_range_three_elements(self):
        b = deserialize_block({
            "type": "document",
            "page_range": [0, 5, 10],
        })
        assert b.page_range is None

    def test_page_range_zero_zero_valid(self):
        b = deserialize_block({
            "type": "document",
            "page_range": [0, 0],
        })
        assert b.page_range == (0, 0)

    def test_page_range_none(self):
        b = deserialize_block({
            "type": "document",
            "page_range": None,
        })
        assert b.page_range is None
