"""Content block type system for multimodal message content.

Defines the internal representation for rich content (images, documents,
thinking blocks) that flows through the tool → session → provider pipeline.
Every block carries a text_fallback for graceful degradation on text-only models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

# Media type constants
IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}

PDF_MEDIA_TYPE = "application/pdf"


@dataclass(frozen=True)
class ContentBlock:
    """Base for all content block types.

    Every block has a type tag and a text_fallback that is used when the
    model does not support the block's native type.
    """

    type: str
    text_fallback: str


@dataclass(frozen=True)
class TextBlock(ContentBlock):
    """Plain text content."""

    type: str = "text"
    text: str = ""
    text_fallback: str = ""

    def __post_init__(self):
        if not self.text_fallback:
            object.__setattr__(self, "text_fallback", self.text)


@dataclass(frozen=True)
class ImageBlock(ContentBlock):
    """An image with lazy base64 encoding.

    Stores the file path, not the bytes. The provider encodes on demand.
    This keeps session storage small and avoids redundant re-encoding.
    """

    type: str = "image"
    source_path: str = ""
    media_type: str = "image/png"
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    text_fallback: str = ""


@dataclass(frozen=True)
class DocumentBlock(ContentBlock):
    """A document (PDF) with optional extracted text.

    For providers that support native document blocks (Anthropic), the raw
    file is sent directly. For others, pages are rendered as images (if
    vision-capable) or the extracted text is used.
    """

    type: str = "document"
    source_path: str = ""
    media_type: str = "application/pdf"
    page_count: int = 0
    size_bytes: int = 0
    extracted_text: str = ""
    page_range: tuple[int, int] | None = None
    text_fallback: str = ""


@dataclass(frozen=True)
class ThinkingBlock(ContentBlock):
    """Model thinking/reasoning content.

    Anthropic returns these with a signature that must be passed back
    unmodified. Ollama returns a thinking field on the message.
    """

    type: str = "thinking"
    thinking: str = ""
    signature: str = ""
    text_fallback: str = ""

    def __post_init__(self):
        if not self.text_fallback:
            object.__setattr__(self, "text_fallback", "")


# Type alias for message content
Content = str | list[ContentBlock]


def serialize_block(block: ContentBlock) -> dict:
    """Serialize a ContentBlock for JSON storage.

    Images are stored as path references, not base64.
    """
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    elif isinstance(block, ImageBlock):
        return {
            "type": "image",
            "source_path": block.source_path,
            "media_type": block.media_type,
            "width": block.width,
            "height": block.height,
            "size_bytes": block.size_bytes,
            "text_fallback": block.text_fallback,
        }
    elif isinstance(block, DocumentBlock):
        return {
            "type": "document",
            "source_path": block.source_path,
            "media_type": block.media_type,
            "page_count": block.page_count,
            "size_bytes": block.size_bytes,
            "page_range": list(block.page_range) if block.page_range else None,
            "text_fallback": block.text_fallback,
        }
    elif isinstance(block, ThinkingBlock):
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "signature": block.signature,
        }
    return {"type": block.type, "text_fallback": block.text_fallback}


def deserialize_block(data: dict) -> ContentBlock:
    """Reconstruct a ContentBlock from stored JSON."""
    btype = data.get("type", "text")
    if btype == "text":
        return TextBlock(text=data.get("text", ""))
    elif btype == "image":
        return ImageBlock(
            source_path=data.get("source_path", ""),
            media_type=data.get("media_type", "image/png"),
            width=data.get("width", 0),
            height=data.get("height", 0),
            size_bytes=data.get("size_bytes", 0),
            text_fallback=data.get("text_fallback", ""),
        )
    elif btype == "document":
        pr = data.get("page_range")
        return DocumentBlock(
            source_path=data.get("source_path", ""),
            media_type=data.get("media_type", PDF_MEDIA_TYPE),
            page_count=data.get("page_count", 0),
            size_bytes=data.get("size_bytes", 0),
            page_range=tuple(pr) if pr else None,
            text_fallback=data.get("text_fallback", ""),
        )
    elif btype == "thinking":
        return ThinkingBlock(
            thinking=data.get("thinking", ""),
            signature=data.get("signature", ""),
        )
    return TextBlock(text=data.get("text_fallback", ""))


def serialize_blocks(blocks: list[ContentBlock]) -> str:
    """Serialize a list of content blocks to a JSON string."""
    return json.dumps([serialize_block(b) for b in blocks])


def deserialize_blocks(data: str) -> list[ContentBlock]:
    """Deserialize a JSON string to a list of content blocks."""
    try:
        items = json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(items, list):
        return []
    return [deserialize_block(item) for item in items if isinstance(item, dict)]
