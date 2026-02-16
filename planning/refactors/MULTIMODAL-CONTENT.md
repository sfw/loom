# Multimodal Content Support: From Text-Only to Rich Content Blocks

## Problem Statement

Loom's entire message pipeline is text-only. `ToolResult.output` is a `str`. Session messages
store `content` as JSON strings. Model providers construct text-only content blocks. When a user
asks the agent to read an image, it gets back `"[Image file: photo.png, 25,432 bytes, type: .png]"`
— the model never sees the actual image.

Meanwhile, all three supported backends have multimodal capabilities that go unused:

| Backend   | Text | Images | PDFs/Documents | Audio | Thinking | Citations |
|-----------|------|--------|----------------|-------|----------|-----------|
| Anthropic | ✓    | ✓ base64/URL | ✓ native document blocks (100pg, 32MB) | ✗ | ✓ extended thinking | ✓ |
| OpenAI    | ✓    | ✓ base64/URL | ✓ file content parts | ✓ input/output | ✗ (internal) | ✓ annotations |
| Ollama    | ✓    | ✓ base64 (images field) | ✗ | ✗ | ✓ thinking field | ✗ |

**Current state:** 100% text. Models with eyes are blind. PDFs are lossy text extraction only.
**Target state:** Content blocks flow end-to-end. Vision models see images. PDFs render natively
where supported, fall back to text extraction where not. Thinking blocks are captured and surfaced.

---

## Design Principles

1. **Text is always the fallback.** Every content block must carry or generate a text
   representation. If the model doesn't support vision, it gets the text version automatically.
   No feature should break text-only models.

2. **Provider-agnostic internal format, provider-specific wire format.** The internal
   representation is uniform. Each provider converts to its native API format at the boundary.
   Session storage is provider-independent.

3. **Lazy encoding, eager validation.** Validate files (size, format, corruption) at read time.
   Encode to base64 only when building the API request, not when storing in the session. This
   keeps the session DB lean and avoids re-encoding on model switches.

4. **Budget-aware.** Images and documents consume tokens. The context window manager must
   account for multimodal token costs, not just text token counts.

5. **Progressive capability.** Ship in phases. Phase 1 (images) is useful alone. Phase 2 (PDFs)
   builds on the same plumbing. Phase 3 (thinking/citations) is additive polish.

---

## Architecture

### Layer 1: Content Block Type System

A new module `src/loom/content.py` defines the internal content representation:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

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
    source_path: str = ""          # absolute path to image file
    media_type: str = "image/png"  # MIME type
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    text_fallback: str = ""        # "[Image: diagram.png, 640x480, 24KB]"


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
    extracted_text: str = ""       # pypdf text extraction (always populated)
    page_range: tuple[int, int] | None = None  # (start, end) 0-indexed, for chunked reads
    text_fallback: str = ""


@dataclass(frozen=True)
class ThinkingBlock(ContentBlock):
    """Model thinking/reasoning content.

    Anthropic returns these with a signature that must be passed back
    unmodified. Ollama returns a thinking field on the message.
    """
    type: str = "thinking"
    thinking: str = ""
    signature: str = ""            # Anthropic opaque signature
    text_fallback: str = ""        # Usually empty — thinking is hidden from user

    def __post_init__(self):
        if not self.text_fallback:
            object.__setattr__(self, "text_fallback", "")


# Type alias for message content
Content = str | list[ContentBlock]
```

**Key decisions:**
- `frozen=True` — content blocks are immutable values, safe to store and compare.
- `text_fallback` on every block — guarantees graceful degradation.
- `ImageBlock` stores `source_path`, not bytes — avoids 1.3x base64 bloat in session DB.
- `ThinkingBlock` captures Anthropic signatures for multi-turn thinking.

### Layer 2: ToolResult Enhancement

```python
@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: str                                    # text output (always present)
    content_blocks: list[ContentBlock] | None = None  # NEW: rich content
    data: dict | None = None
    files_changed: list[str] = field(default_factory=list)
    error: str | None = None

    MAX_OUTPUT_SIZE = 30720  # 30KB

    def to_json(self) -> str:
        """Serialize for session storage. Content blocks stored as typed dicts."""
        payload = {
            "success": self.success,
            "output": self.output[:self.MAX_OUTPUT_SIZE],
            "error": self.error,
            "files_changed": self.files_changed,
        }
        if self.content_blocks:
            payload["content_blocks"] = [
                _serialize_block(b) for b in self.content_blocks
            ]
        return json.dumps(payload)

    @classmethod
    def ok(cls, output: str, **kwargs) -> ToolResult:
        return cls(success=True, output=output, **kwargs)

    @classmethod
    def multimodal(
        cls, output: str, blocks: list[ContentBlock], **kwargs
    ) -> ToolResult:
        """Create a result with both text and content blocks."""
        return cls(success=True, output=output, content_blocks=blocks, **kwargs)


def _serialize_block(block: ContentBlock) -> dict:
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
            # extracted_text NOT stored in JSON — re-extract on replay if needed
        }
    elif isinstance(block, ThinkingBlock):
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "signature": block.signature,
        }
    return {"type": block.type, "text_fallback": block.text_fallback}


def _deserialize_block(data: dict) -> ContentBlock:
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
```

### Layer 3: Model Capability Declaration

Add capability flags to `ModelConfig`:

```python
@dataclass(frozen=True)
class ModelCapabilities:
    """What content types this model can handle."""
    vision: bool = False           # Can process images
    native_pdf: bool = False       # Can process PDF document blocks natively
    thinking: bool = False         # Returns thinking/reasoning blocks
    citations: bool = False        # Returns citation annotations
    audio_input: bool = False      # Can process audio (future)
    audio_output: bool = False     # Can generate audio (future)

    @classmethod
    def auto_detect(cls, provider: str, model: str) -> ModelCapabilities:
        """Infer capabilities from provider and model name.

        This is best-effort. Users can override in loom.toml.
        """
        model_lower = model.lower()

        if provider == "anthropic":
            return cls(
                vision=True,           # All Claude 3+ models
                native_pdf=True,       # Anthropic document blocks
                thinking="opus" in model_lower or "sonnet" in model_lower,
                citations=True,
            )

        if provider == "ollama":
            # Vision models have known names
            vision_models = {
                "llava", "bakllava", "gemma3", "smolvlm",
                "llama3.2-vision", "moondream", "minicpm-v",
            }
            has_vision = any(v in model_lower for v in vision_models)
            return cls(
                vision=has_vision,
                thinking="deepseek" in model_lower or "qwq" in model_lower,
            )

        if provider == "openai_compatible":
            # Common vision-capable models
            has_vision = any(v in model_lower for v in [
                "gpt-4o", "gpt-4-vision", "gpt-4-turbo",
                "gemini", "pixtral", "internvl",
            ])
            return cls(
                vision=has_vision,
                native_pdf="gpt-4o" in model_lower,
            )

        return cls()  # all False — safe default
```

**Config extension in loom.toml:**

```toml
[models.llava]
provider = "ollama"
model = "llava:13b"
roles = ["executor"]

[models.llava.capabilities]
vision = true        # Override auto-detection
native_pdf = false
thinking = false
```

The `ModelConfig` dataclass gains:

```python
@dataclass(frozen=True)
class ModelConfig:
    provider: str
    base_url: str = ""
    model: str = ""
    max_tokens: int = 4096
    temperature: float = 0.1
    roles: list[str] = field(default_factory=lambda: ["executor"])
    api_key: str = ""
    tier: int = 0
    capabilities: ModelCapabilities | None = None  # NEW — None = auto-detect

    @property
    def resolved_capabilities(self) -> ModelCapabilities:
        if self.capabilities is not None:
            return self.capabilities
        return ModelCapabilities.auto_detect(self.provider, self.model)
```

### Layer 4: ReadFileTool — Multimodal Output

```python
class ReadFileTool(Tool):
    """Read file with multimodal support."""

    # Size limits
    MAX_IMAGE_BYTES = 5 * 1024 * 1024    # 5MB per image
    MAX_PDF_BYTES = 32 * 1024 * 1024     # 32MB per PDF
    MAX_PDF_PAGES_PER_READ = 20          # Chunk large PDFs
    IMAGE_MAX_DIMENSION = 2048           # Resize if larger

    def _read_image(self, path: Path) -> ToolResult:
        """Return image as a content block with text fallback."""
        size = path.stat().st_size

        if size > self.MAX_IMAGE_BYTES:
            return ToolResult.ok(
                f"[Image too large: {path.name}, {size:,} bytes, "
                f"limit is {self.MAX_IMAGE_BYTES:,} bytes. "
                f"Consider resizing or converting to JPEG.]"
            )

        suffix = path.suffix.lower()
        media_type = IMAGE_MEDIA_TYPES.get(suffix)
        if not media_type:
            return ToolResult.ok(f"[Unsupported image format: {suffix}]")

        # Get dimensions without loading full image
        width, height = self._get_image_dimensions(path)

        text_fallback = (
            f"[Image: {path.name}, {width}x{height}, "
            f"{size:,} bytes, {suffix}]"
        )

        block = ImageBlock(
            source_path=str(path),
            media_type=media_type,
            width=width,
            height=height,
            size_bytes=size,
            text_fallback=text_fallback,
        )

        return ToolResult.multimodal(
            output=text_fallback,
            blocks=[block],
            data={"type": "image", "name": path.name,
                  "size": size, "format": suffix},
        )

    def _read_pdf(self, path: Path, page_start: int = 0,
                  page_end: int | None = None) -> ToolResult:
        """Read PDF with pagination and multimodal output."""
        size = path.stat().st_size

        if size > self.MAX_PDF_BYTES:
            return ToolResult.fail(
                f"PDF too large: {size:,} bytes "
                f"(limit: {self.MAX_PDF_BYTES:,} bytes)"
            )

        try:
            import pypdf
        except ImportError:
            return ToolResult.ok(
                f"[PDF: {path.name}, {size:,} bytes]\n"
                "Install 'pypdf' to extract text: pip install pypdf",
            )

        reader = pypdf.PdfReader(path)
        total_pages = len(reader.pages)

        # Clamp page range
        if page_end is None:
            page_end = min(page_start + self.MAX_PDF_PAGES_PER_READ, total_pages)
        page_end = min(page_end, total_pages)
        page_start = max(0, page_start)

        # Always extract text (used as fallback and for text-only models)
        pages_text = []
        for i in range(page_start, page_end):
            text = reader.pages[i].extract_text() or ""
            if text.strip():
                pages_text.append(f"--- Page {i + 1} ---\n{text}")

        extracted = "\n\n".join(pages_text) if pages_text else ""
        pagination_note = ""
        if page_end < total_pages:
            pagination_note = (
                f"\n\n[Showing pages {page_start+1}-{page_end} of "
                f"{total_pages}. Use page_start={page_end} to continue.]"
            )

        text_fallback = extracted + pagination_note if extracted else (
            f"[PDF: {path.name}, {total_pages} pages, no extractable text]"
            + pagination_note
        )

        block = DocumentBlock(
            source_path=str(path),
            page_count=total_pages,
            size_bytes=size,
            extracted_text=extracted,
            page_range=(page_start, page_end),
            text_fallback=text_fallback,
        )

        return ToolResult.multimodal(
            output=text_fallback,
            blocks=[block],
            data={"type": "pdf", "pages": total_pages,
                  "page_range": [page_start, page_end]},
        )

    @staticmethod
    def _get_image_dimensions(path: Path) -> tuple[int, int]:
        """Get image dimensions by reading only the header."""
        try:
            # Use struct-based header parsing — no PIL dependency
            # Falls back to (0, 0) if format not recognized
            ...
        except Exception:
            return (0, 0)
```

**New parameter on read_file tool:**

```json
{
  "name": "read_file",
  "parameters": {
    "properties": {
      "file_path": {"type": "string"},
      "offset": {"type": "integer"},
      "limit": {"type": "integer"},
      "page_start": {"type": "integer", "description": "PDF: first page (0-indexed)"},
      "page_end": {"type": "integer", "description": "PDF: last page (exclusive)"}
    }
  }
}
```

### Layer 5: Provider Content Block Conversion

Each provider gets a `_build_content()` method that converts `list[ContentBlock]` to its
native format, respecting the model's capabilities.

**Anthropic:**

```python
def _build_tool_result_content(
    self, result_json: str, caps: ModelCapabilities,
) -> str | list[dict]:
    """Convert a tool result with possible content blocks to Anthropic format."""
    parsed = json.loads(result_json)
    blocks = parsed.get("content_blocks")
    if not blocks:
        return parsed.get("output", "")

    anthropic_blocks = []
    for block in blocks:
        btype = block["type"]

        if btype == "text":
            anthropic_blocks.append({"type": "text", "text": block["text"]})

        elif btype == "image" and caps.vision:
            image_data = self._encode_image(block["source_path"])
            if image_data:
                anthropic_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block["media_type"],
                        "data": image_data,
                    },
                })
            else:
                # File gone — use fallback
                anthropic_blocks.append({
                    "type": "text",
                    "text": block["text_fallback"],
                })

        elif btype == "document" and caps.native_pdf:
            doc_data = self._encode_file(block["source_path"])
            if doc_data:
                anthropic_blocks.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": doc_data,
                    },
                })
            else:
                anthropic_blocks.append({
                    "type": "text",
                    "text": block.get("text_fallback", ""),
                })

        else:
            # Unsupported type or capability — use text fallback
            fb = block.get("text_fallback", "")
            if fb:
                anthropic_blocks.append({"type": "text", "text": fb})

    return anthropic_blocks if anthropic_blocks else parsed.get("output", "")
```

**Ollama:**

```python
def _build_message(
    self, msg: dict, caps: ModelCapabilities,
) -> dict:
    """Convert internal message format to Ollama format."""
    result = {"role": msg["role"], "content": msg.get("content", "")}

    # Ollama puts images in a separate field, not in content
    if caps.vision and msg.get("role") == "tool":
        parsed = json.loads(msg.get("content", "{}"))
        blocks = parsed.get("content_blocks", [])
        images = []
        for block in blocks:
            if block["type"] == "image":
                data = self._encode_image(block["source_path"])
                if data:
                    images.append(data)
            elif block["type"] == "document":
                # Ollama has no native PDF — render pages as images
                page_images = self._pdf_pages_to_images(
                    block["source_path"], block.get("page_range")
                )
                images.extend(page_images)
        if images:
            result["images"] = images

    return result
```

**OpenAI-compatible:**

```python
def _build_user_content(
    self, blocks: list[dict], caps: ModelCapabilities,
) -> str | list[dict]:
    """Convert content blocks to OpenAI multipart content."""
    parts = []
    for block in blocks:
        btype = block["type"]

        if btype == "text":
            parts.append({"type": "text", "text": block["text"]})

        elif btype == "image" and caps.vision:
            data = self._encode_image(block["source_path"])
            if data:
                media = block["media_type"]
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media};base64,{data}",
                        "detail": "high",
                    },
                })
            else:
                parts.append({"type": "text", "text": block["text_fallback"]})

        elif btype == "document" and caps.native_pdf:
            data = self._encode_file(block["source_path"])
            if data:
                parts.append({
                    "type": "file",
                    "file": {
                        "filename": Path(block["source_path"]).name,
                        "file_data": f"data:application/pdf;base64,{data}",
                    },
                })
            else:
                parts.append({"type": "text", "text": block["text_fallback"]})

        else:
            fb = block.get("text_fallback", "")
            if fb:
                parts.append({"type": "text", "text": fb})

    return parts if parts else ""
```

### Layer 6: Session Layer Changes

**Message storage:** Content blocks are stored in the tool result JSON alongside `output`.
No schema change to the SQLite store — `content` column already stores arbitrary JSON.

**Context window management:** The `_context_window()` method in `session.py` currently
estimates token count using `len(json.dumps(msg)) // 4`. This must account for multimodal
token costs:

```python
def _estimate_message_tokens(self, msg: dict) -> int:
    """Estimate token count including multimodal content."""
    if msg["role"] == "tool":
        parsed = json.loads(msg.get("content", "{}"))
        blocks = parsed.get("content_blocks", [])
        tokens = len(parsed.get("output", "")) // 4  # text tokens

        for block in blocks:
            if block["type"] == "image":
                # Anthropic formula: (w * h) / 750
                w = block.get("width", 1024)
                h = block.get("height", 1024)
                tokens += (w * h) // 750
            elif block["type"] == "document":
                # ~1500 tokens per page (empirical)
                pages = block.get("page_count", 1)
                pr = block.get("page_range")
                if pr:
                    pages = pr[1] - pr[0]
                tokens += pages * 1500
        return tokens

    # Text-only messages
    return len(json.dumps(msg.get("content", ""))) // 4
```

**Context window eviction:** When trimming to fit the window, multimodal messages are
eviction candidates before text messages (they consume more tokens per message). The
eviction order becomes:

1. Oldest image/document tool results (high token cost, low reference value over time)
2. Oldest text tool results
3. Oldest user/assistant turns (never evict the last 2 turns)

### Layer 7: Image Processing Utilities

A small utility module for image operations that avoids hard PIL dependency:

```python
# src/loom/content_utils.py

import base64
import struct
from pathlib import Path


def encode_image_base64(path: Path, max_dimension: int = 2048) -> str | None:
    """Read and base64-encode an image file.

    If the image exceeds max_dimension on either axis and Pillow is
    available, resize it proportionally. Without Pillow, returns the
    raw image if within the 5MB limit, or None if too large.
    """
    if not path.exists():
        return None

    size = path.stat().st_size
    if size > 5 * 1024 * 1024:
        # Try to resize with Pillow
        try:
            return _resize_and_encode(path, max_dimension)
        except ImportError:
            return None  # Too large and no Pillow to resize

    data = path.read_bytes()

    # Check if resize needed
    w, h = get_image_dimensions_from_bytes(data[:64], path.suffix)
    if (w > max_dimension or h > max_dimension) and w > 0:
        try:
            return _resize_and_encode(path, max_dimension)
        except ImportError:
            pass  # Send as-is, let the API handle it

    return base64.b64encode(data).decode("ascii")


def get_image_dimensions(path: Path) -> tuple[int, int]:
    """Read image dimensions from file header without loading the full image.

    Supports PNG, JPEG, GIF, BMP, WebP headers.
    Returns (0, 0) for unrecognized formats.
    """
    try:
        header = path.read_bytes()[:32]
        return get_image_dimensions_from_bytes(header, path.suffix)
    except Exception:
        return (0, 0)


def get_image_dimensions_from_bytes(
    header: bytes, suffix: str,
) -> tuple[int, int]:
    """Parse image dimensions from header bytes."""
    if header[:8] == b"\x89PNG\r\n\x1a\n" and len(header) >= 24:
        w, h = struct.unpack(">II", header[16:24])
        return (w, h)
    if header[:2] == b"\xff\xd8":  # JPEG — need to scan markers
        return (0, 0)  # Complex; defer to Pillow or skip
    if header[:6] in (b"GIF87a", b"GIF89a") and len(header) >= 10:
        w, h = struct.unpack("<HH", header[6:10])
        return (w, h)
    if header[:2] == b"BM" and len(header) >= 26:
        w, h = struct.unpack("<II", header[18:26])
        return (w, h)
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        # VP8 lossy
        if header[12:16] == b"VP8 " and len(header) >= 30:
            w = struct.unpack("<H", header[26:28])[0] & 0x3FFF
            h = struct.unpack("<H", header[28:30])[0] & 0x3FFF
            return (w, h)
    return (0, 0)


def _resize_and_encode(path: Path, max_dim: int) -> str:
    """Resize with Pillow and return base64. Raises ImportError if no Pillow."""
    from PIL import Image
    img = Image.open(path)
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    import io
    buf = io.BytesIO()
    fmt = "PNG" if path.suffix.lower() == ".png" else "JPEG"
    img.save(buf, format=fmt, quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def pdf_pages_to_images(
    path: Path, page_range: tuple[int, int] | None = None,
    dpi: int = 150, max_dimension: int = 2048,
) -> list[str]:
    """Render PDF pages as base64-encoded images.

    Used for Ollama (no native PDF support) with vision-capable models.
    Requires pdf2image (poppler) or pymupdf. Returns empty list if
    neither is available.
    """
    try:
        import fitz  # pymupdf
        doc = fitz.open(path)
        start, end = page_range or (0, len(doc))
        images = []
        for i in range(start, min(end, len(doc))):
            page = doc[i]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            if pix.width > max_dimension or pix.height > max_dimension:
                scale = max_dimension / max(pix.width, pix.height)
                mat = fitz.Matrix(scale * dpi / 72, scale * dpi / 72)
                pix = page.get_pixmap(matrix=mat)
            images.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
        return images
    except ImportError:
        pass

    try:
        from pdf2image import convert_from_path
        start, end = page_range or (0, 999)
        pages = convert_from_path(
            path, dpi=dpi, first_page=start + 1, last_page=end,
        )
        images = []
        for page in pages:
            page.thumbnail((max_dimension, max_dimension))
            import io
            buf = io.BytesIO()
            page.save(buf, format="PNG")
            images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        return images
    except ImportError:
        return []
```

### Layer 8: Thinking Block Support

**Anthropic provider** — capture thinking blocks from response:

```python
# In _parse_response():
thinking_blocks = []
for block in response_content:
    if block["type"] == "thinking":
        thinking_blocks.append(ThinkingBlock(
            thinking=block["thinking"],
            signature=block["signature"],
        ))
    elif block["type"] == "redacted_thinking":
        thinking_blocks.append(ThinkingBlock(
            thinking="[redacted]",
            signature=block["data"],
        ))

# Store in ModelResponse
response.thinking = thinking_blocks  # New field
```

**Ollama provider** — capture thinking field:

```python
thinking = message.get("thinking", "")
if thinking:
    response.thinking = [ThinkingBlock(thinking=thinking)]
```

**ModelResponse extension:**

```python
@dataclass
class ModelResponse:
    text: str
    tool_calls: list[ToolCall] | None = None
    thinking: list[ThinkingBlock] | None = None   # NEW
    raw: str | dict = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    latency_ms: int = 0
```

**Session layer** — pass thinking blocks back in multi-turn:

```python
# When storing assistant message with thinking:
if response.thinking:
    msg["thinking_blocks"] = [
        {"thinking": t.thinking, "signature": t.signature}
        for t in response.thinking
    ]

# When building messages for Anthropic:
# Prepend thinking blocks before text content in assistant messages
```

### Layer 9: TUI Display

**Image content:**
- Tool output preview: `"[Image: photo.png, 640x480]"` — no inline image in terminal
- Collapsible section shows dimensions, size, path
- Future: use terminal image protocols (iTerm2 inline images, Kitty graphics protocol)

**PDF content:**
- Preview: `"[PDF: report.pdf, pages 1-20 of 47]"`
- Collapsible section shows extracted text
- Shows pagination hint if more pages available

**Thinking blocks:**
- Collapsible `<thinking>` section in dim text
- Hidden by default, expandable by user

---

## File Size Limits and Chunking Strategy

### Images

| Constraint         | Limit         | Action                                  |
|--------------------|---------------|-----------------------------------------|
| Max file size      | 5 MB          | Try Pillow resize → fail with message   |
| Max dimension      | 2048 px       | Resize proportionally with Pillow       |
| Max per message    | 10             | Warn and skip extras                    |
| SVG files          | —             | Text fallback only (not rasterized)     |
| Animated GIF       | —             | Send as-is (APIs take first frame)      |

### PDFs

| Constraint              | Limit          | Action                              |
|-------------------------|----------------|-------------------------------------|
| Max file size           | 32 MB          | Reject with error message           |
| Max pages per read      | 20             | Return page range + pagination hint |
| Max total pages         | 100            | Warn at read time                   |
| Scanned (no text)       | —              | Render as images if vision-capable   |
| Password-protected      | —              | Fail with clear error               |

### Chunking / Pagination

PDFs use explicit pagination via `page_start` / `page_end` parameters on `read_file`:

```
Agent: read_file("report.pdf")
→ Pages 1-20 of 47. Use page_start=20 to continue.

Agent: read_file("report.pdf", page_start=20)
→ Pages 21-40 of 47. Use page_start=40 to continue.

Agent: read_file("report.pdf", page_start=40)
→ Pages 41-47 of 47.
```

The model sees the pagination hint in the text output and can request more pages.
This keeps any single tool result within reasonable token budget (~30K tokens for
20 text-heavy pages or ~20 image renders).

---

## Fallback Matrix

What happens when a model doesn't support a content type:

| Content Type | Vision Model | Text-Only Model |
|---|---|---|
| Image | Native image block | `text_fallback`: "[Image: name, WxH, size]" |
| PDF (Anthropic) | Native document block | pypdf text extraction |
| PDF (OpenAI) | Native file block | pypdf text extraction |
| PDF (Ollama) | Pages rendered as images | pypdf text extraction |
| Thinking | Stored + passed back | Ignored (not sent) |
| SVG | Text fallback | Text fallback (XML source) |

---

## Implementation Phases

### Phase 1: Content Block Foundation + Images (Priority: HIGH)

**Files changed:**
- `NEW src/loom/content.py` — ContentBlock types, serialize/deserialize
- `NEW src/loom/content_utils.py` — base64 encoding, image dimensions, resize
- `MOD src/loom/tools/registry.py` — `content_blocks` field on ToolResult
- `MOD src/loom/tools/file_ops.py` — ReadFileTool._read_image() returns ImageBlock
- `MOD src/loom/config.py` — ModelCapabilities dataclass, capabilities on ModelConfig
- `MOD src/loom/models/base.py` — No changes needed (messages are already list[dict])
- `MOD src/loom/models/anthropic_provider.py` — Image content blocks in tool results
- `MOD src/loom/models/ollama_provider.py` — Images field on messages
- `MOD src/loom/models/openai_provider.py` — image_url content parts
- `MOD src/loom/cowork/session.py` — Token estimation for images
- `NEW tests/test_content.py` — ContentBlock serialization round-trips
- `NEW tests/test_content_utils.py` — Image encoding, dimensions, resize
- `MOD tests/test_tools.py` — ReadFileTool image tests with content blocks

**Estimated scope:** ~600 lines new code, ~200 lines modified

### Phase 2: PDF / Document Support (Priority: HIGH)

**Files changed:**
- `MOD src/loom/content_utils.py` — pdf_pages_to_images()
- `MOD src/loom/tools/file_ops.py` — ReadFileTool._read_pdf() with pagination + DocumentBlock
- `MOD src/loom/models/anthropic_provider.py` — Document blocks
- `MOD src/loom/models/ollama_provider.py` — PDF-as-images for vision models
- `MOD src/loom/models/openai_provider.py` — File content parts
- `MOD src/loom/cowork/session.py` — Token estimation for documents
- `MOD tests/test_tools.py` — PDF pagination tests
- `NEW tests/test_pdf_multimodal.py` — PDF rendering + provider conversion tests

**Estimated scope:** ~400 lines new code, ~150 lines modified

### Phase 3: Thinking Blocks + Context Window (Priority: MEDIUM)

**Files changed:**
- `MOD src/loom/models/base.py` — `thinking` field on ModelResponse
- `MOD src/loom/models/anthropic_provider.py` — Parse thinking/redacted_thinking blocks
- `MOD src/loom/models/ollama_provider.py` — Parse thinking field
- `MOD src/loom/cowork/session.py` — Store/replay thinking blocks, eviction strategy
- `MOD src/loom/tui/widgets/tool_call.py` — Thinking block display
- `MOD tests/test_models.py` — Thinking block parsing tests

**Estimated scope:** ~200 lines new code, ~100 lines modified

### Phase 4: TUI + Display Polish (Priority: LOW)

**Files changed:**
- `MOD src/loom/tui/widgets/tool_call.py` — Image/PDF preview rendering
- `MOD src/loom/tui/app.py` — Pagination controls for PDF viewer
- `MOD src/loom/cowork/display.py` — CLI multimodal output indicators

**Estimated scope:** ~150 lines new code, ~50 lines modified

---

## Testing Strategy

1. **Unit tests for content blocks** — Serialization round-trips, text_fallback generation
2. **Unit tests for image utils** — Header parsing (PNG, GIF, BMP, WebP), base64 encoding,
   resize (mock Pillow), dimension limits
3. **Unit tests for PDF handling** — Pagination math, page range clamping, text extraction
   fallback, size limit enforcement
4. **Provider integration tests** — Each provider correctly converts content blocks to
   its native format. Mock the API calls, verify the request payload structure.
5. **Capability detection tests** — Auto-detect returns correct flags for known model names
6. **Fallback tests** — Vision model gets image block; text-only model gets text_fallback.
   Same for PDFs across all three providers.
7. **Token estimation tests** — Images report correct token costs; context window eviction
   respects multimodal ordering.
8. **End-to-end test** — read_file on a test image → tool result → provider conversion →
   verify the API payload contains the base64 data in the correct format.

---

## Dependencies

**Required (already available):**
- `pypdf` — PDF text extraction (existing)
- `base64`, `struct`, `json` — stdlib

**Optional (graceful degradation):**
- `Pillow` — Image resize when oversized. Without it: images >5MB rejected, no resize.
- `pymupdf` or `pdf2image` — PDF page rendering for Ollama vision. Without them: text
  extraction only for Ollama, even with vision models. Anthropic/OpenAI use native PDF
  blocks and don't need rendering.

**Required (newly added):**
- `python-docx` — Word document (.docx) text extraction (required dependency)
- `python-pptx` — PowerPoint (.pptx) text extraction (required dependency)

**NOT required:**
- No other new hard dependencies. Remaining multimodal features degrade gracefully.

---

## Implemented: Office Document Support

Word (.docx/.doc) and PowerPoint (.pptx/.ppt) support has been implemented:

- `content_utils.py` — `extract_docx_text()` and `extract_pptx_text()` functions
- `file_ops.py` — `ReadFileTool._read_docx()` and `_read_pptx()` methods
- Both return `DocumentBlock` with extracted text as `text_fallback`
- 64MB file size limit for office documents
- `python-docx` and `python-pptx` are required dependencies (not optional)
- 16 tests covering extraction, integration, empty docs, corrupt files, and legacy extensions

---

## Open Questions

1. **Should SVG files be rasterized?** Currently planned as text fallback (raw XML or
   metadata). Could render with cairosvg but adds a dependency for an edge case.

2. **Image caching?** If the same image is referenced in multiple tool results across a
   conversation, should we cache the base64 encoding? Probably not worth the complexity
   — base64 encoding is fast and the files are already on disk.

3. **User-facing image display in TUI?** Terminal image protocols (iTerm2, Kitty) could
   show thumbnails inline. Worth doing but only as Phase 4+ polish.

4. **Audio support?** OpenAI supports audio input/output. Ollama has open feature requests.
   Not relevant for a code assistant today, but the ContentBlock architecture supports
   adding AudioBlock later without structural changes.
