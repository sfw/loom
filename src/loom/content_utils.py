"""Image and document processing utilities for multimodal content.

Provides base64 encoding, image dimension parsing, resize support,
and PDF-to-image rendering. All heavy dependencies (Pillow, pymupdf)
are optional — functions degrade gracefully when unavailable.
"""

from __future__ import annotations

import base64
import struct
from pathlib import Path

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5MB per image
MAX_PDF_BYTES = 32 * 1024 * 1024  # 32MB per PDF
IMAGE_MAX_DIMENSION = 2048


def encode_image_base64(path: Path, max_dimension: int = IMAGE_MAX_DIMENSION) -> str | None:
    """Read and base64-encode an image file.

    If the image exceeds max_dimension on either axis and Pillow is
    available, resize it proportionally. Without Pillow, returns the
    raw image if within the 5MB limit, or None if too large.
    """
    if not path.exists():
        return None

    size = path.stat().st_size
    if size > MAX_IMAGE_BYTES:
        try:
            return _resize_and_encode(path, max_dimension)
        except ImportError:
            return None

    data = path.read_bytes()

    w, h = get_image_dimensions_from_bytes(data[:64], path.suffix)
    if (w > max_dimension or h > max_dimension) and w > 0:
        try:
            return _resize_and_encode(path, max_dimension)
        except ImportError:
            pass  # Send as-is, let the API handle it

    return base64.b64encode(data).decode("ascii")


def encode_file_base64(path: Path) -> str | None:
    """Read and base64-encode any file (used for PDFs)."""
    if not path.exists():
        return None
    try:
        data = path.read_bytes()
        return base64.b64encode(data).decode("ascii")
    except Exception:
        return None


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
    # PNG
    if header[:8] == b"\x89PNG\r\n\x1a\n" and len(header) >= 24:
        w, h = struct.unpack(">II", header[16:24])
        return (w, h)
    # JPEG — complex header; skip for now
    if header[:2] == b"\xff\xd8":
        return (0, 0)
    # GIF
    if header[:6] in (b"GIF87a", b"GIF89a") and len(header) >= 10:
        w, h = struct.unpack("<HH", header[6:10])
        return (w, h)
    # BMP
    if header[:2] == b"BM" and len(header) >= 26:
        w, h = struct.unpack("<II", header[18:26])
        return (w, h)
    # WebP
    if header[:4] == b"RIFF" and len(header) >= 16 and header[8:12] == b"WEBP":
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
    path: Path,
    page_range: tuple[int, int] | None = None,
    dpi: int = 150,
    max_dimension: int = IMAGE_MAX_DIMENSION,
) -> list[str]:
    """Render PDF pages as base64-encoded images.

    Used for Ollama (no native PDF support) with vision-capable models.
    Requires pymupdf or pdf2image. Returns empty list if neither is available.
    """
    try:
        import fitz

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
        import io

        images = []
        for page in pages:
            page.thumbnail((max_dimension, max_dimension))
            buf = io.BytesIO()
            page.save(buf, format="PNG")
            images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        return images
    except ImportError:
        return []
