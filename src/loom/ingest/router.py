"""Deterministic content-kind detection for fetched payloads."""

from __future__ import annotations

import mimetypes
import re
from urllib.parse import urlparse


class ContentKind:
    """Logical payload classes used by ingest handlers."""

    TEXT = "text"
    HTML = "html"
    PDF = "pdf"
    OFFICE_DOC = "office_doc"
    IMAGE = "image"
    ARCHIVE = "archive"
    UNKNOWN_BINARY = "unknown_binary"


_HTML_MIME_MARKERS = ("text/html", "application/xhtml+xml")
_TEXT_MIME_PREFIXES = ("text/",)
_TEXT_MIME_EXACT = frozenset({
    "application/json",
    "application/ld+json",
    "application/xml",
    "application/javascript",
    "application/x-javascript",
    "application/x-www-form-urlencoded",
    "application/yaml",
    "application/x-yaml",
    "application/toml",
    "application/x-toml",
    "application/csv",
})
_PDF_MIME_TYPES = frozenset({"application/pdf", "application/x-pdf"})
_OFFICE_MIME_TYPES = frozenset({
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.oasis.opendocument.text",
    "application/vnd.oasis.opendocument.presentation",
    "application/vnd.oasis.opendocument.spreadsheet",
})
_ARCHIVE_MIME_TYPES = frozenset({
    "application/zip",
    "application/x-zip-compressed",
    "application/x-tar",
    "application/gzip",
    "application/x-gzip",
    "application/x-7z-compressed",
    "application/x-rar-compressed",
    "application/octet-stream",
})
_OFFICE_EXTENSIONS = frozenset({
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".odt",
    ".odp",
    ".ods",
})
_ARCHIVE_EXTENSIONS = frozenset({
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
})
_IMAGE_EXTENSIONS = frozenset({
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".svg",
    ".ico",
})
_TEXT_EXTENSIONS = frozenset({
    ".txt",
    ".md",
    ".rst",
    ".csv",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".log",
})
_HTML_EXTENSIONS = frozenset({".html", ".htm", ".xhtml"})
_MAGIC_PDF = b"%PDF-"
_MAGIC_PNG = b"\x89PNG\r\n\x1a\n"
_MAGIC_JPEG = b"\xff\xd8\xff"
_MAGIC_GIF = (b"GIF87a", b"GIF89a")
_MAGIC_BMP = b"BM"
_MAGIC_ZIP = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_MAGIC_GZIP = b"\x1f\x8b\x08"
_MAGIC_RAR = (b"Rar!\x1a\x07\x00", b"Rar!\x1a\x07\x01\x00")
_MAGIC_7Z = b"7z\xbc\xaf\x27\x1c"


def normalize_media_type(content_type: str | None) -> str:
    """Normalize HTTP content-type header to a bare media type."""
    raw = str(content_type or "").strip().lower()
    if not raw:
        return ""
    return raw.split(";", 1)[0].strip()


def _kind_from_mime(media_type: str) -> str | None:
    if not media_type:
        return None
    if any(marker in media_type for marker in _HTML_MIME_MARKERS):
        return ContentKind.HTML
    if media_type in _PDF_MIME_TYPES:
        return ContentKind.PDF
    if media_type.startswith("image/"):
        return ContentKind.IMAGE
    if media_type in _OFFICE_MIME_TYPES:
        return ContentKind.OFFICE_DOC
    if media_type.startswith(_TEXT_MIME_PREFIXES) or media_type in _TEXT_MIME_EXACT:
        return ContentKind.TEXT
    if media_type in _ARCHIVE_MIME_TYPES:
        return ContentKind.ARCHIVE
    return None


def _kind_from_magic(content_bytes: bytes) -> str | None:
    sample = bytes(content_bytes[:4096] if content_bytes else b"")
    if not sample:
        return None
    if sample.startswith(_MAGIC_PDF):
        return ContentKind.PDF
    if sample.startswith(_MAGIC_PNG):
        return ContentKind.IMAGE
    if sample.startswith(_MAGIC_JPEG):
        return ContentKind.IMAGE
    if any(sample.startswith(sig) for sig in _MAGIC_GIF):
        return ContentKind.IMAGE
    if sample.startswith(_MAGIC_BMP):
        return ContentKind.IMAGE
    if sample[:4] == b"RIFF" and sample[8:12] == b"WEBP":
        return ContentKind.IMAGE
    if any(sample.startswith(sig) for sig in _MAGIC_ZIP):
        return ContentKind.ARCHIVE
    if sample.startswith(_MAGIC_GZIP):
        return ContentKind.ARCHIVE
    if any(sample.startswith(sig) for sig in _MAGIC_RAR):
        return ContentKind.ARCHIVE
    if sample.startswith(_MAGIC_7Z):
        return ContentKind.ARCHIVE

    # Lightweight HTML sniff when servers mislabel pages.
    try:
        snippet = sample[:2048].decode("utf-8", errors="ignore").lower()
    except Exception:
        return None
    if "<!doctype html" in snippet:
        return ContentKind.HTML
    if re.search(r"<html(?:\s|>)", snippet):
        return ContentKind.HTML
    return None


def _kind_from_url(url: str) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    path = parsed.path or ""
    if "." not in path:
        return None
    suffix = "." + path.rsplit(".", 1)[-1].lower()
    if suffix in _HTML_EXTENSIONS:
        return ContentKind.HTML
    if suffix in _TEXT_EXTENSIONS:
        return ContentKind.TEXT
    if suffix == ".pdf":
        return ContentKind.PDF
    if suffix in _OFFICE_EXTENSIONS:
        return ContentKind.OFFICE_DOC
    if suffix in _IMAGE_EXTENSIONS:
        return ContentKind.IMAGE
    if suffix in _ARCHIVE_EXTENSIONS:
        return ContentKind.ARCHIVE
    guessed, _enc = mimetypes.guess_type(path)
    guessed_kind = _kind_from_mime(normalize_media_type(guessed))
    if guessed_kind:
        return guessed_kind
    return None


def detect_content_kind(
    *,
    content_type: str | None,
    content_bytes: bytes,
    url: str = "",
) -> str:
    """Classify payload by MIME, bytes, and URL hints."""
    media_type = normalize_media_type(content_type)
    kind_mime = _kind_from_mime(media_type)
    kind_magic = _kind_from_magic(content_bytes)
    kind_url = _kind_from_url(url)

    # MIME signal is strongest when specific and non-generic.
    if kind_mime and media_type not in {"application/octet-stream", "*/*"}:
        if kind_mime == ContentKind.TEXT and kind_magic == ContentKind.HTML:
            return ContentKind.HTML
        # Allow magic/extension to upgrade generic zip to office docs.
        if (
            kind_mime == ContentKind.ARCHIVE
            and (kind_url == ContentKind.OFFICE_DOC or kind_magic == ContentKind.OFFICE_DOC)
        ):
            return ContentKind.OFFICE_DOC
        return kind_mime

    # Byte signature has next priority.
    if kind_magic:
        if kind_magic == ContentKind.ARCHIVE and kind_url == ContentKind.OFFICE_DOC:
            return ContentKind.OFFICE_DOC
        return kind_magic

    # URL extension hint as fallback.
    if kind_url:
        return kind_url

    # Unknown defaults to binary-safe handling.
    return ContentKind.UNKNOWN_BINARY
