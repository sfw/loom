"""Filetype handlers for artifact summarization/extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loom.content_utils import extract_docx_text, extract_pptx_text, get_image_dimensions
from loom.ingest.router import ContentKind


@dataclass(frozen=True)
class ArtifactSummary:
    """Summarized artifact payload for tool output/context."""

    handler: str
    summary_text: str
    extracted_text: str = ""
    extraction_truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ArtifactHandler:
    """Base contract for artifact handlers."""

    name = "base"

    def can_handle(self, *, content_kind: str, media_type: str, path: Path) -> bool:
        del content_kind, media_type, path
        return False

    def extract(
        self,
        *,
        path: Path,
        content_kind: str,
        media_type: str,
        max_chars: int,
    ) -> ArtifactSummary:
        del content_kind, media_type, max_chars
        size = path.stat().st_size if path.exists() else 0
        return ArtifactSummary(
            handler=self.name,
            summary_text=f"[Fetched artifact: {path.name}, {size:,} bytes]",
            metadata={"size_bytes": size},
        )


class PdfHandler(ArtifactHandler):
    name = "pdf_handler"
    _MAX_PAGES = 20

    def can_handle(self, *, content_kind: str, media_type: str, path: Path) -> bool:
        if content_kind == ContentKind.PDF:
            return True
        if str(media_type or "").lower().startswith("application/pdf"):
            return True
        return path.suffix.lower() == ".pdf"

    def extract(
        self,
        *,
        path: Path,
        content_kind: str,
        media_type: str,
        max_chars: int,
    ) -> ArtifactSummary:
        del content_kind, media_type
        size = path.stat().st_size if path.exists() else 0
        try:
            import pypdf
        except Exception:
            return ArtifactSummary(
                handler=self.name,
                summary_text=(
                    f"[Fetched PDF artifact: {path.name}, {size:,} bytes]\n"
                    "PDF text extraction unavailable (missing pypdf)."
                ),
                metadata={"size_bytes": size, "extractable": False, "reason": "missing_pypdf"},
            )

        try:
            reader = pypdf.PdfReader(path)
        except Exception as exc:
            return ArtifactSummary(
                handler=self.name,
                summary_text=(
                    f"[Fetched PDF artifact: {path.name}, {size:,} bytes]\n"
                    f"PDF extraction failed: {exc}"
                ),
                metadata={"size_bytes": size, "extractable": False, "reason": "read_error"},
            )

        total_pages = len(reader.pages)
        pages_to_read = min(total_pages, self._MAX_PAGES)
        chunks: list[str] = []
        for idx in range(pages_to_read):
            try:
                text = reader.pages[idx].extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                chunks.append(f"--- Page {idx + 1} ---\n{text.strip()}")

        extracted = "\n\n".join(chunks).strip()
        truncated = False
        if max_chars > 0 and len(extracted) > max_chars:
            extracted = extracted[:max_chars].rstrip()
            truncated = True
        if pages_to_read < total_pages:
            truncated = True

        lines = [f"[Fetched PDF artifact: {path.name}, {size:,} bytes, {total_pages} pages]"]
        if extracted:
            lines.append("")
            lines.append(extracted)
            if truncated:
                lines.append("")
                lines.append(
                    "[PDF extraction truncated for context safety; "
                    "use targeted follow-up reads.]",
                )
        else:
            lines.append("No extractable PDF text found.")
        summary = "\n".join(lines).strip()
        return ArtifactSummary(
            handler=self.name,
            summary_text=summary,
            extracted_text=extracted,
            extraction_truncated=truncated,
            metadata={
                "size_bytes": size,
                "total_pages": total_pages,
                "pages_read": pages_to_read,
            },
        )


class OfficeDocumentHandler(ArtifactHandler):
    name = "office_doc_handler"

    def can_handle(self, *, content_kind: str, media_type: str, path: Path) -> bool:
        if content_kind == ContentKind.OFFICE_DOC:
            return True
        suffix = path.suffix.lower()
        if suffix in {".doc", ".docx", ".ppt", ".pptx"}:
            return True
        media = str(media_type or "").lower()
        return "wordprocessingml.document" in media or "presentationml.presentation" in media

    def extract(
        self,
        *,
        path: Path,
        content_kind: str,
        media_type: str,
        max_chars: int,
    ) -> ArtifactSummary:
        del content_kind, media_type
        suffix = path.suffix.lower()
        size = path.stat().st_size if path.exists() else 0
        extracted = ""
        handler_error = ""
        try:
            if suffix in {".doc", ".docx"}:
                extracted = extract_docx_text(path)
            elif suffix in {".ppt", ".pptx"}:
                extracted = extract_pptx_text(path)
        except Exception as exc:
            handler_error = str(exc)

        truncated = False
        if max_chars > 0 and len(extracted) > max_chars:
            extracted = extracted[:max_chars].rstrip()
            truncated = True

        if extracted:
            summary = (
                f"[Fetched office artifact: {path.name}, {size:,} bytes]\n\n"
                f"{extracted}"
            )
            if truncated:
                summary += "\n\n[Office extraction truncated for context safety.]"
        elif handler_error:
            summary = (
                f"[Fetched office artifact: {path.name}, {size:,} bytes]\n"
                f"Office extraction failed: {handler_error}"
            )
        else:
            summary = (
                f"[Fetched office artifact: {path.name}, {size:,} bytes]\n"
                "No extractable text found."
            )

        return ArtifactSummary(
            handler=self.name,
            summary_text=summary,
            extracted_text=extracted,
            extraction_truncated=truncated,
            metadata={"size_bytes": size, "suffix": suffix, "error": handler_error},
        )


class ImageHandler(ArtifactHandler):
    name = "image_handler"

    def can_handle(self, *, content_kind: str, media_type: str, path: Path) -> bool:
        if content_kind == ContentKind.IMAGE:
            return True
        if str(media_type or "").lower().startswith("image/"):
            return True
        return path.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".webp",
            ".tif",
            ".tiff",
        }

    def extract(
        self,
        *,
        path: Path,
        content_kind: str,
        media_type: str,
        max_chars: int,
    ) -> ArtifactSummary:
        del content_kind, media_type, max_chars
        size = path.stat().st_size if path.exists() else 0
        width, height = get_image_dimensions(path)
        summary = (
            f"[Fetched image artifact: {path.name}, {size:,} bytes, "
            f"{width}x{height}]"
        )
        return ArtifactSummary(
            handler=self.name,
            summary_text=summary,
            metadata={"size_bytes": size, "width": width, "height": height},
        )


class BinaryFallbackHandler(ArtifactHandler):
    name = "binary_fallback_handler"

    def can_handle(self, *, content_kind: str, media_type: str, path: Path) -> bool:
        del content_kind, media_type, path
        return True

    def extract(
        self,
        *,
        path: Path,
        content_kind: str,
        media_type: str,
        max_chars: int,
    ) -> ArtifactSummary:
        del max_chars
        size = path.stat().st_size if path.exists() else 0
        label = content_kind or ContentKind.UNKNOWN_BINARY
        media = media_type or "application/octet-stream"
        summary = (
            f"[Fetched {label} artifact: {path.name}, {size:,} bytes, "
            f"media_type={media}]"
        )
        return ArtifactSummary(
            handler=self.name,
            summary_text=summary,
            metadata={"size_bytes": size, "media_type": media, "content_kind": label},
        )


_HANDLERS: tuple[ArtifactHandler, ...] = (
    PdfHandler(),
    OfficeDocumentHandler(),
    ImageHandler(),
    BinaryFallbackHandler(),
)


def summarize_artifact(
    *,
    path: Path,
    content_kind: str,
    media_type: str,
    max_chars: int = 3200,
) -> ArtifactSummary:
    """Route to the first handler that can summarize the artifact."""
    for handler in _HANDLERS:
        if handler.can_handle(content_kind=content_kind, media_type=media_type, path=path):
            return handler.extract(
                path=path,
                content_kind=content_kind,
                media_type=media_type,
                max_chars=max_chars,
            )
    return BinaryFallbackHandler().extract(
        path=path,
        content_kind=content_kind,
        media_type=media_type,
        max_chars=max_chars,
    )
