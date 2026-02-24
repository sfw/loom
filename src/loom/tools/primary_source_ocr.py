"""Primary-source OCR tool (local, keyless)."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loom.tools.registry import Tool, ToolContext, ToolResult

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
_PDF_SUFFIX = ".pdf"
_WS_RE = re.compile(r"[ \t]+")
_BLANK_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class OcrPage:
    page: int
    text: str
    engine: str
    confidence: float | None = None
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "chars": len(self.text),
            "engine": self.engine,
            "confidence": self.confidence,
            "warning": self.warning,
        }


class PrimarySourceOcrTool(Tool):
    """Extract OCR text from scanned PDFs/images without API keys."""

    @property
    def name(self) -> str:
        return "primary_source_ocr"

    @property
    def description(self) -> str:
        return (
            "Extract text from scanned images/PDFs using local OCR engines "
            "(Tesseract/OCRmyPDF when available)."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Input image or PDF path.",
                },
                "language": {
                    "type": "string",
                    "description": "OCR language code(s) (for example eng, eng+deu).",
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Maximum pages to process for PDFs.",
                },
                "force_ocr": {
                    "type": "boolean",
                    "description": "Force OCR even if embedded PDF text exists.",
                },
                "cleanup": {
                    "type": "string",
                    "enum": ["none", "light", "llm_cleanup"],
                    "description": "Post-processing cleanup mode.",
                },
                "make_searchable_pdf": {
                    "type": "boolean",
                    "description": "Run OCRmyPDF when available and input is PDF.",
                },
                "searchable_output_path": {
                    "type": "string",
                    "description": "Output path for searchable PDF (OCRmyPDF mode).",
                },
                "output_text_path": {
                    "type": "string",
                    "description": "Optional path to write extracted text report.",
                },
            },
            "required": ["path"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 90

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.workspace is None:
            return ToolResult.fail("No workspace set")

        path_text = str(args.get("path", "")).strip()
        if not path_text:
            return ToolResult.fail("path is required")

        path = self._resolve_read_path(path_text, ctx.workspace, ctx.read_roots)
        if not path.exists() or not path.is_file():
            return ToolResult.fail(f"File not found: {path_text}")

        language = str(args.get("language", "eng")).strip() or "eng"
        max_pages = _clamp_int(args.get("max_pages"), default=25, lo=1, hi=400)
        force_ocr = bool(args.get("force_ocr", False))
        cleanup = str(args.get("cleanup", "light")).strip().lower() or "light"
        if cleanup not in {"none", "light", "llm_cleanup"}:
            return ToolResult.fail("cleanup must be none/light/llm_cleanup")

        make_searchable_pdf = bool(args.get("make_searchable_pdf", False))
        searchable_output_path = str(args.get("searchable_output_path", "")).strip()
        output_text_path = str(args.get("output_text_path", "")).strip()

        pages: list[OcrPage] = []
        warnings: list[str] = []
        engine_used = ""

        suffix = path.suffix.lower()
        if suffix in _IMAGE_SUFFIXES:
            if _which("tesseract") is None:
                return ToolResult.fail("Tesseract is not installed (required for image OCR)")
            page = await _ocr_image(path, language=language)
            pages = [page]
            engine_used = page.engine
        elif suffix == _PDF_SUFFIX:
            if not force_ocr:
                embedded_pages = _extract_embedded_pdf_text(path=path, max_pages=max_pages)
                if embedded_pages:
                    pages = embedded_pages
                    engine_used = "embedded_pdf_text"
                else:
                    warnings.append("No embedded PDF text detected; switching to OCR.")

            if not pages:
                if _which("tesseract") is None:
                    return ToolResult.fail(
                        "Tesseract is not installed and PDF has no extractable embedded text."
                    )
                ocr_pages, ocr_engine, ocr_warnings = await _ocr_pdf(
                    path=path,
                    language=language,
                    max_pages=max_pages,
                )
                pages = ocr_pages
                engine_used = ocr_engine
                warnings.extend(ocr_warnings)

            if make_searchable_pdf:
                if _which("ocrmypdf") is None:
                    warnings.append(
                        "make_searchable_pdf requested but ocrmypdf is not installed."
                    )
                elif searchable_output_path:
                    out = self._resolve_path(searchable_output_path, ctx.workspace)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    searchable_warning = await _run_ocrmypdf(path, out, language=language)
                    if searchable_warning:
                        warnings.append(searchable_warning)
                    else:
                        warnings.append(f"Searchable PDF written: {out.relative_to(ctx.workspace)}")
        else:
            return ToolResult.fail("Unsupported file type. Use image or PDF input.")

        raw_text = "\n\n".join(page.text for page in pages).strip()
        if cleanup == "light":
            raw_text = _light_cleanup(raw_text)
        elif cleanup == "llm_cleanup":
            # Keep deterministic behavior in-tool; downstream process can invoke
            # model cleanup explicitly if needed.
            raw_text = _light_cleanup(raw_text)
            warnings.append("llm_cleanup requested; applied deterministic light cleanup in-tool.")

        payload = {
            "path": str(path),
            "engine": engine_used,
            "language": language,
            "page_count": len(pages),
            "pages": [page.to_dict() for page in pages],
            "text": raw_text,
            "warnings": warnings,
            "keyless": True,
        }

        files_changed: list[str] = []
        if output_text_path:
            out = self._resolve_path(output_text_path, ctx.workspace)
            out.parent.mkdir(parents=True, exist_ok=True)
            if ctx.changelog is not None:
                ctx.changelog.record_before_write(str(out), subtask_id=ctx.subtask_id)
            lines = [
                "# Primary Source OCR",
                "",
                f"- **Input**: {path.name}",
                f"- **Engine**: {engine_used}",
                f"- **Language**: {language}",
                f"- **Pages**: {len(pages)}",
                "",
                "## Extracted Text",
                "",
                raw_text,
                "",
                "## Metadata",
                "",
                "```json",
                json.dumps(payload, indent=2),
                "```",
                "",
            ]
            out.write_text("\n".join(lines), encoding="utf-8")
            files_changed.append(str(out.relative_to(ctx.workspace)))

        summary_lines = [
            (
                f"OCR complete: {path.name} -> {len(raw_text)} chars across "
                f"{len(pages)} page(s) using {engine_used or 'unknown_engine'}."
            )
        ]
        if files_changed:
            summary_lines.append("Artifact: " + ", ".join(files_changed))
        if warnings:
            summary_lines.append("Warnings: " + "; ".join(warnings))

        return ToolResult.ok(
            "\n".join(summary_lines),
            data=payload,
            files_changed=files_changed,
        )


def _which(name: str) -> str | None:
    return shutil.which(name)


def _clamp_int(value: object, *, default: int, lo: int, hi: int) -> int:
    try:
        if value is None:
            parsed = default
        else:
            parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lo, min(hi, parsed))


def _light_cleanup(text: str) -> str:
    out_lines: list[str] = []
    for line in text.splitlines():
        out_lines.append(_WS_RE.sub(" ", line).strip())
    cleaned = "\n".join(out_lines)
    cleaned = _BLANK_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def _extract_embedded_pdf_text(*, path: Path, max_pages: int) -> list[OcrPage]:
    try:
        import pypdf
    except Exception:
        return []

    try:
        reader = pypdf.PdfReader(path)
    except Exception:
        return []

    pages: list[OcrPage] = []
    for idx, page in enumerate(reader.pages[:max_pages], start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            pages.append(OcrPage(page=idx, text=text, engine="embedded_pdf_text"))
    return pages


async def _ocr_image(path: Path, *, language: str) -> OcrPage:
    text, warning = await _run_tesseract(path, language=language)
    return OcrPage(page=1, text=text, engine="tesseract", warning=warning)


async def _ocr_pdf(
    *,
    path: Path,
    language: str,
    max_pages: int,
) -> tuple[list[OcrPage], str, list[str]]:
    warnings: list[str] = []
    if _which("pdftoppm") is not None:
        pages = await _ocr_pdf_via_pdftoppm(path=path, language=language, max_pages=max_pages)
        if pages:
            return pages, "tesseract+pdftoppm", warnings
        warnings.append("pdftoppm pipeline produced no OCR text; trying direct tesseract.")

    text, warning = await _run_tesseract(path, language=language)
    if warning:
        warnings.append(warning)
    if not text.strip():
        warnings.append("Direct tesseract PDF OCR returned empty text.")
    return [OcrPage(page=1, text=text, engine="tesseract", warning=warning)], "tesseract", warnings


async def _ocr_pdf_via_pdftoppm(
    *,
    path: Path,
    language: str,
    max_pages: int,
) -> list[OcrPage]:
    pages: list[OcrPage] = []
    with tempfile.TemporaryDirectory(prefix="loom-ocr-") as tmp:
        prefix = Path(tmp) / "page"
        cmd = ["pdftoppm", "-png", str(path), str(prefix)]
        await _run_command(cmd, timeout=60)
        images = sorted(Path(tmp).glob("page-*.png"))
        for idx, img in enumerate(images[:max_pages], start=1):
            text, warning = await _run_tesseract(img, language=language)
            pages.append(
                OcrPage(
                    page=idx,
                    text=text,
                    engine="tesseract+pdftoppm",
                    warning=warning,
                )
            )
    return pages


async def _run_tesseract(path: Path, *, language: str) -> tuple[str, str]:
    cmd = ["tesseract", str(path), "stdout", "-l", language]
    completed = await _run_command(cmd, timeout=75)
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    warning = ""
    if completed.returncode != 0:
        warning = stderr or "tesseract returned non-zero exit status"
    elif stderr:
        warning = stderr
    return stdout, warning


async def _run_ocrmypdf(src: Path, dst: Path, *, language: str) -> str:
    cmd = [
        "ocrmypdf",
        "--skip-text",
        "-l",
        language,
        str(src),
        str(dst),
    ]
    completed = await _run_command(cmd, timeout=120)
    if completed.returncode != 0:
        return (completed.stderr or completed.stdout or "ocrmypdf failed").strip()
    return ""


async def _run_command(cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return await asyncio.to_thread(
        subprocess.run,
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
