"""File operation tools: read, write, edit, delete, move files."""

from __future__ import annotations

import difflib
from pathlib import Path

from loom.content import (
    IMAGE_MEDIA_TYPES,
    DocumentBlock,
    ImageBlock,
)
from loom.content_utils import extract_docx_text, extract_pptx_text, get_image_dimensions
from loom.tools.code_analysis import detect_language
from loom.tools.registry import Tool, ToolContext, ToolResult
from loom.utils.concurrency import run_blocking_io

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg"})
_PDF_EXTENSION = ".pdf"
_DOCX_EXTENSIONS = frozenset({".doc", ".docx"})
_PPTX_EXTENSIONS = frozenset({".ppt", ".pptx"})

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5MB
MAX_PDF_BYTES = 32 * 1024 * 1024  # 32MB
MAX_OFFICE_BYTES = 64 * 1024 * 1024  # 64MB
MAX_PDF_PAGES_PER_READ = 20


class ReadFileTool(Tool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file. Optionally specify a line range. "
            "Supports text files, PDFs (with pagination), Word documents "
            "(.doc/.docx), PowerPoint presentations (.ppt/.pptx), and image "
            "files (returned as multimodal content for vision-capable models)."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace"},
                "line_start": {"type": "integer", "description": "Start line (1-based, optional)"},
                "line_end": {"type": "integer", "description": "End line (inclusive, optional)"},
                "page_start": {
                    "type": "integer",
                    "description": "PDF: first page (0-indexed)",
                },
                "page_end": {
                    "type": "integer",
                    "description": "PDF: last page (exclusive)",
                },
            },
            "required": ["path"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 10

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        return await run_blocking_io(self._execute_sync, args, ctx)

    def _execute_sync(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.workspace is None:
            return ToolResult.fail("No workspace set")

        path = self._resolve_read_path(
            args["path"],
            ctx.workspace,
            ctx.read_roots,
        )
        if not path.exists():
            return ToolResult.fail(f"File not found: {args['path']}")
        if not path.is_file():
            return ToolResult.fail(f"Not a file: {args['path']}")

        suffix = path.suffix.lower()

        # PDF handling
        if suffix == _PDF_EXTENSION:
            return self._read_pdf(
                path,
                page_start=args.get("page_start", 0),
                page_end=args.get("page_end"),
            )

        # Word document handling
        if suffix in _DOCX_EXTENSIONS:
            return self._read_docx(path)

        # PowerPoint handling
        if suffix in _PPTX_EXTENSIONS:
            return self._read_pptx(path)

        # Image handling
        if suffix in _IMAGE_EXTENSIONS:
            return self._read_image(path)

        # Text file
        content = path.read_text(encoding="utf-8", errors="replace")

        line_start = args.get("line_start")
        line_end = args.get("line_end")
        if line_start is not None or line_end is not None:
            lines = content.splitlines(keepends=True)
            start = (line_start or 1) - 1
            end = line_end or len(lines)
            content = "".join(lines[start:end])

        return ToolResult.ok(content)

    @staticmethod
    def _read_pdf(
        path: Path,
        page_start: int = 0,
        page_end: int | None = None,
    ) -> ToolResult:
        """Read PDF with pagination and multimodal output."""
        size = path.stat().st_size

        if size > MAX_PDF_BYTES:
            return ToolResult.fail(
                f"PDF too large: {size:,} bytes "
                f"(limit: {MAX_PDF_BYTES:,} bytes)"
            )

        try:
            import pypdf
        except ImportError:
            return ToolResult.ok(
                f"[PDF file: {path.name}, {size:,} bytes]\n"
                "Install 'pypdf' to extract text: pip install pypdf",
                data={"type": "pdf", "name": path.name, "size": size},
            )

        try:
            reader = pypdf.PdfReader(path)
        except Exception as e:
            return ToolResult.fail(f"Error reading PDF: {e}")

        total_pages = len(reader.pages)

        if page_end is None:
            page_end = min(page_start + MAX_PDF_PAGES_PER_READ, total_pages)
        page_end = min(page_end, total_pages)
        page_start = max(0, page_start)

        pages_text = []
        for i in range(page_start, page_end):
            try:
                text = reader.pages[i].extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                pages_text.append(f"--- Page {i + 1} ---\n{text}")

        extracted = "\n\n".join(pages_text) if pages_text else ""
        pagination_note = ""
        if page_end < total_pages:
            pagination_note = (
                f"\n\n[Showing pages {page_start + 1}-{page_end} of "
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
    def _read_image(path: Path) -> ToolResult:
        """Return image as a content block with text fallback."""
        size = path.stat().st_size

        if size > MAX_IMAGE_BYTES:
            return ToolResult.ok(
                f"[Image too large: {path.name}, {size:,} bytes, "
                f"limit is {MAX_IMAGE_BYTES:,} bytes. "
                f"Consider resizing or converting to JPEG.]"
            )

        suffix = path.suffix.lower()
        media_type = IMAGE_MEDIA_TYPES.get(suffix)
        if not media_type:
            # SVG and other unsupported formats get text fallback only
            return ToolResult.ok(
                f"[Image file: {path.name}, {size:,} bytes, type: {suffix}]",
                data={"type": "image", "name": path.name,
                      "size": size, "format": suffix},
            )

        width, height = get_image_dimensions(path)

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


    @staticmethod
    def _read_docx(path: Path) -> ToolResult:
        """Read a Word document and return extracted text."""
        size = path.stat().st_size
        if size > MAX_OFFICE_BYTES:
            return ToolResult.fail(
                f"Word document too large: {size:,} bytes "
                f"(limit: {MAX_OFFICE_BYTES:,} bytes)"
            )

        try:
            text = extract_docx_text(path)
        except Exception as e:
            return ToolResult.fail(f"Error reading Word document: {e}")

        if not text.strip():
            text_fallback = f"[Word document: {path.name}, {size:,} bytes, no extractable text]"
        else:
            text_fallback = text

        block = DocumentBlock(
            source_path=str(path),
            media_type=(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
            size_bytes=size,
            extracted_text=text,
            text_fallback=text_fallback,
        )

        return ToolResult.multimodal(
            output=text_fallback,
            blocks=[block],
            data={"type": "docx", "name": path.name, "size": size},
        )

    @staticmethod
    def _read_pptx(path: Path) -> ToolResult:
        """Read a PowerPoint presentation and return extracted text."""
        size = path.stat().st_size
        if size > MAX_OFFICE_BYTES:
            return ToolResult.fail(
                f"PowerPoint file too large: {size:,} bytes "
                f"(limit: {MAX_OFFICE_BYTES:,} bytes)"
            )

        try:
            text = extract_pptx_text(path)
        except Exception as e:
            return ToolResult.fail(f"Error reading PowerPoint file: {e}")

        if not text.strip():
            text_fallback = (
                f"[PowerPoint: {path.name}, {size:,} bytes, no extractable text]"
            )
        else:
            text_fallback = text

        block = DocumentBlock(
            source_path=str(path),
            media_type=(
                "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            ),
            size_bytes=size,
            extracted_text=text,
            text_fallback=text_fallback,
        )

        return ToolResult.multimodal(
            output=text_fallback,
            blocks=[block],
            data={"type": "pptx", "name": path.name, "size": size},
        )


class WriteFileTool(Tool):
    MAX_CONTENT_SIZE = 1_048_576  # 1 MB limit for written content

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def is_mutating(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return "Write content to a file. Creates parent directories if needed. Max 1 MB."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace"},
                "content": {"type": "string", "description": "Content to write (max 1 MB)"},
            },
            "required": ["path", "content"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 10

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.workspace is None:
            return ToolResult.fail("No workspace set")

        content = args.get("content", "")
        if len(content) > self.MAX_CONTENT_SIZE:
            return ToolResult.fail(
                f"Content too large ({len(content):,} bytes). "
                f"Maximum is {self.MAX_CONTENT_SIZE:,} bytes."
            )

        path = self._resolve_path(args["path"], ctx.workspace)
        rel_path = str(path.relative_to(ctx.workspace.resolve()))

        # Record in changelog before writing
        if ctx.changelog is not None:
            ctx.changelog.record_before_write(rel_path, subtask_id=ctx.subtask_id)

        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(content, encoding="utf-8")

        return ToolResult.ok(
            f"Wrote {len(content)} bytes to {rel_path}",
            files_changed=[rel_path],
        )


class DeleteFileTool(Tool):
    @property
    def name(self) -> str:
        return "delete_file"

    @property
    def is_mutating(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return "Delete a file or empty directory. Cannot delete workspace root or .git/."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace"},
            },
            "required": ["path"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 10

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.workspace is None:
            return ToolResult.fail("No workspace set")

        path = self._resolve_path(args["path"], ctx.workspace)
        rel_path = str(path.relative_to(ctx.workspace.resolve()))

        # Safety: refuse to delete workspace root
        if path.resolve() == ctx.workspace.resolve():
            return ToolResult.fail("Cannot delete workspace root")

        # Safety: refuse to delete .git directory
        if ".git" in path.parts:
            return ToolResult.fail("Cannot delete .git directory or its contents")

        if not path.exists():
            return ToolResult.fail(f"Path not found: {args['path']}")

        # Record in changelog before deleting
        if ctx.changelog is not None and path.is_file():
            ctx.changelog.record_delete(rel_path, subtask_id=ctx.subtask_id)

        if path.is_file():
            path.unlink()
            return ToolResult.ok(f"Deleted file: {rel_path}", files_changed=[rel_path])
        elif path.is_dir():
            try:
                path.rmdir()  # Only removes empty directories
                return ToolResult.ok(f"Deleted empty directory: {rel_path}")
            except OSError:
                return ToolResult.fail(
                    f"Directory not empty: {rel_path}. Remove contents first."
                )
        else:
            return ToolResult.fail(f"Unsupported path type: {rel_path}")


class MoveFileTool(Tool):
    @property
    def name(self) -> str:
        return "move_file"

    @property
    def is_mutating(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return "Move or rename a file or directory within the workspace."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source path relative to workspace"},
                "destination": {
                    "type": "string",
                    "description": "Destination path relative to workspace",
                },
            },
            "required": ["source", "destination"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 10

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.workspace is None:
            return ToolResult.fail("No workspace set")

        source = self._resolve_path(args["source"], ctx.workspace)
        destination = self._resolve_path(args["destination"], ctx.workspace)

        if not source.exists():
            return ToolResult.fail(f"Source not found: {args['source']}")

        if destination.exists():
            return ToolResult.fail(f"Destination already exists: {args['destination']}")

        rel_source = str(source.relative_to(ctx.workspace.resolve()))
        rel_dest = str(destination.relative_to(ctx.workspace.resolve()))

        # Record in changelog before moving
        if ctx.changelog is not None:
            ctx.changelog.record_rename(rel_source, rel_dest, subtask_id=ctx.subtask_id)

        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)

        return ToolResult.ok(
            f"Moved {rel_source} -> {rel_dest}",
            files_changed=[rel_source, rel_dest],
        )


class EditFileTool(Tool):
    """Edit a file by replacing strings, with fuzzy matching for local model tolerance.

    Supports single edits (old_str/new_str) and batched edits (edits array).
    When an exact match fails, attempts whitespace-normalized fuzzy matching
    so that local models with imprecise string reproduction still succeed.
    """

    # Minimum similarity ratio for fuzzy matching (0.0-1.0).
    FUZZY_THRESHOLD = 0.85

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def is_mutating(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Edit a file by replacing strings. Supports two modes:\n"
            "1. Single edit: provide old_str and new_str (old_str must be unique).\n"
            "2. Batch edit: provide an 'edits' array of {old_str, new_str} objects, "
            "applied sequentially.\n"
            "Whitespace-tolerant: minor whitespace differences are handled automatically."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace"},
                "old_str": {
                    "type": "string",
                    "description": "Exact string to find (must be unique). Use for single edits.",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement string. Use for single edits.",
                },
                "edits": {
                    "type": "array",
                    "description": (
                        "Array of edits to apply sequentially. "
                        "Each element: {old_str: string, new_str: string}."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_str": {"type": "string"},
                            "new_str": {"type": "string"},
                        },
                        "required": ["old_str", "new_str"],
                    },
                },
            },
            "required": ["path"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 10

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.workspace is None:
            return ToolResult.fail("No workspace set")

        path = self._resolve_path(args["path"], ctx.workspace)
        if not path.exists():
            return ToolResult.fail(f"File not found: {args['path']}")

        content = path.read_text(encoding="utf-8")
        rel_path = str(path.relative_to(ctx.workspace.resolve()))

        # Build edit list from either single or batch mode
        edits_raw = args.get("edits")
        if edits_raw:
            if not isinstance(edits_raw, list):
                return ToolResult.fail("'edits' must be an array of {old_str, new_str} objects.")
            edit_pairs = []
            for idx, e in enumerate(edits_raw):
                if not isinstance(e, dict):
                    return ToolResult.fail(
                        f"edits[{idx}] must be an object with old_str and new_str."
                    )
                edit_pairs.append((e.get("old_str", ""), e.get("new_str", "")))
        elif "old_str" in args:
            edit_pairs = [(args["old_str"], args.get("new_str", ""))]
        else:
            return ToolResult.fail("Provide either old_str/new_str or an edits array.")

        # Validate all edits before applying any
        for i, (old_str, _) in enumerate(edit_pairs):
            if not old_str:
                label = f"edit[{i}]" if len(edit_pairs) > 1 else "old_str"
                return ToolResult.fail(f"{label}: old_str cannot be empty.")

        original_content = content
        applied = []
        fuzzy_used = []

        language = detect_language(rel_path)

        for i, (old_str, new_str) in enumerate(edit_pairs):
            label = f"edit[{i}]" if len(edit_pairs) > 1 else ""
            result = self._apply_single_edit(content, old_str, new_str, rel_path, label, language)

            if isinstance(result, ToolResult):
                # Failed — don't write anything
                return result

            content, was_fuzzy = result
            applied.append((old_str, new_str))
            if was_fuzzy:
                fuzzy_used.append(i)

        # Record in changelog only after all edits validated, before writing
        if ctx.changelog is not None:
            ctx.changelog.record_before_write(rel_path, subtask_id=ctx.subtask_id)

        # Write the final result
        path.write_text(content, encoding="utf-8")

        # Build diff for the response
        diff_text = _generate_compact_diff(original_content, content, rel_path)

        # Build summary message
        n = len(applied)
        if n == 1:
            lines_old = applied[0][0].count("\n") + 1
            lines_new = applied[0][1].count("\n") + 1
            msg = f"Edited {rel_path}: replaced {lines_old} lines with {lines_new} lines"
        else:
            msg = f"Edited {rel_path}: applied {n} edits"

        if fuzzy_used:
            if n == 1:
                msg += " (fuzzy match used)"
            else:
                indices = ", ".join(str(i) for i in fuzzy_used)
                msg += f" (fuzzy match used for edit{'s' if len(fuzzy_used) > 1 else ''} {indices})"

        if diff_text:
            msg += f"\n\n{diff_text}"

        return ToolResult.ok(msg, files_changed=[rel_path])

    def _apply_single_edit(
        self,
        content: str,
        old_str: str,
        new_str: str,
        rel_path: str,
        label: str,
        language: str = "unknown",
    ) -> tuple[str, bool] | ToolResult:
        """Apply a single old_str->new_str edit, falling back to fuzzy match.

        Returns (new_content, was_fuzzy) on success, or ToolResult on failure.
        """
        prefix = f"{label}: " if label else ""

        # --- Exact match attempt ---
        count = content.count(old_str)
        if count == 1:
            return content.replace(old_str, new_str, 1), False
        if count > 1:
            return ToolResult.fail(
                f"{prefix}old_str appears {count} times in {rel_path}. "
                "Provide more surrounding context to make it unique."
            )

        # --- Fuzzy match attempt (structural + sliding window) ---
        match = self._fuzzy_find(content, old_str, language)
        if match is not None:
            matched_text, start, end = match
            return content[:start] + new_str + content[end:], True

        # --- Helpful error ---
        return ToolResult.fail(
            f"{prefix}old_str not found in {rel_path}. "
            f"Closest match:\n{self._closest_snippet(content, old_str)}"
        )

    def _fuzzy_find(
        self, content: str, old_str: str, language: str = "unknown"
    ) -> tuple[str, int, int] | None:
        """Find old_str in content using whitespace-normalized fuzzy matching.

        When tree-sitter is available and *language* is supported, structural
        candidates (function/class boundaries) are tried first to narrow the
        search space.  Falls back to a full sliding-window scan.

        Returns (matched_text, start_index, end_index) or None.
        Only matches if the best candidate is unambiguously better than
        any runner-up (or the only match above threshold).
        """
        # Try structural matching first when tree-sitter is available
        structural_result = self._structural_fuzzy_find(content, old_str, language)
        if structural_result is not None:
            return structural_result

        # Full sliding-window fallback
        return self._sliding_window_fuzzy_find(content, old_str)

    def _structural_fuzzy_find(
        self, content: str, old_str: str, language: str
    ) -> tuple[str, int, int] | None:
        """Try fuzzy matching anchored to structural (tree-sitter) nodes."""
        from loom.tools.treesitter import find_structural_candidates, is_available

        if not is_available() or language == "unknown":
            return None

        candidates = find_structural_candidates(content, language)
        if not candidates:
            return None

        old_lines = old_str.splitlines()
        if not old_lines:
            return None

        n = len(old_lines)
        best_ratio = 0.0
        second_ratio = 0.0
        best_line_range: tuple[int, int] | None = None

        content_lines = content.splitlines()
        line_starts = _line_start_offsets(content)

        for start_char, end_char in candidates:
            # Convert character offsets to line numbers
            start_line = _offset_to_line(line_starts, start_char)
            end_line = _offset_to_line(line_starts, end_char)
            # Ensure we have enough lines in this candidate region
            region_lines = content_lines[start_line:end_line + 1]
            if len(region_lines) < n:
                continue

            # Slide window within this structural region
            for offset in range(len(region_lines) - n + 1):
                candidate = region_lines[offset:offset + n]
                ratio = _line_similarity(old_lines, candidate)
                if ratio > best_ratio:
                    second_ratio = best_ratio
                    best_ratio = ratio
                    abs_start = start_line + offset
                    best_line_range = (abs_start, abs_start + n)
                elif ratio > second_ratio:
                    second_ratio = ratio

        if best_ratio < self.FUZZY_THRESHOLD or best_line_range is None:
            return None

        # Reject ambiguous matches
        if second_ratio >= self.FUZZY_THRESHOLD and (best_ratio - second_ratio) < 0.05:
            return None

        return self._line_range_to_match(
            content, content_lines, line_starts, best_line_range, old_str
        )

    def _sliding_window_fuzzy_find(
        self, content: str, old_str: str
    ) -> tuple[str, int, int] | None:
        """Original sliding-window fuzzy matching over the full file."""
        old_lines = old_str.splitlines()
        content_lines = content.splitlines()

        if not old_lines:
            return None

        n = len(old_lines)
        best_ratio = 0.0
        second_ratio = 0.0
        best_range: tuple[int, int] | None = None

        # Slide a window of size n over content lines
        for start in range(len(content_lines) - n + 1):
            candidate = content_lines[start:start + n]
            ratio = _line_similarity(old_lines, candidate)
            if ratio > best_ratio:
                second_ratio = best_ratio
                best_ratio = ratio
                best_range = (start, start + n)
            elif ratio > second_ratio:
                second_ratio = ratio

        if best_ratio < self.FUZZY_THRESHOLD or best_range is None:
            return None

        # Reject if runner-up is too close — match is ambiguous
        if second_ratio >= self.FUZZY_THRESHOLD and (best_ratio - second_ratio) < 0.05:
            return None

        line_starts = _line_start_offsets(content)
        return self._line_range_to_match(
            content, content_lines, line_starts, best_range, old_str
        )

    @staticmethod
    def _line_range_to_match(
        content: str,
        content_lines: list[str],
        line_starts: list[int],
        line_range: tuple[int, int],
        old_str: str,
    ) -> tuple[str, int, int]:
        """Convert a (start_line, end_line) range to (matched_text, char_start, char_end)."""
        char_start = line_starts[line_range[0]]

        end_line = line_range[1] - 1  # last matched line (inclusive)
        line_end_offset = line_starts[end_line] + len(content_lines[end_line])
        char_end = line_end_offset

        matched_text = content[char_start:char_end]

        # Handle trailing newline edge case: if the original old_str ended
        # with a newline, extend the match to include it
        if old_str.endswith("\n") and char_end < len(content) and content[char_end] == "\n":
            char_end += 1
            matched_text = content[char_start:char_end]

        return matched_text, char_start, char_end

    def _closest_snippet(self, content: str, old_str: str, context: int = 3) -> str:
        """Find the closest matching region and return it for error diagnostics."""
        old_lines = old_str.splitlines()
        content_lines = content.splitlines()

        if not old_lines or not content_lines:
            return "(empty)"

        # Find the best matching single line to anchor the snippet
        best_ratio = 0.0
        best_line = 0
        first_old = old_lines[0]
        for i, line in enumerate(content_lines):
            ratio = difflib.SequenceMatcher(None, first_old.strip(), line.strip()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_line = i

        start = max(0, best_line - context)
        end = min(len(content_lines), best_line + len(old_lines) + context)
        snippet = content_lines[start:end]
        numbered = [f"  {start + j + 1:4d} | {line}" for j, line in enumerate(snippet)]
        return "\n".join(numbered)


def _offset_to_line(line_starts: list[int], char_offset: int) -> int:
    """Return the 0-based line index containing *char_offset*."""
    lo, hi = 0, len(line_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if line_starts[mid] <= char_offset:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _line_start_offsets(content: str) -> list[int]:
    """Return the character offset where each line begins.

    Works correctly for \\n, \\r\\n, and \\r line endings.
    """
    offsets = [0]
    i = 0
    while i < len(content):
        ch = content[i]
        if ch == "\r":
            # \r\n or standalone \r
            if i + 1 < len(content) and content[i + 1] == "\n":
                i += 2
            else:
                i += 1
            offsets.append(i)
        elif ch == "\n":
            i += 1
            offsets.append(i)
        else:
            i += 1
    return offsets


def _line_similarity(a: list[str], b: list[str]) -> float:
    """Compare two line sequences with whitespace normalization.

    Returns a ratio between 0.0 (no match) and 1.0 (identical after normalization).
    """
    if len(a) != len(b):
        return 0.0

    total = 0.0
    for la, lb in zip(a, b):
        # Normalize: strip trailing whitespace, normalize internal whitespace runs
        na = " ".join(la.split())
        nb = " ".join(lb.split())
        total += difflib.SequenceMatcher(None, na, nb).ratio()

    return total / len(a)


def _generate_compact_diff(before: str, after: str, path: str, max_lines: int = 40) -> str:
    """Generate a compact unified diff, truncated if too long."""
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        before_lines, after_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    ))

    if not diff:
        return ""

    if len(diff) <= max_lines:
        return "\n".join(line.rstrip() for line in diff)

    truncated = diff[:max_lines]
    remaining = len(diff) - max_lines
    truncated.append(f"... ({remaining} more diff lines)")
    return "\n".join(line.rstrip() for line in truncated)
