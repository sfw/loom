"""Modal file viewer with pluggable per-extension renderers."""

from __future__ import annotations

import asyncio
import csv
import html as html_lib
import io
import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path

from rich.syntax import Syntax
from rich.table import Table
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Label, Markdown, Static

from loom.content_utils import extract_docx_text, extract_pptx_text, get_image_dimensions

FileRenderer = Callable[[Path, str | None], Widget]
_RENDERERS: dict[str, FileRenderer] = {}
_NAME_RENDERERS: dict[str, FileRenderer] = {}
MAX_PREVIEW_BYTES = 512 * 1024
MAX_RENDER_CHARS = 400_000
MAX_TABLE_ROWS = 200
MAX_TABLE_COLS = 24
MAX_CELL_CHARS = 120
MAX_PDF_PREVIEW_PAGES = 20
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"})
_PATH_ONLY_EXTS = frozenset({
    ".pdf",
    ".docx",
    ".pptx",
    *_IMAGE_EXTS,
})
_PREVIEW_LOAD_TIMEOUT_SECONDS = 8.0

_CODE_LEXERS = {
    ".txt": "text",
    ".log": "text",
    ".rst": "rst",
    ".mdx": "markdown",
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".css": "css",
    ".scss": "scss",
    ".less": "lesscss",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".swift": "swift",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".sql": "sql",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
}
_NAME_LEXERS = {
    "dockerfile": "docker",
    "makefile": "make",
}
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL,
)


def register_file_renderer(extensions: Iterable[str], renderer: FileRenderer) -> None:
    """Register a renderer for one or more file extensions."""
    for ext in extensions:
        normalized = ext.lower()
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        _RENDERERS[normalized] = renderer


def register_file_name_renderer(names: Iterable[str], renderer: FileRenderer) -> None:
    """Register a renderer for exact file names (e.g. Dockerfile)."""
    for name in names:
        _NAME_RENDERERS[name.lower()] = renderer


def resolve_file_renderer(path: Path) -> FileRenderer | None:
    """Resolve a renderer for a file path by suffix or file name."""
    suffix_renderer = _RENDERERS.get(path.suffix.lower())
    if suffix_renderer is not None:
        return suffix_renderer
    return _NAME_RENDERERS.get(path.name.lower())


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Truncate text by character count."""
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _truncate_cell(value: str) -> str:
    """Clamp long table cell values for readability."""
    if len(value) <= MAX_CELL_CHARS:
        return value
    return value[: MAX_CELL_CHARS - 3] + "..."


def _render_plain_text(content: str, lexer: str = "text", *, line_numbers: bool = True) -> Widget:
    """Render text with syntax highlighting."""
    return Static(
        Syntax(
            content,
            lexer,
            line_numbers=line_numbers,
            word_wrap=True,
        ),
    )


def _render_markdown(_path: Path, content: str | None) -> Widget:
    """Render Markdown content."""
    return Markdown(content or "")


def _render_code(path: Path, content: str | None) -> Widget:
    """Render code/text with language-aware syntax highlighting."""
    text = content or ""
    name = path.name.lower()
    lexer = _NAME_LEXERS.get(name) or _CODE_LEXERS.get(path.suffix.lower(), "text")
    return _render_plain_text(text, lexer)


def _render_json(path: Path, content: str | None) -> Widget:
    """Render JSON with pretty formatting and syntax highlighting."""
    text = content or ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return _render_code(path, text)
    pretty = json.dumps(parsed, indent=2, ensure_ascii=False, sort_keys=True)
    return _render_plain_text(pretty, "json")


def _render_delimited_table(path: Path, content: str | None) -> Widget:
    """Render CSV/TSV content into a readable table preview."""
    text = content or ""
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)

    rows: list[list[str]] = []
    truncated_rows = False
    for index, row in enumerate(reader):
        if index > MAX_TABLE_ROWS:
            truncated_rows = True
            break
        rows.append([cell.strip() for cell in row])

    if not rows:
        return Static("[dim]No rows found.[/dim]")

    max_cols_in_rows = max((len(row) for row in rows), default=0)
    shown_cols = min(max_cols_in_rows, MAX_TABLE_COLS)
    header = rows[0]
    body = rows[1:]

    table = Table(show_lines=False, expand=True, header_style="bold #7dcfff")
    for column_index in range(shown_cols):
        heading = (
            header[column_index]
            if column_index < len(header) and header[column_index]
            else f"col{column_index + 1}"
        )
        table.add_column(_truncate_cell(heading), overflow="fold")

    for row in body:
        values = [
            _truncate_cell(row[column_index] if column_index < len(row) else "")
            for column_index in range(shown_cols)
        ]
        table.add_row(*values)

    notes: list[str] = []
    if truncated_rows:
        notes.append(f"rows truncated to first {MAX_TABLE_ROWS:,}")
    if max_cols_in_rows > shown_cols:
        notes.append(f"columns truncated to first {shown_cols}")
    if notes:
        table.caption = "; ".join(notes)

    return Static(table)


def _strip_html_text(content: str) -> str:
    """Extract readable text from HTML."""
    text = _HTML_SCRIPT_STYLE_RE.sub(" ", content)
    text = _HTML_TAG_RE.sub(" ", text)
    text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _render_html(path: Path, content: str | None) -> Widget:
    """Render HTML as extracted readable text."""
    text = content or ""
    stripped = _strip_html_text(text)
    if not stripped:
        return _render_code(path, text)
    return _render_plain_text(stripped, "text", line_numbers=False)


def _render_pdf(path: Path, _content: str | None) -> Widget:
    """Render PDF extracted text preview (first pages)."""
    try:
        import pypdf
    except ImportError:
        return Static(
            "[bold #f7768e]PDF preview unavailable[/]\n"
            "Dependency missing in this environment. Run `uv sync` to install `pypdf`.",
        )

    try:
        reader = pypdf.PdfReader(path)
    except Exception as e:
        return Static(f"[bold #f7768e]Failed to read PDF:[/] {e}")

    total_pages = len(reader.pages)
    pages_to_read = min(total_pages, MAX_PDF_PREVIEW_PAGES)
    blocks: list[str] = []
    for index in range(pages_to_read):
        try:
            page_text = reader.pages[index].extract_text() or ""
        except Exception:
            page_text = ""
        if page_text.strip():
            blocks.append(f"--- Page {index + 1} ---\n{page_text.strip()}")

    if not blocks:
        output = f"[PDF: {path.name}, {total_pages} pages, no extractable text]"
    else:
        output = "\n\n".join(blocks)
        if total_pages > pages_to_read:
            output += (
                f"\n\n[Showing first {pages_to_read} of {total_pages} pages.]"
            )

    output, char_truncated = _truncate_text(output, MAX_RENDER_CHARS)
    if char_truncated:
        output += "\n\n[Preview text truncated.]"
    return _render_plain_text(output, "text", line_numbers=False)


def _render_docx(path: Path, _content: str | None) -> Widget:
    """Render extracted text from Word documents."""
    try:
        text = extract_docx_text(path)
    except Exception as e:
        return Static(f"[bold #f7768e]Failed to read Word document:[/] {e}")
    if not text.strip():
        text = f"[Word document: {path.name}, no extractable text]"
    text, truncated = _truncate_text(text, MAX_RENDER_CHARS)
    if truncated:
        text += "\n\n[Preview text truncated.]"
    return _render_plain_text(text, "text", line_numbers=False)


def _render_pptx(path: Path, _content: str | None) -> Widget:
    """Render extracted text from PowerPoint files."""
    try:
        text = extract_pptx_text(path)
    except Exception as e:
        return Static(f"[bold #f7768e]Failed to read PowerPoint file:[/] {e}")
    if not text.strip():
        text = f"[PowerPoint: {path.name}, no extractable text]"
    text, truncated = _truncate_text(text, MAX_RENDER_CHARS)
    if truncated:
        text += "\n\n[Preview text truncated.]"
    return _render_plain_text(text, "text", line_numbers=False)


def _render_image_metadata(path: Path, _content: str | None) -> Widget:
    """Render image metadata preview."""
    width, height = get_image_dimensions(path)
    size = path.stat().st_size
    text = (
        f"Image preview metadata\n"
        f"  Name: {path.name}\n"
        f"  Format: {path.suffix.lower() or '<none>'}\n"
        f"  Size: {size:,} bytes\n"
        f"  Dimensions: {width} x {height}"
    )
    return _render_plain_text(text, "text", line_numbers=False)


register_file_renderer((".md", ".markdown"), _render_markdown)
register_file_renderer((".json",), _render_json)
register_file_renderer((".csv", ".tsv"), _render_delimited_table)
register_file_renderer((".html", ".htm"), _render_html)
register_file_renderer((".diff", ".patch"), _render_code)
register_file_renderer((".docx",), _render_docx)
register_file_renderer((".pptx",), _render_pptx)
register_file_renderer((".pdf",), _render_pdf)
register_file_renderer(_IMAGE_EXTS, _render_image_metadata)
register_file_renderer(tuple(_CODE_LEXERS.keys()), _render_code)
register_file_name_renderer(tuple(_NAME_LEXERS.keys()), _render_code)


class FileViewerScreen(ModalScreen[None]):
    """Read-only file preview modal."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
    ]

    CSS = """
    FileViewerScreen {
        align: center middle;
    }
    #file-viewer-dialog {
        width: 90%;
        height: 85%;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }
    #file-viewer-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #file-viewer-body {
        height: 1fr;
    }
    #file-viewer-note {
        margin-bottom: 1;
        color: $text-muted;
    }
    #file-viewer-footer {
        margin-top: 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        path: Path,
        workspace: Path,
        *,
        defer_heavy_load: bool = False,
    ) -> None:
        super().__init__()
        self._path = path
        self._workspace = workspace
        self._viewer: Widget | None = None
        self._error: str | None = None
        self._truncated = False
        self._defer_heavy_load = bool(defer_heavy_load)
        self._loading = False
        self._load_request_id = 0
        if self._defer_heavy_load:
            self._loading = True
        else:
            self._load_preview()

    def _display_path(self) -> str:
        """Return a workspace-relative display path when possible."""
        try:
            return str(self._path.relative_to(self._workspace))
        except ValueError:
            return str(self._path)

    def _load_preview(self) -> None:
        """Resolve renderer and load file content into instance preview state."""
        viewer, error, truncated = self._load_preview_state()
        self._viewer = viewer
        self._error = error
        self._truncated = truncated

    def _load_preview_state(self) -> tuple[Widget | None, str | None, bool]:
        """Resolve renderer and load file content into a preview widget."""
        renderer = resolve_file_renderer(self._path)
        if renderer is None:
            suffix = self._path.suffix or "<none>"
            return (
                None,
                f"No viewer renderer registered for '{suffix}'. "
                "Supported now: markdown, code/text, JSON, CSV/TSV, HTML, "
                "diff/patch, docx, pptx, pdf, image metadata."
                ,
                False,
            )

        try:
            if self._path.suffix.lower() in _PATH_ONLY_EXTS:
                return renderer(self._path, None), None, False

            raw = self._path.read_bytes()
            truncated = False
            if len(raw) > MAX_PREVIEW_BYTES:
                raw = raw[:MAX_PREVIEW_BYTES]
                truncated = True
            content = raw.decode("utf-8", errors="replace")
            content, char_truncated = _truncate_text(content, MAX_RENDER_CHARS)
            truncated = truncated or char_truncated
            return renderer(self._path, content), None, truncated
        except Exception as e:
            return None, f"Failed to read file: {e}", False

    def _render_body(self) -> list[Widget]:
        """Build body widgets for current loading/error/view state."""
        if self._loading:
            return [Static("[dim]Loading preview...[/dim]")]
        if self._error:
            return [Static(f"[bold #f7768e]Unable to preview file[/]\n{self._error}")]
        widgets: list[Widget] = []
        if self._truncated:
            widgets.append(
                Static(
                    f"[dim]Preview truncated to first {MAX_PREVIEW_BYTES:,} bytes.[/dim]",
                    id="file-viewer-note",
                )
            )
        if self._viewer is not None:
            widgets.append(self._viewer)
        return widgets

    def _apply_body_state(self) -> None:
        """Refresh mounted body content after async load completes."""
        body = self.query_one("#file-viewer-body", VerticalScroll)
        for child in list(body.children):
            child.remove()
        for widget in self._render_body():
            body.mount(widget)

    async def on_mount(self) -> None:
        if not self._loading:
            return
        self._load_request_id += 1
        request_id = self._load_request_id

        async def _load_worker() -> None:
            try:
                viewer, error, truncated = await asyncio.wait_for(
                    asyncio.to_thread(self._load_preview_state),
                    timeout=_PREVIEW_LOAD_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                viewer, error, truncated = (
                    None,
                    (
                        "Preview timed out after "
                        f"{_PREVIEW_LOAD_TIMEOUT_SECONDS:.1f}s."
                    ),
                    False,
                )
            except Exception as e:
                viewer, error, truncated = None, f"Failed to read file: {e}", False
            finally:
                if request_id != self._load_request_id:
                    return
                self._viewer = viewer
                self._error = error
                self._truncated = truncated
                self._loading = False
                self._apply_body_state()

        self.run_worker(
            _load_worker(),
            group="file-viewer-preview",
            exclusive=True,
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="file-viewer-dialog"):
            yield Label(
                f"[bold #e0af68]File Viewer[/bold #e0af68]  [dim]{self._display_path()}[/dim]",
                id="file-viewer-title",
            )
            with VerticalScroll(id="file-viewer-body"):
                yield from self._render_body()
            yield Label("[dim]Esc or q: close[/dim]", id="file-viewer-footer")

    def action_close(self) -> None:
        self.dismiss(None)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        """Close when clicking outside the centered dialog."""
        dialog = self.query_one("#file-viewer-dialog")
        if dialog.region.contains(event.screen_x, event.screen_y):
            return
        self.dismiss(None)
        event.stop()
        event.prevent_default()
