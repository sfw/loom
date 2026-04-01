"""Web fetch tool for retrieving URL content.

Fetches a URL and returns text content. Includes safety controls:
timeout, max response size, and URL validation.
"""

from __future__ import annotations

import asyncio
import html as html_lib
import ipaddress
import logging
import os
import re
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx

from loom.config import Config
from loom.ingest.artifacts import (
    DEFAULT_RETENTION_MAX_AGE_DAYS,
    DEFAULT_RETENTION_MAX_BYTES_PER_SCOPE,
    DEFAULT_RETENTION_MAX_FILES_PER_SCOPE,
    persist_fetch_artifact,
)
from loom.ingest.handlers import summarize_artifact
from loom.ingest.router import (
    ContentKind,
    detect_content_kind,
    normalize_media_type,
)
from loom.models.base import ModelProvider
from loom.models.retry import ModelRetryPolicy, call_with_model_retry
from loom.models.router import ModelRouter, ResponseValidator
from loom.research.text import jaccard_similarity, token_overlap_ratio, tokenize
from loom.tools.registry import Tool, ToolContext, ToolResult
from loom.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)

# Safety: block private/internal networks
_BLOCKED_HOSTS = re.compile(
    r"^(localhost|127\.\d+\.\d+\.\d+|0\.0\.0\.0|10\.\d+\.\d+\.\d+|"
    r"172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+|\[::1\])",
    re.IGNORECASE,
)

MAX_RESPONSE_SIZE = 512 * 1024  # 512KB
MAX_DOWNLOAD_BYTES = MAX_RESPONSE_SIZE * 4  # 2MB bounded download
MAX_HTML_SOURCE_DOWNLOAD_BYTES = MAX_RESPONSE_SIZE  # 512KB raw HTML source cap
MAX_BINARY_SUMMARY_CHARS = 3_600
FETCH_TIMEOUT = 30.0
MAX_FETCH_ATTEMPTS = 3
FETCH_RETRY_BASE_DELAY = 0.4
RETRYABLE_HTTP_STATUS = frozenset({403, 408, 425, 429, 500, 502, 503, 504})
DEFAULT_WEB_USER_AGENT = "Loom/1.0 (+https://github.com/sfw/loom)"
DIRECT_TEXT_TOKEN_LIMIT = 5_000
SUMMARY_TEXT_TOKEN_LIMIT = 20_000
TEXT_QUERY_CHUNK_TOKENS = 500
MAX_SUMMARY_HEADINGS = 8
MAX_SUMMARY_BULLETS = 8
MAX_QUERY_SNIPPETS = 4
MAX_SNIPPET_CHARS = 900
MAX_HEAD_TAIL_CHARS = 3_600
_BOILERPLATE_HINT_RE = re.compile(
    r"(breadcrumb|comment|cookie|disclosure|footer|gdpr|menu|modal|nav|pager|popup|"
    r"promo|related|share|sidebar|social|subscribe|toolbar)",
    re.IGNORECASE,
)
_MAIN_CONTENT_HINT_RE = re.compile(
    r"(article|content|doc|main|page|post|story)",
    re.IGNORECASE,
)
_HEADING_LEVELS = {
    "h1": 1,
    "h2": 2,
    "h3": 3,
    "h4": 4,
    "h5": 5,
    "h6": 6,
}
_BLOCKISH_TAGS = frozenset({
    "article",
    "blockquote",
    "body",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "li",
    "main",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "ul",
})
_CONTAINER_TAGS = frozenset({"article", "body", "div", "header", "main", "section"})
_SKIP_RENDER_TAGS = frozenset({
    "button",
    "canvas",
    "dialog",
    "form",
    "iframe",
    "input",
    "noscript",
    "script",
    "select",
    "style",
    "svg",
    "template",
    "textarea",
})


@dataclass(frozen=True)
class PreparedWebContent:
    text: str
    format: str
    processor: str
    title: str = ""
    headings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RenderedWebContent:
    output: str
    strategy: str
    estimated_tokens: int
    output_format: str


class _WebFetchExtractor:
    def __init__(self, config: Config | None = None):
        self._config = config
        self._router: ModelRouter | None = None
        self._validator = ResponseValidator()
        if config is not None and config.models:
            try:
                self._router = ModelRouter.from_config(config)
            except Exception:
                logger.debug("Failed to initialize model router for web_fetch", exc_info=True)

    async def summarize_medium(
        self,
        prepared: PreparedWebContent,
        *,
        url: str,
        query: str,
    ) -> str | None:
        model = self._select_model()
        if model is None:
            return None

        prompt = _build_extractor_prompt(prepared, url=url, query=query)
        policy = (
            ModelRetryPolicy.from_execution_config(self._config.execution)
            if self._config is not None
            else ModelRetryPolicy()
        )

        async def _invoke():
            return await model.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=min(1200, model.configured_max_tokens or 1200),
            )

        try:
            response = await call_with_model_retry(_invoke, policy=policy)
        except Exception:
            logger.debug("web_fetch extractor model call failed", exc_info=True)
            return None

        validation = self._validator.validate_json_response(
            response,
            expected_keys=["summary", "key_sections", "key_points"],
        )
        if not validation.valid:
            logger.debug("web_fetch extractor response invalid: %s", validation.error)
            return None

        payload = validation.parsed
        if not isinstance(payload, dict):
            return None
        return _format_extractor_payload(prepared, payload)

    def _select_model(self) -> ModelProvider | None:
        if self._router is None:
            return None
        candidates = [
            (1, "extractor"),
            (2, "extractor"),
            (1, "verifier"),
            (2, "verifier"),
            (1, "executor"),
        ]
        for tier, role in candidates:
            try:
                return self._router.select(tier=tier, role=role)
            except Exception:
                continue
        return None


def _is_private_ip(ip_str: str) -> bool:
    """Check if an IP address belongs to a private/reserved range."""
    try:
        addr = ipaddress.ip_address(ip_str)
        return (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_reserved
            or addr.is_unspecified
        )
    except ValueError:
        return False


def is_safe_url(url: str) -> tuple[bool, str]:
    """Check if a URL is safe to fetch."""
    if not url.startswith(("http://", "https://")):
        return False, "Only http:// and https:// URLs are allowed"

    # Extract host
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
    except Exception:
        return False, "Invalid URL format"

    if _BLOCKED_HOSTS.match(host):
        return False, f"Blocked host: {host} (private/internal network)"

    # Resolve hostname and check resolved IPs against private ranges
    try:
        infos = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _family, _type, _proto, _canonname, sockaddr in infos:
            ip_str = sockaddr[0]
            if _is_private_ip(ip_str):
                return False, f"Blocked host: {host} resolves to private address {ip_str}"
    except socket.gaierror:
        # DNS resolution failed — allow the request to proceed and fail naturally
        pass

    return True, ""


def _build_request_headers() -> dict[str, str]:
    """Build default request headers for web requests.

    `LOOM_WEB_USER_AGENT` can override the default user-agent.
    """
    user_agent = os.environ.get("LOOM_WEB_USER_AGENT", "").strip()
    if not user_agent:
        user_agent = DEFAULT_WEB_USER_AGENT
    return {
        "User-Agent": user_agent,
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "text/plain;q=0.8,*/*;q=0.7"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }


def _should_retry_status(status_code: int) -> bool:
    """Return True if an HTTP status is likely transient."""
    return status_code in RETRYABLE_HTTP_STATUS


def _hidden_int_arg(
    args: dict,
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = args.get(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = int(default)
    return max(minimum, min(maximum, value))


async def _get_with_retries(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    stream: bool = False,
) -> httpx.Response:
    """GET URL with bounded retries for transient failures."""
    for attempt in range(MAX_FETCH_ATTEMPTS):
        try:
            request = client.build_request("GET", url, headers=headers)
            response = await client.send(request, stream=stream)
            if (
                _should_retry_status(response.status_code)
                and attempt < MAX_FETCH_ATTEMPTS - 1
            ):
                await response.aclose()
                await asyncio.sleep(FETCH_RETRY_BASE_DELAY * (2 ** attempt))
                continue
            return response
        except (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
        ):
            if attempt >= MAX_FETCH_ATTEMPTS - 1:
                raise
            await asyncio.sleep(FETCH_RETRY_BASE_DELAY * (2 ** attempt))

    raise RuntimeError(f"Fetch failed after {MAX_FETCH_ATTEMPTS} attempts: {url}")


async def _read_response_limited(
    response: httpx.Response,
    max_bytes: int,
) -> tuple[bytes, bool]:
    """Read response body up to max_bytes and return (data, truncated)."""
    data = bytearray()
    truncated = False
    async for chunk in response.aiter_bytes():
        if not chunk:
            continue
        remaining = max_bytes - len(data)
        if remaining <= 0:
            truncated = True
            break
        if len(chunk) > remaining:
            data.extend(chunk[:remaining])
            truncated = True
            break
        data.extend(chunk)
    return bytes(data), truncated


def _decode_response_bytes(response: httpx.Response, content: bytes) -> str:
    """Decode response bytes using response charset with safe fallback."""
    encoding = response.encoding or "utf-8"
    try:
        return content.decode(encoding, errors="replace")
    except LookupError:
        return content.decode("utf-8", errors="replace")


def _looks_like_html(content: str, content_type: str) -> bool:
    """Best-effort HTML detection for mislabeled responses."""
    ctype = (content_type or "").lower()
    if "html" in ctype:
        return True

    sample = (content or "")[:4096].lower()
    if not sample:
        return False

    if "<!doctype html" in sample:
        return True
    if re.search(r"<html(?:\s|>)", sample):
        return True
    if re.search(r"<(head|body|title|meta|script|style|main|article|nav|footer)(\s|>)", sample):
        return True

    # Fallback heuristic: many HTML-like tags near start of payload.
    tag_like = re.findall(r"</?[a-z][a-z0-9:-]*(?:\s[^<>]*)?>", sample)
    return len(tag_like) >= 8


async def _execute_web_fetch(
    url: str,
    *,
    extract_text: bool,
    max_download_bytes: int,
    query: str = "",
    summarizer: _WebFetchExtractor | None = None,
    enable_filetype_ingest_router: bool = True,
    artifact_retention_max_age_days: int = DEFAULT_RETENTION_MAX_AGE_DAYS,
    artifact_retention_max_files_per_scope: int = DEFAULT_RETENTION_MAX_FILES_PER_SCOPE,
    artifact_retention_max_bytes_per_scope: int = DEFAULT_RETENTION_MAX_BYTES_PER_SCOPE,
    ctx: ToolContext | None = None,
) -> ToolResult:
    if not url:
        return ToolResult.fail("No URL provided")

    safe, reason = is_safe_url(url)
    if not safe:
        return ToolResult.fail(reason)

    try:
        headers = _build_request_headers()
        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(FETCH_TIMEOUT),
            headers=headers,
        ) as client:
            response = await _get_with_retries(client, url, stream=True)

            # Follow redirects manually to validate each target against SSRF
            redirect_count = 0
            while response.is_redirect and redirect_count < 5:
                redirect_count += 1
                location = response.headers.get("location", "")
                if not location:
                    break
                location = urljoin(str(response.url), location)
                redir_safe, redir_reason = is_safe_url(location)
                if not redir_safe:
                    await response.aclose()
                    return ToolResult.fail(f"Redirect blocked: {redir_reason}")
                await response.aclose()
                response = await _get_with_retries(
                    client, location, stream=True,
                )

            if response.is_redirect:
                await response.aclose()
                return ToolResult.fail("Too many redirects (max 5)")

            response.raise_for_status()

            # Bounded streaming read to avoid huge response bodies.
            # Keep only the first max_download_bytes bytes.
            content_length = response.headers.get("content-length")
            content_type = response.headers.get("content-type", "")
            declared_size = None
            if content_length:
                try:
                    declared_size = int(content_length)
                except ValueError:
                    declared_size = None
            content_bytes, stream_truncated = await _read_response_limited(
                response, max_download_bytes,
            )
            resolved_url = str(response.url)
            status_code = response.status_code
            await response.aclose()
            media_type = normalize_media_type(content_type)
            truncation_notes: list[str] = []
            if stream_truncated or (
                declared_size is not None and declared_size > max_download_bytes
            ):
                truncation_notes.append(
                    f"download truncated to first {max_download_bytes} bytes"
                )

            if not enable_filetype_ingest_router:
                content = _decode_response_bytes(response, content_bytes)
                if extract_text and _looks_like_html(content, media_type):
                    content = _strip_html(content)
                if truncation_notes:
                    content += "\n\n... (" + "; ".join(truncation_notes) + ")"
                return ToolResult.ok(
                    content,
                    data={
                        "url": resolved_url,
                        "status_code": status_code,
                        "content_type": media_type or content_type,
                        "content_kind": ContentKind.TEXT,
                        "size_bytes": len(content),
                        "declared_size_bytes": declared_size,
                        "truncated": bool(truncation_notes),
                        "extract_text": extract_text,
                    },
                )

            content_kind = detect_content_kind(
                content_type=media_type,
                content_bytes=content_bytes,
                url=resolved_url,
            )

            if content_kind in {ContentKind.TEXT, ContentKind.HTML}:
                content = _decode_response_bytes(response, content_bytes)
                prepared: PreparedWebContent | None = None
                rendered: RenderedWebContent | None = None

                if extract_text:
                    prepared = _prepare_textual_web_content(
                        content,
                        content_kind=content_kind,
                        media_type=media_type,
                        url=resolved_url,
                    )
                    rendered = await _render_textual_web_content(
                        prepared,
                        url=resolved_url,
                        query=query,
                        summarizer=summarizer,
                    )
                    content = rendered.output

                artifact_record = None
                artifact_media_type = ""
                if (
                    extract_text
                    and prepared is not None
                    and rendered is not None
                    and rendered.strategy != "full_content"
                    and ctx is not None
                ):
                    artifact_media_type = (
                        "text/markdown" if prepared.format == "markdown" else "text/plain"
                    )
                    artifact_record = persist_fetch_artifact(
                        content_bytes=prepared.text.encode("utf-8"),
                        source_url=resolved_url,
                        media_type=artifact_media_type,
                        content_kind=ContentKind.TEXT,
                        workspace=ctx.workspace,
                        scratch_dir=ctx.scratch_dir,
                        subtask_id=ctx.subtask_id,
                        retention_max_age_days=artifact_retention_max_age_days,
                        retention_max_files_per_scope=artifact_retention_max_files_per_scope,
                        retention_max_bytes_per_scope=artifact_retention_max_bytes_per_scope,
                    )
                    content += (
                        "\n\n"
                        f"[Full cleaned text persisted as artifact_ref "
                        f"{artifact_record.artifact_ref}; use read_artifact for follow-up.]"
                    )

                if truncation_notes:
                    content += "\n\n[" + "; ".join(truncation_notes) + "]"

                data: dict[str, object] = {
                    "url": resolved_url,
                    "status_code": status_code,
                    "content_type": media_type or content_type,
                    "content_kind": content_kind,
                    "size_bytes": len(content),
                    "declared_size_bytes": declared_size,
                    "truncated": bool(truncation_notes),
                    "extract_text": extract_text,
                    "download_bytes": len(content_bytes),
                }
                if prepared is not None and rendered is not None:
                    data.update({
                        "source_format": prepared.format,
                        "source_processor": prepared.processor,
                        "render_strategy": rendered.strategy,
                        "estimated_tokens": rendered.estimated_tokens,
                        "output_format": rendered.output_format,
                        "cleaned_chars": len(prepared.text),
                        "output_chars": len(content),
                        "source_title": prepared.title,
                        "heading_count": len(prepared.headings),
                    })
                    if prepared.headings:
                        data["headings"] = list(prepared.headings[:MAX_SUMMARY_HEADINGS])
                    if query.strip():
                        data["query"] = query.strip()
                if artifact_record is not None:
                    data.update({
                        "artifact_ref": artifact_record.artifact_ref,
                        "artifact_path": str(artifact_record.path),
                        "artifact_content_kind": ContentKind.TEXT,
                        "artifact_content_type": artifact_media_type,
                        "artifact_text_format": prepared.format if prepared is not None else "",
                    })
                    if artifact_record.workspace_relpath:
                        data["artifact_workspace_relpath"] = artifact_record.workspace_relpath
                    cleanup_stats = dict(getattr(artifact_record, "cleanup_stats", {}) or {})
                    if cleanup_stats:
                        data["artifact_retention"] = {
                            "scopes_scanned": int(cleanup_stats.get("scopes_scanned", 0)),
                            "files_deleted": int(cleanup_stats.get("files_deleted", 0)),
                            "bytes_deleted": int(cleanup_stats.get("bytes_deleted", 0)),
                        }
                return ToolResult.ok(content, data=data)

            # Binary/document path: persist bytes and return artifact-backed summary.
            workspace = None
            scratch_dir = None
            subtask_id = ""
            if ctx is not None:
                workspace = ctx.workspace
                scratch_dir = ctx.scratch_dir
                subtask_id = ctx.subtask_id
            record = persist_fetch_artifact(
                content_bytes=content_bytes,
                source_url=resolved_url,
                media_type=media_type or content_type,
                content_kind=content_kind,
                workspace=workspace,
                scratch_dir=scratch_dir,
                subtask_id=subtask_id,
                retention_max_age_days=artifact_retention_max_age_days,
                retention_max_files_per_scope=artifact_retention_max_files_per_scope,
                retention_max_bytes_per_scope=artifact_retention_max_bytes_per_scope,
            )
            summary = summarize_artifact(
                path=record.path,
                content_kind=content_kind,
                media_type=media_type or content_type,
                max_chars=MAX_BINARY_SUMMARY_CHARS,
            )
            output = summary.summary_text
            if truncation_notes:
                output += "\n\n... (" + "; ".join(truncation_notes) + ")"

            data: dict = {
                "url": resolved_url,
                "status_code": status_code,
                "content_type": media_type or content_type,
                "content_kind": content_kind,
                "size_bytes": record.size_bytes,
                "declared_size_bytes": declared_size,
                "truncated": bool(truncation_notes),
                "extract_text": extract_text,
                "artifact_ref": record.artifact_ref,
                "artifact_path": str(record.path),
                "handler": summary.handler,
                "extracted_chars": len(summary.extracted_text),
                "extraction_truncated": bool(summary.extraction_truncated),
            }
            if record.workspace_relpath:
                data["artifact_workspace_relpath"] = record.workspace_relpath
            cleanup_stats = dict(getattr(record, "cleanup_stats", {}) or {})
            if cleanup_stats:
                data["artifact_retention"] = {
                    "scopes_scanned": int(cleanup_stats.get("scopes_scanned", 0)),
                    "files_deleted": int(cleanup_stats.get("files_deleted", 0)),
                    "bytes_deleted": int(cleanup_stats.get("bytes_deleted", 0)),
                }
            if summary.metadata:
                data["handler_metadata"] = summary.metadata
            return ToolResult.ok(output, data=data)
    except httpx.HTTPStatusError as e:
        target = str(e.request.url) if e.request else url
        return ToolResult.fail(f"HTTP {e.response.status_code}: {target}")
    except httpx.TimeoutException:
        return ToolResult.fail(f"Timeout fetching: {url}")
    except httpx.ConnectError:
        return ToolResult.fail(f"Connection failed: {url}")
    except httpx.RemoteProtocolError:
        return ToolResult.fail(f"Protocol error fetching: {url}")
    except Exception as e:
        return ToolResult.fail(f"Fetch error: {e}")


class WebFetchTool(Tool):
    def __init__(
        self,
        config: Config | None = None,
        *,
        summarizer: _WebFetchExtractor | None = None,
    ):
        self._summarizer = summarizer if summarizer is not None else _WebFetchExtractor(config)

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch content from a URL and return cleaned text or markdown. "
            "HTML pages are converted into a content-focused markdown-like view. "
            "Large pages are summarized automatically, and optional query can "
            "focus very large pages on relevant snippets. "
            "Use for reading documentation, API specs, etc. "
            "Blocked: private/internal networks."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch (http or https)",
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Optional focus query. Helpful when the page is very large "
                        "and only relevant snippets should be returned."
                    ),
                },
            },
            "required": ["url"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 45

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        url = args.get("url", "")
        query = str(args.get("query", "") or "").strip()
        enable_router = bool(args.get("_enable_filetype_ingest_router", True))
        retention_max_age_days = _hidden_int_arg(
            args,
            "_artifact_retention_max_age_days",
            DEFAULT_RETENTION_MAX_AGE_DAYS,
            minimum=0,
            maximum=3650,
        )
        retention_max_files_per_scope = _hidden_int_arg(
            args,
            "_artifact_retention_max_files_per_scope",
            DEFAULT_RETENTION_MAX_FILES_PER_SCOPE,
            minimum=1,
            maximum=200_000,
        )
        retention_max_bytes_per_scope = _hidden_int_arg(
            args,
            "_artifact_retention_max_bytes_per_scope",
            DEFAULT_RETENTION_MAX_BYTES_PER_SCOPE,
            minimum=1024,
            maximum=20_000_000_000,
        )
        return await _execute_web_fetch(
            url,
            extract_text=True,
            max_download_bytes=MAX_DOWNLOAD_BYTES,
            query=query,
            summarizer=self._summarizer,
            enable_filetype_ingest_router=enable_router,
            artifact_retention_max_age_days=retention_max_age_days,
            artifact_retention_max_files_per_scope=retention_max_files_per_scope,
            artifact_retention_max_bytes_per_scope=retention_max_bytes_per_scope,
            ctx=ctx,
        )


class WebFetchHtmlTool(Tool):
    @property
    def name(self) -> str:
        return "web_fetch_html"

    @property
    def description(self) -> str:
        return (
            "Fetch raw HTML source from a URL (no tag stripping). "
            "Use when source markup is required, such as web design/debug tasks. "
            "Blocked: private/internal networks."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch (http or https)",
                },
            },
            "required": ["url"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 45

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        url = args.get("url", "")
        enable_router = bool(args.get("_enable_filetype_ingest_router", True))
        retention_max_age_days = _hidden_int_arg(
            args,
            "_artifact_retention_max_age_days",
            DEFAULT_RETENTION_MAX_AGE_DAYS,
            minimum=0,
            maximum=3650,
        )
        retention_max_files_per_scope = _hidden_int_arg(
            args,
            "_artifact_retention_max_files_per_scope",
            DEFAULT_RETENTION_MAX_FILES_PER_SCOPE,
            minimum=1,
            maximum=200_000,
        )
        retention_max_bytes_per_scope = _hidden_int_arg(
            args,
            "_artifact_retention_max_bytes_per_scope",
            DEFAULT_RETENTION_MAX_BYTES_PER_SCOPE,
            minimum=1024,
            maximum=20_000_000_000,
        )
        return await _execute_web_fetch(
            url,
            extract_text=False,
            max_download_bytes=MAX_HTML_SOURCE_DOWNLOAD_BYTES,
            enable_filetype_ingest_router=enable_router,
            artifact_retention_max_age_days=retention_max_age_days,
            artifact_retention_max_files_per_scope=retention_max_files_per_scope,
            artifact_retention_max_bytes_per_scope=retention_max_bytes_per_scope,
            ctx=ctx,
        )


def _strip_html(html_text: str) -> str:
    """Strip HTML markup and return compact plain text."""
    text = html_text or ""
    # Remove blocks that do not contribute readable content.
    text = re.sub(
        r"<!--.*?-->",
        " ",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"<(script|style|noscript|svg|canvas|template|iframe)[^>]*>.*?</\1>",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Treat block-level tags as line boundaries before stripping all tags.
    text = re.sub(
        r"</?(p|div|article|section|main|aside|header|footer|nav|li|ul|ol|h[1-6]|br|tr|td|th|table|pre|blockquote)[^>]*>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<[^>]+>", " ", text)

    # Decode entities and normalize whitespace.
    text = html_lib.unescape(text)
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def _prepare_textual_web_content(
    content: str,
    *,
    content_kind: str,
    media_type: str,
    url: str,
) -> PreparedWebContent:
    if content_kind == ContentKind.HTML or _looks_like_html(content, media_type):
        return _html_to_semantic_markdown(content, url=url)

    normalized = str(content or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return PreparedWebContent(
        text=normalized,
        format="text",
        processor="plain_text",
    )


def _html_to_semantic_markdown(html_text: str, *, url: str) -> PreparedWebContent:
    title = _extract_html_title(html_text)
    try:
        from lxml import html as lxml_html
    except Exception:
        stripped = _strip_html(html_text)
        return PreparedWebContent(
            text=stripped,
            format="text",
            processor="html_strip_fallback",
            title=title,
        )

    try:
        root = lxml_html.fromstring(html_text)
    except Exception:
        stripped = _strip_html(html_text)
        return PreparedWebContent(
            text=stripped,
            format="text",
            processor="html_parse_fallback",
            title=title,
        )

    if url:
        try:
            root.make_links_absolute(url, resolve_base_href=True)
        except Exception:
            pass

    for comment in root.xpath("//comment()"):
        parent = comment.getparent()
        if parent is not None:
            parent.remove(comment)

    for node in root.xpath(
        "//script|//style|//noscript|//svg|//canvas|//template|//iframe|"
        "//form|//button|//input|//select|//textarea|//dialog|//nav|//footer|//aside"
    ):
        parent = node.getparent()
        if parent is not None:
            parent.remove(node)

    for node in root.xpath(
        "//*[@hidden or @aria-hidden='true' or "
        "contains(translate(@style, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), "
        "'display:none') or "
        "contains(translate(@style, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), "
        "'visibility:hidden')]"
    ):
        parent = node.getparent()
        if parent is not None:
            parent.remove(node)

    for node in root.xpath("//*[@id or @class]"):
        hint = " ".join(filter(None, [node.get("id", ""), node.get("class", "")]))
        if _BOILERPLATE_HINT_RE.search(hint):
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)

    content_root = _select_primary_content_node(root)
    headings = _extract_headings(content_root)
    if not title:
        title = headings[0] if headings else ""
    blocks = _render_markdown_blocks(content_root, base_url=url)
    markdown = _collapse_rendered_blocks(blocks)
    stripped = _strip_html(html_text)
    if not markdown or len(markdown) < max(80, len(stripped) // 5):
        return PreparedWebContent(
            text=stripped,
            format="text",
            processor="html_strip_fallback",
            title=title,
            headings=headings,
        )
    return PreparedWebContent(
        text=markdown,
        format="markdown",
        processor="semantic_markdown",
        title=title,
        headings=headings,
    )


def _extract_html_title(html_text: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html_text or "", re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\s+", " ", html_lib.unescape(match.group(1))).strip()


def _select_primary_content_node(root):
    body_nodes = root.xpath("//body")
    body = body_nodes[0] if body_nodes else root

    preferred = body.xpath(".//main | .//article | .//*[@role='main']")
    best_preferred = _best_scored_content_node(preferred)
    if best_preferred is not None:
        return best_preferred

    candidates = body.xpath(".//*[self::article or self::section or self::main or self::div]")
    best_candidate = _best_scored_content_node(candidates)
    return best_candidate or body


def _best_scored_content_node(nodes):
    best_node = None
    best_score = float("-inf")
    for node in nodes:
        text = _node_text(node)
        if len(text) < 180:
            continue
        hint = " ".join(filter(None, [node.get("id", ""), node.get("class", "")]))
        tag = _local_tag(node)
        p_count = len(node.xpath(".//p"))
        heading_count = len(node.xpath(".//h1 | .//h2 | .//h3"))
        link_text_len = sum(len(_node_text(link)) for link in node.xpath(".//a"))
        score = (
            len(text)
            + (p_count * 140)
            + (heading_count * 220)
            - int(link_text_len * 0.35)
        )
        if tag in {"article", "main"}:
            score += 320
        if _MAIN_CONTENT_HINT_RE.search(hint):
            score += 180
        if _BOILERPLATE_HINT_RE.search(hint):
            score -= 600
        if score > best_score:
            best_score = score
            best_node = node
    return best_node


def _extract_headings(root) -> tuple[str, ...]:
    headings: list[str] = []
    for node in root.xpath(".//h1 | .//h2 | .//h3"):
        text = _node_text(node)
        if not text or text in headings:
            continue
        headings.append(text)
        if len(headings) >= 12:
            break
    return tuple(headings)


def _local_tag(node) -> str:
    tag = getattr(node, "tag", "")
    if not isinstance(tag, str):
        return ""
    if "}" in tag:
        tag = tag.rsplit("}", 1)[-1]
    return tag.lower()


def _node_text(node) -> str:
    raw = " ".join(str(piece or "") for piece in node.itertext())
    return re.sub(r"\s+", " ", raw).strip()


def _render_markdown_blocks(node, *, base_url: str, indent: int = 0) -> list[str]:
    tag = _local_tag(node)
    if not tag or tag in _SKIP_RENDER_TAGS:
        return []
    if tag in _HEADING_LEVELS:
        text = _render_inline_markdown(node, base_url=base_url).strip()
        return [f"{'#' * _HEADING_LEVELS[tag]} {text}"] if text else []
    if tag == "p":
        text = _render_inline_markdown(node, base_url=base_url).strip()
        return [text] if text else []
    if tag == "pre":
        text = node.text_content().strip("\n")
        if not text:
            return []
        return [f"```text\n{text}\n```"]
    if tag == "blockquote":
        quoted_blocks: list[str] = []
        for child in node:
            quoted_blocks.extend(
                _render_markdown_blocks(child, base_url=base_url, indent=indent),
            )
        if not quoted_blocks:
            inline = _render_inline_markdown(node, base_url=base_url).strip()
            if inline:
                quoted_blocks.append(inline)
        rendered: list[str] = []
        for block in quoted_blocks:
            if not block.strip():
                continue
            rendered.append(
                "\n".join(
                    "> " + line if line.strip() else ">"
                    for line in block.splitlines()
                ),
            )
        return rendered
    if tag in {"ul", "ol"}:
        items: list[str] = []
        for idx, child in enumerate(node, 1):
            if _local_tag(child) != "li":
                continue
            rendered = _render_list_item(
                child,
                base_url=base_url,
                ordered=(tag == "ol"),
                index=idx,
                indent=indent,
            )
            if rendered:
                items.append(rendered)
        return ["\n".join(items)] if items else []
    if tag == "table":
        table = _render_markdown_table(node, base_url=base_url)
        return [table] if table else []
    if tag in _CONTAINER_TAGS:
        blocks: list[str] = []
        has_block_child = any(_local_tag(child) in _BLOCKISH_TAGS for child in node)
        if not has_block_child:
            inline = _render_inline_markdown(node, base_url=base_url).strip()
            return [inline] if inline else []
        for child in node:
            blocks.extend(_render_markdown_blocks(child, base_url=base_url, indent=indent))
        return blocks
    if tag == "li":
        item = _render_list_item(
            node,
            base_url=base_url,
            ordered=False,
            index=1,
            indent=indent,
        )
        return [item] if item else []
    inline = _render_inline_markdown(node, base_url=base_url).strip()
    return [inline] if inline else []


def _render_list_item(
    node,
    *,
    base_url: str,
    ordered: bool,
    index: int,
    indent: int,
) -> str:
    prefix = f"{index}. " if ordered else "- "
    inline_parts: list[str] = []
    nested_blocks: list[str] = []

    leading = re.sub(r"\s+", " ", str(node.text or "")).strip()
    if leading:
        inline_parts.append(leading)

    for child in node:
        child_tag = _local_tag(child)
        if child_tag in {"ul", "ol"}:
            nested_blocks.extend(
                _render_markdown_blocks(child, base_url=base_url, indent=indent + 2),
            )
        elif child_tag in {"p", "pre", "blockquote", "table"}:
            nested_blocks.extend(
                _render_markdown_blocks(child, base_url=base_url, indent=indent + 2),
            )
        else:
            fragment = _render_inline_markdown(child, base_url=base_url).strip()
            if fragment:
                inline_parts.append(fragment)
        tail = re.sub(r"\s+", " ", str(child.tail or "")).strip()
        if tail:
            inline_parts.append(tail)

    lines: list[str] = []
    inline_text = re.sub(r"\s+", " ", " ".join(inline_parts)).strip()
    if inline_text:
        lines.append((" " * indent) + prefix + inline_text)
    for block in nested_blocks:
        for line in block.splitlines():
            if line.strip():
                lines.append((" " * (indent + 2)) + line)
    return "\n".join(lines).strip()


def _render_inline_markdown(node, *, base_url: str) -> str:
    parts: list[str] = []

    def _append(fragment: str) -> None:
        text = str(fragment or "")
        if not text:
            return
        if parts and not parts[-1].endswith((" ", "\n", "(", "[", "{")):
            if not text.startswith(("\n", ")", "]", "}", ".", ",", ":", ";", "?", "!")):
                parts.append(" ")
        parts.append(text)

    leading = re.sub(r"\s+", " ", str(node.text or "")).strip()
    if leading:
        _append(leading)

    for child in node:
        tag = _local_tag(child)
        if not tag or tag in _SKIP_RENDER_TAGS:
            pass
        elif tag in _HEADING_LEVELS or tag in {"p", "pre", "blockquote", "li", "table"}:
            text = _node_text(child)
            if text:
                _append(text)
        elif tag == "a":
            label = _render_inline_markdown(child, base_url=base_url).strip() or _node_text(child)
            href = _resolve_href(child.get("href", ""), base_url=base_url)
            if href and label:
                _append(f"[{label}]({href})")
            elif label:
                _append(label)
            elif href:
                _append(href)
        elif tag in {"strong", "b"}:
            text = _render_inline_markdown(child, base_url=base_url).strip() or _node_text(child)
            if text:
                _append(f"**{text}**")
        elif tag in {"em", "i"}:
            text = _render_inline_markdown(child, base_url=base_url).strip() or _node_text(child)
            if text:
                _append(f"*{text}*")
        elif tag in {"code", "kbd", "samp"}:
            text = _node_text(child)
            if text:
                _append(f"`{text}`")
        elif tag == "br":
            if parts and not parts[-1].endswith("\n"):
                parts.append("\n")
        else:
            text = _render_inline_markdown(child, base_url=base_url).strip() or _node_text(child)
            if text:
                _append(text)
        tail = re.sub(r"\s+", " ", str(child.tail or "")).strip()
        if tail:
            _append(tail)

    rendered = "".join(parts)
    rendered = re.sub(r"[ \t]+\n", "\n", rendered)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered)
    rendered = re.sub(r" {2,}", " ", rendered)
    return rendered.strip()


def _resolve_href(href: str, *, base_url: str) -> str:
    target = str(href or "").strip()
    if not target or target.startswith("#"):
        return ""
    if target.lower().startswith(("javascript:", "mailto:")):
        return ""
    if not base_url:
        return target
    try:
        return urljoin(base_url, target)
    except Exception:
        return target


def _render_markdown_table(node, *, base_url: str) -> str:
    rows: list[list[str]] = []
    has_header = False
    for row_node in node.xpath(".//tr"):
        cells = row_node.xpath("./th | ./td")
        if not cells:
            continue
        row = [
            _escape_table_cell(
                _render_inline_markdown(cell, base_url=base_url).strip() or _node_text(cell),
            )
            for cell in cells
        ]
        if any(cell for cell in row):
            rows.append(row)
        if row_node.xpath("./th"):
            has_header = True
    if not rows:
        return ""

    col_count = max(len(row) for row in rows)
    padded_rows = [row + [""] * (col_count - len(row)) for row in rows]
    if has_header:
        header = padded_rows[0]
        body = padded_rows[1:]
    else:
        header = [f"Column {idx + 1}" for idx in range(col_count)]
        body = padded_rows

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * col_count) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _escape_table_cell(value: str) -> str:
    return str(value or "").replace("|", r"\|").replace("\n", " ").strip()


def _collapse_rendered_blocks(blocks: list[str]) -> str:
    cleaned: list[str] = []
    for block in blocks:
        text = str(block or "").strip()
        if not text:
            continue
        text = re.sub(r"\n{3,}", "\n\n", text)
        if cleaned and cleaned[-1] == text:
            continue
        cleaned.append(text)
    return "\n\n".join(cleaned).strip()


def _build_extractor_prompt(
    prepared: PreparedWebContent,
    *,
    url: str,
    query: str,
) -> str:
    lines = [
        "You are extracting useful information from a fetched web page for another agent.",
        "Return JSON only with keys: summary, key_sections, key_points.",
        "summary must be a short paragraph.",
        "key_sections must be an array of concise section labels.",
        "key_points must be an array of concrete facts, statistics, requirements, or arguments.",
        "Ignore navigation, cookie banners, and visual fluff.",
        "Do not mention formatting instructions or that you are an AI.",
        f"URL: {url}",
        f"Source format: {prepared.format}",
    ]
    if prepared.title:
        lines.append(f"Title: {prepared.title}")
    if query:
        lines.append(f"Focus query: {query}")
        lines.append("Prioritize information relevant to the focus query.")
    lines.extend([
        "",
        "CONTENT:",
        prepared.text,
    ])
    return "\n".join(lines)


def _format_extractor_payload(prepared: PreparedWebContent, payload: dict) -> str:
    summary = str(payload.get("summary", "") or "").strip()
    raw_sections = payload.get("key_sections", [])
    raw_points = payload.get("key_points", [])
    sections = [
        str(item).strip()
        for item in raw_sections
        if str(item).strip()
    ][:MAX_SUMMARY_HEADINGS]
    points = [
        str(item).strip()
        for item in raw_points
        if str(item).strip()
    ][:MAX_SUMMARY_BULLETS]
    lines: list[str] = []
    if prepared.title:
        lines.append(f"Title: {prepared.title}")
    if summary:
        lines.append(summary)
    if sections:
        lines.append("Key sections:")
        lines.extend(f"- {item}" for item in sections)
    if points:
        lines.append("Extracted facts and arguments:")
        lines.extend(f"- {item}" for item in points)
    return "\n".join(lines).strip()


async def _render_textual_web_content(
    prepared: PreparedWebContent,
    *,
    url: str,
    query: str,
    summarizer: _WebFetchExtractor | None = None,
) -> RenderedWebContent:
    clean_query = str(query or "").strip()
    estimated = estimate_tokens(prepared.text)
    if estimated <= DIRECT_TEXT_TOKEN_LIMIT:
        return RenderedWebContent(
            output=_wrap_xml_payload(
                "source_content",
                prepared.text,
                url=url,
                format=prepared.format,
                strategy="full_content",
                estimated_tokens=estimated,
                title=prepared.title,
            ),
            strategy="full_content",
            estimated_tokens=estimated,
            output_format=prepared.format,
        )

    if estimated <= SUMMARY_TEXT_TOKEN_LIMIT:
        summary = None
        if summarizer is not None:
            summary = await summarizer.summarize_medium(
                prepared,
                url=url,
                query=clean_query,
            )
        if not summary:
            summary = _build_medium_summary(prepared, query=clean_query)
        return RenderedWebContent(
            output=_wrap_xml_payload(
                "source_summary",
                summary,
                url=url,
                format="text",
                strategy="extracted_summary",
                estimated_tokens=estimated,
                title=prepared.title,
            ),
            strategy="extracted_summary",
            estimated_tokens=estimated,
            output_format="text",
        )

    if clean_query:
        summary = _build_query_snippets(prepared, query=clean_query)
        strategy = "query_snippets"
        tag = "source_snippets"
    else:
        summary = _build_head_tail_excerpt(prepared)
        strategy = "head_tail_excerpt"
        tag = "source_excerpt"
    return RenderedWebContent(
        output=_wrap_xml_payload(
            tag,
            summary,
            url=url,
            format="text",
            strategy=strategy,
            estimated_tokens=estimated,
            title=prepared.title,
        ),
        strategy=strategy,
        estimated_tokens=estimated,
        output_format="text",
    )


def _wrap_xml_payload(tag: str, body: str, **attrs: object) -> str:
    attr_text = " ".join(
        f'{key}="{html_lib.escape(str(value), quote=True)}"'
        for key, value in attrs.items()
        if str(value or "").strip()
    )
    attr_text = f" {attr_text}" if attr_text else ""
    payload = str(body or "").replace("]]>", "]]]]><![CDATA[>")
    return f"<{tag}{attr_text}>\n<![CDATA[\n{payload}\n]]>\n</{tag}>"


def _split_text_blocks(text: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n{2,}", str(text or "")) if block.strip()]
    return blocks or [str(text or "").strip()]


def _collect_ranked_units(
    prepared: PreparedWebContent,
    *,
    query: str,
) -> list[tuple[float, int, str, str]]:
    units: list[tuple[float, int, str, str]] = []
    query_tokens = tokenize(query)
    title_tokens = tokenize(prepared.title)
    current_heading = ""
    blocks = _split_text_blocks(prepared.text)
    total_blocks = max(1, len(blocks))
    for idx, block in enumerate(blocks):
        if not block:
            continue
        if block.startswith("#"):
            current_heading = block.lstrip("# ").strip()
            continue
        block_tokens = tokenize(block)
        heading_tokens = tokenize(current_heading)
        score = max(0.0, 0.9 - (idx / total_blocks))
        if query_tokens:
            score += token_overlap_ratio(query_tokens, block_tokens) * 3.0
            score += jaccard_similarity(query_tokens, block_tokens)
        elif title_tokens:
            score += token_overlap_ratio(title_tokens, block_tokens) * 1.2
        if heading_tokens:
            score += token_overlap_ratio(heading_tokens, block_tokens) * 0.8 + 0.15
        if re.search(r"\b\d", block):
            score += 0.5
        if re.search(r"[%$€£]", block):
            score += 0.2
        if 80 <= len(block) <= 700:
            score += 0.45
        elif len(block) > 1_200 or len(block) < 40:
            score -= 0.35
        if block.startswith("```"):
            score -= 0.8
        if block.startswith("| "):
            score -= 0.5
        units.append((score, idx, block, current_heading))
    units.sort(key=lambda item: (-item[0], item[1]))
    return units


def _build_medium_summary(prepared: PreparedWebContent, *, query: str) -> str:
    ranked = _collect_ranked_units(prepared, query=query)
    seen: set[str] = set()
    bullets: list[str] = []
    for _score, _idx, block, heading in ranked:
        normalized = re.sub(r"\s+", " ", block).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        excerpt = _truncate_excerpt(block, max_chars=320)
        if heading:
            bullets.append(f"[{heading}] {excerpt}")
        else:
            bullets.append(excerpt)
        if len(bullets) >= MAX_SUMMARY_BULLETS:
            break
    if not bullets:
        bullets = [_truncate_excerpt(prepared.text, max_chars=320)]

    lines: list[str] = []
    if prepared.title:
        lines.append(f"Title: {prepared.title}")
    if prepared.headings:
        lines.append("Key sections:")
        lines.extend(
            f"- {heading}"
            for heading in prepared.headings[:MAX_SUMMARY_HEADINGS]
        )
    lines.append("Extracted facts and arguments:")
    lines.extend(f"- {bullet}" for bullet in bullets)
    return "\n".join(lines).strip()


def _build_query_snippets(prepared: PreparedWebContent, *, query: str) -> str:
    query_tokens = tokenize(query)
    chunks = _chunk_text(prepared.text, target_tokens=TEXT_QUERY_CHUNK_TOKENS)
    ranked: list[tuple[float, int, str]] = []
    for idx, chunk in enumerate(chunks):
        chunk_tokens = tokenize(chunk)
        score = token_overlap_ratio(query_tokens, chunk_tokens) * 3.0
        score += jaccard_similarity(query_tokens, chunk_tokens)
        score += max(0.0, 0.25 - idx * 0.02)
        if score <= 0:
            continue
        ranked.append((score, idx, chunk))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    if not ranked:
        return (
            f"No strong lexical matches found for query: {query}\n\n"
            + _build_head_tail_excerpt(prepared)
        )

    lines: list[str] = []
    if prepared.title:
        lines.append(f"Title: {prepared.title}")
    lines.append(
        "Page too long to inline in full. Returning the most relevant snippets "
        f"for query: {query}",
    )
    for idx, (score, _chunk_idx, chunk) in enumerate(ranked[:MAX_QUERY_SNIPPETS], 1):
        excerpt = _best_chunk_excerpt(chunk, query_tokens=query_tokens)
        lines.append(f"{idx}. [score {score:.2f}] {excerpt}")
    return "\n\n".join(lines).strip()


def _build_head_tail_excerpt(prepared: PreparedWebContent) -> str:
    text = prepared.text
    span = max(800, min(MAX_HEAD_TAIL_CHARS // 2, max(1, len(text) // 5)))
    head = text[:span].strip()
    tail = text[-span:].strip() if len(text) > span else ""
    lines: list[str] = []
    if prepared.title:
        lines.append(f"Title: {prepared.title}")
    lines.append(
        "Page too long to inline in full. Returning the beginning and end of the "
        "cleaned content. Pass query to web_fetch for focused snippets.",
    )
    if head:
        lines.append(head)
    if tail and tail != head:
        lines.append("[... middle omitted ...]")
        lines.append(tail)
    return "\n\n".join(lines).strip()


def _chunk_text(text: str, *, target_tokens: int) -> list[str]:
    blocks = _split_text_blocks(text)
    chunks: list[str] = []
    current: list[str] = []
    for block in blocks:
        tentative = "\n\n".join(current + [block]).strip()
        if current and estimate_tokens(tentative) > target_tokens:
            chunks.append("\n\n".join(current).strip())
            current = [block]
        else:
            current.append(block)
    if current:
        chunks.append("\n\n".join(current).strip())
    if chunks:
        return chunks

    raw = str(text or "").strip()
    if not raw:
        return []
    window = max(1, target_tokens * 4)
    overlap = min(window // 5, 400)
    pieces: list[str] = []
    start = 0
    while start < len(raw):
        end = min(len(raw), start + window)
        pieces.append(raw[start:end].strip())
        if end >= len(raw):
            break
        start = max(start + 1, end - overlap)
    return [piece for piece in pieces if piece]


def _best_chunk_excerpt(chunk: str, *, query_tokens: set[str]) -> str:
    best_text = chunk
    best_score = float("-inf")
    for block in _split_text_blocks(chunk):
        score = token_overlap_ratio(query_tokens, tokenize(block)) * 3.0
        score += jaccard_similarity(query_tokens, tokenize(block))
        if score > best_score:
            best_score = score
            best_text = block
    return _truncate_excerpt(best_text, max_chars=MAX_SNIPPET_CHARS, query_tokens=query_tokens)


def _truncate_excerpt(
    text: str,
    *,
    max_chars: int,
    query_tokens: set[str] | None = None,
) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    if query_tokens:
        lower = value.lower()
        positions = [
            lower.find(token.lower())
            for token in query_tokens
            if token and lower.find(token.lower()) >= 0
        ]
        if positions:
            anchor = min(positions)
            start = max(0, anchor - (max_chars // 3))
            end = min(len(value), start + max_chars)
            excerpt = value[start:end].strip()
            if start > 0:
                excerpt = "..." + excerpt
            if end < len(value):
                excerpt += "..."
            return excerpt
    head = max_chars - 18
    if head <= 0:
        return value[:max_chars]
    return value[:head].rstrip() + " ...[excerpt]"
