"""Web fetch tool for retrieving URL content.

Fetches a URL and returns text content. Includes safety controls:
timeout, max response size, and URL validation.
"""

from __future__ import annotations

import asyncio
import html as html_lib
import ipaddress
import os
import re
import socket

import httpx

from loom.ingest.artifacts import (
    DEFAULT_RETENTION_MAX_AGE_DAYS,
    DEFAULT_RETENTION_MAX_BYTES_PER_SCOPE,
    DEFAULT_RETENTION_MAX_FILES_PER_SCOPE,
    persist_fetch_artifact,
)
from loom.ingest.handlers import summarize_artifact
from loom.ingest.router import ContentKind, detect_content_kind, normalize_media_type
from loom.tools.registry import Tool, ToolContext, ToolResult

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
        from urllib.parse import urlparse
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
                from urllib.parse import urljoin
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

                # Strip HTML for text-oriented fetches.
                # Some servers mislabel HTML as text/plain, so detect by content too.
                if extract_text and (
                    content_kind == ContentKind.HTML
                    or _looks_like_html(content, media_type)
                ):
                    content = _strip_html(content)

                if truncation_notes:
                    content += "\n\n... (" + "; ".join(truncation_notes) + ")"

                return ToolResult.ok(
                    content,
                    data={
                        "url": resolved_url,
                        "status_code": status_code,
                        "content_type": media_type or content_type,
                        "content_kind": content_kind,
                        "size_bytes": len(content),
                        "declared_size_bytes": declared_size,
                        "truncated": bool(truncation_notes),
                        "extract_text": extract_text,
                    },
                )

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
    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch content from a URL and return plain text. "
            "HTML markup is stripped by default. "
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
            extract_text=True,
            max_download_bytes=MAX_DOWNLOAD_BYTES,
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
