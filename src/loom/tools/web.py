"""Web fetch tool for retrieving URL content.

Fetches a URL and returns text content. Includes safety controls:
timeout, max response size, and URL validation.
"""

from __future__ import annotations

import ipaddress
import re
import socket

import httpx

from loom.tools.registry import Tool, ToolContext, ToolResult

# Safety: block private/internal networks
_BLOCKED_HOSTS = re.compile(
    r"^(localhost|127\.\d+\.\d+\.\d+|0\.0\.0\.0|10\.\d+\.\d+\.\d+|"
    r"172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+|\[::1\])",
    re.IGNORECASE,
)

MAX_RESPONSE_SIZE = 512 * 1024  # 512KB
FETCH_TIMEOUT = 30.0


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


class WebFetchTool(Tool):
    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch content from a URL. Returns text content. "
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
                "extract_text": {
                    "type": "boolean",
                    "description": "If true, strip HTML tags and return plain text (default: true)",
                },
            },
            "required": ["url"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 45

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        url = args.get("url", "")
        extract_text = args.get("extract_text", True)

        if not url:
            return ToolResult.fail("No URL provided")

        safe, reason = is_safe_url(url)
        if not safe:
            return ToolResult.fail(reason)

        try:
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=httpx.Timeout(FETCH_TIMEOUT),
            ) as client:
                response = await client.get(url)

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
                        return ToolResult.fail(f"Redirect blocked: {redir_reason}")
                    response = await client.get(location)

                response.raise_for_status()

                # Check Content-Length before reading body to prevent OOM
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > MAX_RESPONSE_SIZE * 4:
                    return ToolResult.fail(
                        f"Response too large ({int(content_length)} bytes). "
                        f"Max: {MAX_RESPONSE_SIZE * 4}."
                    )

                content_type = response.headers.get("content-type", "")
                content = response.text

                # Truncate if too large (even after Content-Length check, since
                # Content-Length can be absent or wrong)
                if len(content) > MAX_RESPONSE_SIZE:
                    content = content[:MAX_RESPONSE_SIZE]
                    content += "\n\n... (content truncated)"

                # Strip HTML if requested
                if extract_text and "html" in content_type.lower():
                    content = _strip_html(content)

                return ToolResult.ok(
                    content,
                    data={
                        "url": str(response.url),
                        "status_code": response.status_code,
                        "content_type": content_type,
                        "size_bytes": len(content),
                    },
                )
        except httpx.HTTPStatusError as e:
            return ToolResult.fail(f"HTTP {e.response.status_code}: {url}")
        except httpx.TimeoutException:
            return ToolResult.fail(f"Timeout fetching: {url}")
        except httpx.ConnectError:
            return ToolResult.fail(f"Connection failed: {url}")
        except Exception as e:
            return ToolResult.fail(f"Fetch error: {e}")


def _strip_html(html: str) -> str:
    """Simple HTML tag stripper. Removes tags and collapses whitespace."""
    # Remove script and style blocks
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Decode common entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    return text
