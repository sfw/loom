"""Web fetch tool for retrieving URL content.

Fetches a URL and returns text content. Includes safety controls:
timeout, max response size, and URL validation.
"""

from __future__ import annotations

import re

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
                follow_redirects=True,
                timeout=httpx.Timeout(FETCH_TIMEOUT),
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                content = response.text

                # Truncate if too large
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
