"""Tests for the web fetch tool."""

from __future__ import annotations

from loom.tools.web import _strip_html, is_safe_url

# --- URL Safety ---


class TestURLSafety:
    def test_blocks_localhost(self):
        safe, reason = is_safe_url("http://localhost:8080/api")
        assert not safe
        assert "Blocked" in reason

    def test_blocks_127(self):
        safe, reason = is_safe_url("http://127.0.0.1/secret")
        assert not safe

    def test_blocks_10_network(self):
        safe, reason = is_safe_url("http://10.0.0.1/internal")
        assert not safe

    def test_blocks_172_network(self):
        safe, reason = is_safe_url("http://172.16.0.1/admin")
        assert not safe

    def test_blocks_192_168(self):
        safe, reason = is_safe_url("http://192.168.1.1/config")
        assert not safe

    def test_blocks_ftp(self):
        safe, reason = is_safe_url("ftp://example.com/file")
        assert not safe
        assert "http" in reason.lower()

    def test_blocks_no_scheme(self):
        safe, reason = is_safe_url("example.com/page")
        assert not safe

    def test_allows_https(self):
        safe, reason = is_safe_url("https://example.com/api")
        assert safe
        assert reason == ""

    def test_allows_http(self):
        safe, reason = is_safe_url("http://example.com/page")
        assert safe

    def test_allows_public_ip(self):
        safe, reason = is_safe_url("http://8.8.8.8/dns")
        assert safe

    def test_blocks_zero_address(self):
        safe, reason = is_safe_url("http://0.0.0.0/")
        assert not safe


# --- HTML Stripping ---


class TestStripHtml:
    def test_removes_tags(self):
        html = "<p>Hello <b>world</b></p>"
        result = _strip_html(html)
        assert "Hello" in result
        assert "world" in result
        assert "<" not in result

    def test_removes_script(self):
        html = "<script>alert('xss')</script><p>Safe</p>"
        result = _strip_html(html)
        assert "alert" not in result
        assert "Safe" in result

    def test_removes_style(self):
        html = "<style>body{color:red}</style><p>Content</p>"
        result = _strip_html(html)
        assert "color" not in result
        assert "Content" in result

    def test_decodes_entities(self):
        html = "&amp; &lt; &gt; &quot; &#39;"
        result = _strip_html(html)
        assert "&" in result
        assert "<" in result
        assert ">" in result

    def test_collapses_whitespace(self):
        html = "<p>Hello</p>\n\n\n<p>World</p>"
        result = _strip_html(html)
        # Should not have excessive whitespace
        assert "\n\n\n" not in result

    def test_empty_input(self):
        assert _strip_html("") == ""


# --- DNS-based SSRF check ---


class TestSSRFDnsResolution:
    def test_blocks_ipv6_loopback(self):
        safe, reason = is_safe_url("http://[::1]:8080/")
        assert not safe
        assert "Blocked" in reason

    def test_private_ip_helper(self):
        from loom.tools.web import _is_private_ip
        assert _is_private_ip("127.0.0.1")
        assert _is_private_ip("10.0.0.1")
        assert _is_private_ip("192.168.1.1")
        assert _is_private_ip("172.16.0.1")
        assert _is_private_ip("::1")
        assert not _is_private_ip("8.8.8.8")
