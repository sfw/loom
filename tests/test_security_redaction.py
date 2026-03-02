"""Security-focused redaction tests for OAuth surfaces."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from loom.integrations.mcp.oauth import oauth_state_for_alias, redact_oauth_error_text
from loom.oauth.engine import OAuthEngine, OAuthEngineError, OAuthProviderConfig


@contextmanager
def _failing_token_server():
    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            payload = (
                b'{"error":"invalid_grant",'
                b'"error_description":"access_token=abc refresh_token=def '
                b'Authorization=Bearer xyz"}'
            )
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args):  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/token"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def test_mcp_oauth_redaction_masks_sensitive_artifacts():
    redacted = redact_oauth_error_text(
        "access_token=abc refresh_token=def Authorization=Bearer xyz"
    )
    assert "abc" not in redacted
    assert "def" not in redacted
    assert "xyz" not in redacted
    assert "<redacted>" in redacted


def test_oauth_state_redacts_stored_failure_reason(tmp_path: Path):
    store_path = tmp_path / "mcp_oauth_tokens.json"
    store_path.write_text(
        (
            '{"aliases":{"demo":{"access_token":"token-123",'
            '"last_failure_reason":"Authorization=Bearer secret-token"}}}'
        ),
        encoding="utf-8",
    )
    state = oauth_state_for_alias("demo", store_path=store_path)
    assert "secret-token" not in str(state["last_failure_reason"])
    assert "<redacted>" in str(state["last_failure_reason"])


def test_engine_error_message_redacts_token_material():
    with _failing_token_server() as token_url:
        engine = OAuthEngine(auth_ttl_seconds=60)
        provider = OAuthProviderConfig(
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint=token_url,
            client_id="loom-cli",
        )
        started = engine.start_auth(
            provider=provider,
            preferred_port=0,
            open_browser=False,
            allow_manual_fallback=False,
        )
        engine.submit_callback_input(state=started.state, raw_input="code-1")
        callback = engine.await_callback(state=started.state, timeout_seconds=5)

        with pytest.raises(OAuthEngineError) as exc:
            engine.finish_auth(
                provider=provider,
                state=started.state,
                callback=callback,
                timeout_seconds=5,
            )
        message = str(exc.value)
        assert "abc" not in message
        assert "def" not in message
        assert "xyz" not in message

        engine.shutdown()
