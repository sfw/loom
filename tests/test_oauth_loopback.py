"""Tests for loopback callback binding and state/replay handling."""

from __future__ import annotations

import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from loom.oauth.engine import OAuthEngine, OAuthEngineError, OAuthProviderConfig
from loom.oauth.loopback import OAuthLoopbackError, OAuthLoopbackServer


@contextmanager
def _token_server():
    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            payload = b'{"access_token":"token-123","token_type":"Bearer"}'
            self.send_response(200)
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


def test_loopback_server_rejects_non_localhost_binding():
    with pytest.raises(OAuthLoopbackError) as exc:
        OAuthLoopbackServer(
            host="0.0.0.0",
            port=8765,
            callback_path="/oauth/callback",
            on_callback=lambda _payload: None,  # type: ignore[arg-type]
        )
    assert exc.value.reason_code == "loopback_host_invalid"


def test_loopback_callback_rejects_unknown_and_replayed_state():
    with _token_server() as token_url:
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
        assert started.redirect_uri.startswith("http://127.0.0.1:")

        unknown = httpx.get(
            started.redirect_uri,
            params={"state": "unknown-state", "code": "abc"},
            timeout=5,
        )
        assert unknown.status_code == 400
        assert "state_unknown" in unknown.text

        first = httpx.get(
            started.redirect_uri,
            params={"state": started.state, "code": "ok-code"},
            timeout=5,
        )
        assert first.status_code == 200

        replay = httpx.get(
            started.redirect_uri,
            params={"state": started.state, "code": "replay-code"},
            timeout=5,
        )
        assert replay.status_code == 400
        assert "state_replayed" in replay.text

        callback = engine.await_callback(state=started.state, timeout_seconds=5)
        assert callback.code == "ok-code"
        engine.shutdown()


def test_port_collision_without_manual_fallback_raises():
    with _token_server() as token_url:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        busy_port = sock.getsockname()[1]
        try:
            engine = OAuthEngine(auth_ttl_seconds=60)
            provider = OAuthProviderConfig(
                authorization_endpoint="https://auth.example.com/authorize",
                token_endpoint=token_url,
                client_id="loom-cli",
            )
            with pytest.raises(OAuthEngineError) as exc:
                engine.start_auth(
                    provider=provider,
                    preferred_port=busy_port,
                    open_browser=False,
                    allow_manual_fallback=False,
                )
            assert exc.value.reason_code == "loopback_unavailable"
        finally:
            sock.close()


def test_port_collision_with_manual_fallback_uses_manual_mode():
    with _token_server() as token_url:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        busy_port = sock.getsockname()[1]
        try:
            engine = OAuthEngine(auth_ttl_seconds=60)
            provider = OAuthProviderConfig(
                authorization_endpoint="https://auth.example.com/authorize",
                token_endpoint=token_url,
                client_id="loom-cli",
            )
            started = engine.start_auth(
                provider=provider,
                preferred_port=busy_port,
                open_browser=False,
                allow_manual_fallback=True,
            )
            assert started.callback_mode == "manual"
            assert started.loopback_enabled is False
            assert started.redirect_uri == "urn:ietf:wg:oauth:2.0:oob"
            engine.cancel_auth(state=started.state)
            engine.shutdown()
        finally:
            sock.close()
