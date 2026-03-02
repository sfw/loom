"""Tests for shared OAuth engine lifecycle and token exchange behavior."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import pytest

from loom.oauth.engine import OAuthEngine, OAuthEngineError, OAuthProviderConfig


@contextmanager
def _token_server(*, status_code: int = 200, payload: dict | None = None):
    captured: list[dict[str, str]] = []
    response_payload = dict(payload or {"access_token": "token-123", "token_type": "Bearer"})

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8", errors="replace")
            parsed = {
                key: values[0]
                for key, values in parse_qs(body, keep_blank_values=True).items()
                if values
            }
            captured.append(parsed)
            encoded = json.dumps(response_payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, _format: str, *_args):  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/token"
        yield url, captured
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def test_engine_manual_callback_completes_token_exchange():
    with _token_server(payload={
        "access_token": "fresh-token",
        "refresh_token": "refresh-123",
        "token_type": "Bearer",
        "expires_in": 120,
        "scope": "read write",
    }) as (token_url, captured):
        engine = OAuthEngine(auth_ttl_seconds=60)
        provider = OAuthProviderConfig(
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint=token_url,
            client_id="loom-cli",
            scopes=("read",),
        )
        started = engine.start_auth(
            provider=provider,
            preferred_port=0,
            open_browser=False,
            allow_manual_fallback=False,
        )
        assert started.state
        assert started.authorization_url.startswith("https://auth.example.com/authorize?")
        assert "code_challenge=" in started.authorization_url

        engine.submit_callback_input(
            state=started.state,
            raw_input="auth-code-xyz",
        )
        callback = engine.await_callback(state=started.state, timeout_seconds=5)
        assert callback.code == "auth-code-xyz"

        token_payload = engine.finish_auth(
            provider=provider,
            state=started.state,
            callback=callback,
            timeout_seconds=5,
        )
        assert token_payload["access_token"] == "fresh-token"
        assert token_payload["refresh_token"] == "refresh-123"
        assert engine.pending_count == 0

        assert len(captured) == 1
        posted = captured[0]
        assert posted["grant_type"] == "authorization_code"
        assert posted["code"] == "auth-code-xyz"
        assert posted["client_id"] == "loom-cli"
        assert posted["code_verifier"]

        engine.shutdown()


def test_engine_timeout_clears_pending_state():
    with _token_server() as (token_url, _captured):
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

        with pytest.raises(OAuthEngineError) as exc:
            engine.await_callback(state=started.state, timeout_seconds=1)
        assert exc.value.reason_code == "callback_timeout"
        assert engine.pending_count == 0

        with pytest.raises(OAuthEngineError) as replay_exc:
            engine.submit_callback_input(
                state=started.state,
                raw_input="late-code",
            )
        assert replay_exc.value.reason_code in {"state_expired", "state_unknown"}

        engine.shutdown()


def test_engine_cancel_marks_state_cancelled():
    with _token_server() as (token_url, _captured):
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

        engine.cancel_auth(state=started.state)
        assert engine.pending_count == 0

        with pytest.raises(OAuthEngineError) as exc:
            engine.await_callback(state=started.state, timeout_seconds=1)
        assert exc.value.reason_code == "auth_cancelled"

        engine.shutdown()


def test_engine_multiple_pending_states_are_isolated():
    with _token_server() as (token_url, _captured):
        engine = OAuthEngine(auth_ttl_seconds=60)
        provider = OAuthProviderConfig(
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint=token_url,
            client_id="loom-cli",
        )
        first = engine.start_auth(
            provider=provider,
            preferred_port=0,
            open_browser=False,
            allow_manual_fallback=False,
        )
        second = engine.start_auth(
            provider=provider,
            preferred_port=0,
            open_browser=False,
            allow_manual_fallback=False,
        )
        assert first.state != second.state

        engine.submit_callback_input(state=second.state, raw_input="code-2")
        callback_second = engine.await_callback(state=second.state, timeout_seconds=5)
        assert callback_second.code == "code-2"

        engine.cancel_auth(state=first.state)
        with pytest.raises(OAuthEngineError) as exc:
            engine.await_callback(state=first.state, timeout_seconds=1)
        assert exc.value.reason_code == "auth_cancelled"
        engine.shutdown()
