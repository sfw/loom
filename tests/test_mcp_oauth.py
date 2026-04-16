"""Tests for remote MCP OAuth discovery/provider resolution."""

from __future__ import annotations

import json
from typing import Any

import httpx

from loom.config import MCPOAuthConfig, MCPServerConfig
from loom.integrations.mcp.oauth import (
    bearer_auth_header_for_alias,
    fingerprint_mcp_server,
    resolve_mcp_oauth_provider,
    upsert_mcp_oauth_token,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None) -> None:
        self.status_code = int(status_code)
        self._payload = dict(payload or {})
        self.content = b"{}" if payload is not None else b""
        self.text = "{}" if payload is not None else ""

    def json(self) -> dict[str, Any]:
        return dict(self._payload)


def _patch_fake_mcp_secret_resolver(monkeypatch) -> dict[str, str]:
    import loom.integrations.mcp.oauth as mcp_oauth_mod

    stored: dict[str, str] = {}

    class _FakeSecretResolver:
        def validate_writable(self, _secret_ref: str) -> None:
            return

        def store(self, secret_ref: str, secret_value: str) -> None:
            stored[secret_ref] = secret_value

        def resolve(self, secret_ref: str) -> str:
            return stored.get(secret_ref, "")

    monkeypatch.setattr(mcp_oauth_mod, "SecretResolver", _FakeSecretResolver)
    return stored


def test_resolve_mcp_oauth_provider_registers_client_when_available(monkeypatch):
    captured_post: dict[str, Any] = {}

    def _fake_get(url: str, **_kwargs: object) -> _FakeResponse:
        if url.endswith("/.well-known/oauth-protected-resource"):
            return _FakeResponse(
                200,
                {
                    "authorization_servers": ["https://auth.example.com"],
                },
            )
        if url == "https://auth.example.com/.well-known/oauth-authorization-server":
            return _FakeResponse(
                200,
                {
                    "authorization_endpoint": "https://auth.example.com/oauth2/authorize",
                    "token_endpoint": "https://auth.example.com/oauth2/token",
                    "registration_endpoint": "https://auth.example.com/oauth2/register",
                },
            )
        return _FakeResponse(404, {})

    def _fake_post(
        url: str,
        *,
        json: dict[str, Any] | None = None,
        **_kwargs: object,
    ) -> _FakeResponse:
        captured_post["url"] = url
        captured_post["json"] = dict(json or {})
        return _FakeResponse(
            201,
            {
                "client_id": "registered-client",
                "client_secret": "registered-secret",
            },
        )

    monkeypatch.delenv("LOOM_MCP_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.setattr(httpx, "get", _fake_get)
    monkeypatch.setattr(httpx, "post", _fake_post)

    provider = resolve_mcp_oauth_provider(
        server_url="https://mcp.example.com/mcp",
        scopes=["read:content"],
        redirect_uris=("http://127.0.0.1:8765/oauth/callback",),
    )

    assert provider.client_id == "registered-client"
    assert provider.authorize_params == {"resource": "https://mcp.example.com/mcp"}
    assert provider.token_params == {
        "client_secret": "registered-secret",
        "resource": "https://mcp.example.com/mcp",
    }
    assert captured_post["url"] == "https://auth.example.com/oauth2/register"
    assert captured_post["json"]["token_endpoint_auth_method"] == "none"
    assert captured_post["json"]["redirect_uris"] == ["http://127.0.0.1:8765/oauth/callback"]


def test_resolve_mcp_oauth_provider_falls_back_to_default_client_id(monkeypatch):
    def _fake_get(url: str, **_kwargs: object) -> _FakeResponse:
        if url == "https://mcp.example.com/.well-known/oauth-authorization-server":
            return _FakeResponse(
                200,
                {
                    "authorization_endpoint": "https://auth.example.com/oauth2/authorize",
                    "token_endpoint": "https://auth.example.com/oauth2/token",
                },
            )
        return _FakeResponse(404, {})

    def _fail_post(*_args: object, **_kwargs: object) -> _FakeResponse:
        raise AssertionError("client registration should not be called")

    monkeypatch.delenv("LOOM_MCP_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.setattr(httpx, "get", _fake_get)
    monkeypatch.setattr(httpx, "post", _fail_post)

    provider = resolve_mcp_oauth_provider(
        server_url="https://mcp.example.com/mcp",
        scopes=[],
    )

    assert provider.client_id == "loom-cli"
    assert provider.token_params == {}


def test_resolve_mcp_oauth_provider_uses_explicit_resource_indicator(monkeypatch):
    def _fake_get(url: str, **_kwargs: object) -> _FakeResponse:
        if url.endswith("/.well-known/oauth-protected-resource"):
            return _FakeResponse(
                200,
                {
                    "authorization_server": "https://auth.example.com",
                    "resource": "https://resource.example.com",
                },
            )
        if url == "https://auth.example.com/.well-known/oauth-authorization-server":
            return _FakeResponse(
                200,
                {
                    "authorization_endpoint": "https://auth.example.com/oauth2/authorize",
                    "token_endpoint": "https://auth.example.com/oauth2/token",
                },
            )
        return _FakeResponse(404, {})

    monkeypatch.delenv("LOOM_MCP_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.setattr(httpx, "get", _fake_get)

    provider = resolve_mcp_oauth_provider(
        server_url="https://mcp.example.com/mcp",
        scopes=[],
    )

    assert provider.authorize_params == {"resource": "https://resource.example.com"}
    assert provider.token_params == {"resource": "https://resource.example.com"}


def test_bearer_auth_header_normalizes_bearer_token_type(tmp_path, monkeypatch):
    _patch_fake_mcp_secret_resolver(monkeypatch)
    store = tmp_path / "mcp_oauth_tokens.json"
    server = MCPServerConfig(
        type="remote",
        url="https://example.com/mcp",
        oauth=MCPOAuthConfig(enabled=True),
    )
    upsert_mcp_oauth_token(
        alias="demo",
        server=server,
        access_token="token-123",
        token_type="bearer",
        store_path=store,
    )
    header = bearer_auth_header_for_alias("demo", server=server, store_path=store)
    assert header == "Bearer token-123"


def test_upsert_mcp_oauth_token_binds_credentials_by_server_fingerprint(
    tmp_path,
    monkeypatch,
):
    secret_store = _patch_fake_mcp_secret_resolver(monkeypatch)
    store = tmp_path / "mcp_oauth_tokens.json"
    server = MCPServerConfig(
        type="remote",
        url="https://example.com/mcp",
        headers={"X-Tenant": "acme"},
        oauth=MCPOAuthConfig(enabled=True, scopes=["read:content"]),
    )

    upsert_mcp_oauth_token(
        alias="demo",
        server=server,
        access_token="token-123",
        token_type="Bearer",
        store_path=store,
    )

    raw = json.loads(store.read_text(encoding="utf-8"))
    server_fingerprint = fingerprint_mcp_server(server)
    assert raw["alias_bindings"]["demo"]["server_fingerprint"] == server_fingerprint
    token_ref = raw["servers"][server_fingerprint]["token_ref"]
    assert json.loads(secret_store[token_ref])["access_token"] == "token-123"

    header = bearer_auth_header_for_alias("renamed_demo", server=server, store_path=store)
    assert header == "Bearer token-123"


def test_legacy_alias_token_migrates_to_secret_ref_on_lookup(tmp_path, monkeypatch):
    secret_store = _patch_fake_mcp_secret_resolver(monkeypatch)
    store = tmp_path / "mcp_oauth_tokens.json"
    store.write_text(
        json.dumps(
            {
                "aliases": {
                    "demo": {
                        "access_token": "legacy-token",
                        "token_type": "Bearer",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    server = MCPServerConfig(
        type="remote",
        url="https://example.com/mcp",
        oauth=MCPOAuthConfig(enabled=True),
    )

    header = bearer_auth_header_for_alias("demo", server=server, store_path=store)

    assert header == "Bearer legacy-token"
    raw = json.loads(store.read_text(encoding="utf-8"))
    server_fingerprint = raw["alias_bindings"]["demo"]["server_fingerprint"]
    token_ref = raw["servers"][server_fingerprint]["token_ref"]
    assert json.loads(secret_store[token_ref])["access_token"] == "legacy-token"
