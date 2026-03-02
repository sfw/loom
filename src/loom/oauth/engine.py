"""Shared OAuth PKCE + loopback engine for MCP and /auth flows."""

from __future__ import annotations

import base64
import hashlib
import re
import secrets
import time
import webbrowser
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from loom.oauth.loopback import OAuthLoopbackError, OAuthLoopbackResult, OAuthLoopbackServer
from loom.oauth.state_store import OAuthPendingState, OAuthStateStore, OAuthStateStoreError

_REDACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?i)\b(access_token|refresh_token|id_token|client_secret|code_verifier)\b\s*[:=]\s*([^\s,;]+)"
        ),
        r"\1=<redacted>",
    ),
    (
        re.compile(
            r"(?i)\b(authorization|proxy-authorization)\b\s*[:=]\s*"
            r"([^\s,;]+(?:\s+[^\s,;]+)?)"
        ),
        r"\1=<redacted>",
    ),
    (
        re.compile(r"(?i)\b(Bearer)\s+[A-Za-z0-9._~+/=-]+"),
        r"\1 <redacted>",
    ),
)


class OAuthEngineError(Exception):
    """Raised when OAuth browser flow cannot proceed safely."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class OAuthProviderConfig:
    """Provider metadata required to run authorization_code + PKCE."""

    authorization_endpoint: str
    token_endpoint: str
    client_id: str
    scopes: tuple[str, ...] = ()
    authorize_params: dict[str, str] = field(default_factory=dict)
    token_params: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OAuthStartResult:
    """Information returned by start_auth for UI/CLI rendering."""

    state: str
    authorization_url: str
    redirect_uri: str
    expires_at_unix: int
    callback_mode: str
    loopback_enabled: bool
    browser_opened: bool
    browser_error: str = ""


@dataclass(frozen=True)
class OAuthCallbackResult:
    """Normalized callback payload for finish_auth."""

    state: str
    code: str = ""
    error: str = ""
    error_description: str = ""
    payload: dict[str, str] = field(default_factory=dict)


class OAuthEngine:
    """Deterministic OAuth flow engine with strict pending-state lifecycle."""

    def __init__(
        self,
        *,
        callback_path: str = "/oauth/callback",
        auth_ttl_seconds: int = 300,
        http_timeout_seconds: int = 20,
        state_store: OAuthStateStore | None = None,
    ) -> None:
        self._state_store = state_store or OAuthStateStore()
        self._callback_path = callback_path
        self._auth_ttl_seconds = max(30, int(auth_ttl_seconds))
        self._http_timeout_seconds = max(1, int(http_timeout_seconds))
        self._loopback_server: OAuthLoopbackServer | None = None

    @property
    def pending_count(self) -> int:
        return self._state_store.pending_count

    def start_auth(
        self,
        *,
        provider: OAuthProviderConfig,
        preferred_port: int = 8765,
        open_browser: bool = True,
        allow_manual_fallback: bool = True,
        manual_redirect_uri: str = "urn:ietf:wg:oauth:2.0:oob",
    ) -> OAuthStartResult:
        clean_provider = self._validate_provider(provider)
        state = self._generate_state()
        code_verifier = self._generate_code_verifier()
        code_challenge = self._pkce_challenge(code_verifier)
        created = time.monotonic()
        expires_at_unix = int(time.time()) + self._auth_ttl_seconds

        callback_mode = "loopback"
        loopback_enabled = False
        redirect_uri = str(manual_redirect_uri or "").strip() or "urn:ietf:wg:oauth:2.0:oob"

        try:
            self._ensure_loopback_server(preferred_port=preferred_port)
            if self._loopback_server is not None:
                redirect_uri = self._loopback_server.redirect_uri
                loopback_enabled = True
        except OAuthLoopbackError as e:
            if not allow_manual_fallback:
                raise OAuthEngineError(
                    "loopback_unavailable",
                    self._redact(str(e)),
                ) from e
            callback_mode = "manual"
            loopback_enabled = False

        try:
            self._state_store.create_pending(
                state=state,
                code_verifier=code_verifier,
                redirect_uri=redirect_uri,
                ttl_seconds=self._auth_ttl_seconds,
                metadata={
                    "authorization_endpoint": clean_provider.authorization_endpoint,
                    "token_endpoint": clean_provider.token_endpoint,
                    "client_id": clean_provider.client_id,
                    "created_monotonic": str(created),
                },
            )
        except OAuthStateStoreError as e:
            raise OAuthEngineError(e.reason_code, self._redact(str(e))) from e

        authorization_url = self._build_authorization_url(
            provider=clean_provider,
            state=state,
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
        )
        browser_opened = False
        browser_error = ""
        if open_browser:
            try:
                browser_opened = bool(webbrowser.open(authorization_url, new=2))
            except Exception as e:
                browser_opened = False
                browser_error = self._redact(str(e))

        return OAuthStartResult(
            state=state,
            authorization_url=authorization_url,
            redirect_uri=redirect_uri,
            expires_at_unix=expires_at_unix,
            callback_mode=callback_mode,
            loopback_enabled=loopback_enabled,
            browser_opened=browser_opened,
            browser_error=browser_error,
        )

    def await_callback(
        self,
        *,
        state: str,
        timeout_seconds: int,
    ) -> OAuthCallbackResult:
        clean_state = str(state or "").strip()
        try:
            payload = self._state_store.await_callback(
                state=clean_state,
                timeout_seconds=timeout_seconds,
            )
        except OAuthStateStoreError as e:
            raise OAuthEngineError(e.reason_code, self._redact(str(e))) from e
        return self._normalize_callback(clean_state, payload)

    def submit_callback_payload(
        self,
        *,
        state: str,
        payload: dict[str, str],
    ) -> None:
        clean_state = str(state or "").strip()
        callback_payload = dict(payload)
        callback_payload.setdefault("state", clean_state)
        result = self._state_store.register_callback(
            state=clean_state,
            payload=callback_payload,
        )
        if result != "ok":
            raise OAuthEngineError(
                result,
                f"OAuth callback rejected ({result}).",
            )

    def submit_callback_input(
        self,
        *,
        state: str,
        raw_input: str,
    ) -> OAuthCallbackResult:
        clean_state = str(state or "").strip()
        parsed_payload = self.parse_callback_input(
            raw_input=raw_input,
            expected_state=clean_state,
        )
        self.submit_callback_payload(state=clean_state, payload=parsed_payload)
        return self._normalize_callback(clean_state, parsed_payload)

    def finish_auth(
        self,
        *,
        provider: OAuthProviderConfig,
        state: str,
        callback: OAuthCallbackResult | None = None,
        timeout_seconds: int = 180,
    ) -> dict[str, Any]:
        clean_provider = self._validate_provider(provider)
        clean_state = str(state or "").strip()
        try:
            pending = self._state_store.get_pending(state=clean_state)
        except OAuthStateStoreError as e:
            raise OAuthEngineError(e.reason_code, self._redact(str(e))) from e

        normalized = callback or self.await_callback(
            state=clean_state,
            timeout_seconds=timeout_seconds,
        )
        try:
            if normalized.error:
                description = normalized.error_description or "OAuth authorization failed."
                raise OAuthEngineError(
                    "oauth_authorization_failed",
                    self._redact(description),
                )
            auth_code = str(normalized.code or "").strip()
            if not auth_code:
                raise OAuthEngineError(
                    "callback_invalid",
                    "OAuth callback did not include an authorization code.",
                )
            token_payload = self._exchange_code_for_token(
                provider=clean_provider,
                pending=pending,
                authorization_code=auth_code,
            )
            return token_payload
        finally:
            self._state_store.complete(state=clean_state)

    def cancel_auth(self, *, state: str) -> None:
        self._state_store.cancel(state=str(state or "").strip())

    def shutdown(self) -> None:
        if self._loopback_server is not None:
            self._loopback_server.stop()
            self._loopback_server = None
        self._state_store.clear_all()

    def _ensure_loopback_server(self, *, preferred_port: int) -> None:
        if self._loopback_server is not None and self._loopback_server.is_running:
            return
        server = OAuthLoopbackServer(
            callback_path=self._callback_path,
            host="127.0.0.1",
            port=int(preferred_port),
            on_callback=self._on_loopback_callback,
        )
        try:
            server.start()
        except OAuthLoopbackError:
            raise
        self._loopback_server = server

    def _on_loopback_callback(self, payload: dict[str, str]) -> OAuthLoopbackResult:
        state = str(payload.get("state", "")).strip()
        if not state:
            return OAuthLoopbackResult(False, "missing_state")
        result = self._state_store.register_callback(
            state=state,
            payload=payload,
        )
        return OAuthLoopbackResult(result == "ok", result)

    def _build_authorization_url(
        self,
        *,
        provider: OAuthProviderConfig,
        state: str,
        redirect_uri: str,
        code_challenge: str,
    ) -> str:
        params: dict[str, str] = {
            "response_type": "code",
            "client_id": provider.client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        scopes = [scope for scope in provider.scopes if str(scope).strip()]
        if scopes:
            params["scope"] = " ".join(scopes)
        for key, value in provider.authorize_params.items():
            clean_key = str(key or "").strip()
            clean_value = str(value or "").strip()
            if not clean_key or not clean_value:
                continue
            params[clean_key] = clean_value
        return f"{provider.authorization_endpoint}?{urlencode(params)}"

    def _exchange_code_for_token(
        self,
        *,
        provider: OAuthProviderConfig,
        pending: OAuthPendingState,
        authorization_code: str,
    ) -> dict[str, Any]:
        payload: dict[str, str] = {
            "grant_type": "authorization_code",
            "code": authorization_code,
            "redirect_uri": pending.redirect_uri,
            "client_id": provider.client_id,
            "code_verifier": pending.code_verifier,
        }
        for key, value in provider.token_params.items():
            clean_key = str(key or "").strip()
            clean_value = str(value or "").strip()
            if not clean_key or not clean_value:
                continue
            payload[clean_key] = clean_value

        try:
            response = httpx.post(
                provider.token_endpoint,
                data=payload,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                timeout=self._http_timeout_seconds,
            )
        except Exception as e:
            raise OAuthEngineError(
                "token_exchange_failed",
                f"Token exchange request failed: {self._redact(str(e))}",
            ) from e

        body: dict[str, Any] = {}
        if response.content:
            try:
                parsed = response.json()
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                body = dict(parsed)
        if response.status_code >= 400:
            message = str(
                body.get("error_description")
                or body.get("error")
                or response.text
                or "Token exchange failed"
            ).strip()
            raise OAuthEngineError(
                "token_exchange_failed",
                self._redact(message),
            )

        access_token = str(body.get("access_token", "")).strip()
        if not access_token:
            raise OAuthEngineError(
                "token_exchange_invalid",
                "Token endpoint response missing access_token.",
            )
        token_payload = dict(body)
        token_payload.setdefault("token_type", "Bearer")
        return token_payload

    def _normalize_callback(
        self,
        state: str,
        payload: dict[str, str],
    ) -> OAuthCallbackResult:
        clean_state = str(state or "").strip()
        callback_state = str(payload.get("state", "")).strip()
        if callback_state and callback_state != clean_state:
            raise OAuthEngineError(
                "state_mismatch",
                "OAuth callback state mismatch.",
            )
        return OAuthCallbackResult(
            state=clean_state,
            code=str(payload.get("code", "")).strip(),
            error=str(payload.get("error", "")).strip(),
            error_description=str(payload.get("error_description", "")).strip(),
            payload=dict(payload),
        )

    @staticmethod
    def parse_callback_input(
        *,
        raw_input: str,
        expected_state: str,
    ) -> dict[str, str]:
        text = str(raw_input or "").strip()
        if not text:
            raise OAuthEngineError(
                "callback_missing",
                "Callback input cannot be empty.",
            )
        parsed = urlparse(text)
        if parsed.scheme and parsed.netloc:
            query = {
                key: values[0]
                for key, values in parse_qs(
                    parsed.query,
                    keep_blank_values=True,
                ).items()
                if values
            }
            query.setdefault("state", str(expected_state or "").strip())
            return query
        return {
            "state": str(expected_state or "").strip(),
            "code": text,
        }

    @staticmethod
    def _validate_provider(provider: OAuthProviderConfig) -> OAuthProviderConfig:
        authorization_endpoint = str(provider.authorization_endpoint or "").strip()
        token_endpoint = str(provider.token_endpoint or "").strip()
        client_id = str(provider.client_id or "").strip()
        if not authorization_endpoint:
            raise OAuthEngineError(
                "oauth_config_invalid",
                "Missing authorization_endpoint for OAuth flow.",
            )
        if not token_endpoint:
            raise OAuthEngineError(
                "oauth_config_invalid",
                "Missing token_endpoint for OAuth flow.",
            )
        if not client_id:
            raise OAuthEngineError(
                "oauth_config_invalid",
                "Missing client_id for OAuth flow.",
            )
        return OAuthProviderConfig(
            authorization_endpoint=authorization_endpoint,
            token_endpoint=token_endpoint,
            client_id=client_id,
            scopes=tuple(str(scope).strip() for scope in provider.scopes if str(scope).strip()),
            authorize_params={
                str(key).strip(): str(value).strip()
                for key, value in provider.authorize_params.items()
                if str(key).strip() and str(value).strip()
            },
            token_params={
                str(key).strip(): str(value).strip()
                for key, value in provider.token_params.items()
                if str(key).strip() and str(value).strip()
            },
        )

    @staticmethod
    def _generate_state() -> str:
        return secrets.token_urlsafe(32)

    @staticmethod
    def _generate_code_verifier() -> str:
        return secrets.token_urlsafe(64)

    @staticmethod
    def _pkce_challenge(verifier: str) -> str:
        digest = hashlib.sha256(verifier.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

    @staticmethod
    def _redact(value: object) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        for pattern, replacement in _REDACT_PATTERNS:
            text = pattern.sub(replacement, text)
        return text
