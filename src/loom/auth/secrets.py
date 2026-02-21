"""Secret reference resolution helpers for auth runtime."""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse

_ENV_REF_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class SecretResolutionError(Exception):
    """Raised when a secret reference cannot be resolved."""


class SecretResolver:
    """Resolve `env://...`, `${ENV_VAR}`, and `keychain://...` references."""

    def resolve(self, secret_ref: str) -> str:
        """Resolve one secret reference into a runtime value."""
        raw = str(secret_ref or "").strip()
        if not raw:
            raise SecretResolutionError("Secret reference is empty.")

        env_match = _ENV_REF_RE.match(raw)
        if env_match is not None:
            return self._resolve_env_var(env_match.group(1), raw_ref=raw)

        parsed = urlparse(raw)
        scheme = parsed.scheme.lower()
        if scheme == "env":
            env_name = (parsed.netloc or parsed.path.lstrip("/")).strip()
            if not env_name:
                raise SecretResolutionError(
                    f"Invalid env secret reference {raw!r}: missing env variable name."
                )
            return self._resolve_env_var(env_name, raw_ref=raw)
        if scheme == "keychain":
            return self._resolve_keychain(raw, parsed=parsed)

        raise SecretResolutionError(
            f"Unsupported secret reference scheme in {raw!r}. "
            "Supported: env://..., ${ENV_VAR}, keychain://..."
        )

    def resolve_maybe(self, value: str) -> str:
        """Resolve value when it looks like a secret ref; otherwise pass through."""
        raw = str(value)
        stripped = raw.strip()
        if _ENV_REF_RE.match(stripped):
            return self.resolve(stripped)
        lowered = stripped.lower()
        if lowered.startswith("env://") or lowered.startswith("keychain://"):
            return self.resolve(stripped)
        return raw

    @staticmethod
    def _resolve_env_var(env_name: str, *, raw_ref: str) -> str:
        resolved = os.environ.get(env_name)
        if resolved is None:
            raise SecretResolutionError(
                f"Environment reference {raw_ref!r} could not be resolved; "
                f"missing env var {env_name!r}."
            )
        return resolved

    @staticmethod
    def _resolve_keychain(raw_ref: str, *, parsed) -> str:
        service = str(parsed.netloc or "").strip()
        path_parts = [part for part in parsed.path.split("/") if part]
        if not service:
            raise SecretResolutionError(
                f"Invalid keychain reference {raw_ref!r}: missing service."
            )
        if not path_parts:
            raise SecretResolutionError(
                f"Invalid keychain reference {raw_ref!r}: missing account path."
            )
        account = "/".join(path_parts)
        try:
            import keyring  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover - import path dependent
            raise SecretResolutionError(
                "Keychain secret resolution requires the `keyring` package. "
                "Install it or switch to env:// references."
            ) from e

        try:
            secret = keyring.get_password(service, account)
        except Exception as e:
            raise SecretResolutionError(
                f"Keychain lookup failed for {raw_ref!r}: {e}"
            ) from e
        if secret is None:
            raise SecretResolutionError(
                f"No keychain secret found for {raw_ref!r}."
            )
        return secret

