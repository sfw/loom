"""Local loopback callback listener for OAuth authorization_code flows."""

from __future__ import annotations

import errno
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    from collections.abc import Callable


class OAuthLoopbackError(Exception):
    """Raised when loopback callback listener cannot be started or used."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class OAuthLoopbackResult:
    """Result returned by callback registration handler."""

    accepted: bool
    reason_code: str


class OAuthLoopbackServer:
    """Loopback HTTP server bound to 127.0.0.1 only."""

    def __init__(
        self,
        *,
        callback_path: str = "/oauth/callback",
        host: str = "127.0.0.1",
        port: int = 8765,
        on_callback: Callable[[dict[str, str]], OAuthLoopbackResult],
    ) -> None:
        clean_host = str(host or "").strip()
        if clean_host != "127.0.0.1":
            raise OAuthLoopbackError(
                "loopback_host_invalid",
                "Loopback listener must bind only to 127.0.0.1.",
            )
        self._host = clean_host
        self._port = int(port)
        self._path = "/" + str(callback_path or "").lstrip("/")
        self._on_callback = on_callback
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        with self._lock:
            if self._server is None:
                return self._port
            return int(self._server.server_address[1])

    @property
    def callback_path(self) -> str:
        return self._path

    @property
    def redirect_uri(self) -> str:
        return f"http://127.0.0.1:{self.port}{self._path}"

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._server is not None

    def start(self) -> None:
        with self._lock:
            if self._server is not None:
                return

            owner = self

            class _Handler(BaseHTTPRequestHandler):
                def do_GET(self) -> None:  # noqa: N802
                    parsed = urlparse(self.path)
                    if parsed.path != owner._path:
                        self._write(
                            status=404,
                            body="Not found.",
                        )
                        return
                    query = {
                        key: values[0]
                        for key, values in parse_qs(
                            parsed.query,
                            keep_blank_values=True,
                        ).items()
                        if values
                    }
                    result = owner._on_callback(query)
                    if result.accepted:
                        self._write(
                            status=200,
                            body=(
                                "Authentication received. You can close this tab "
                                "and return to Loom."
                            ),
                        )
                        return
                    self._write(
                        status=400,
                        body=f"OAuth callback rejected ({result.reason_code}).",
                    )

                def log_message(self, _format: str, *_args) -> None:  # noqa: A003
                    # Intentionally silent; callback payloads must not leak via logs.
                    return

                def _write(self, *, status: int, body: str) -> None:
                    payload = body.encode("utf-8", errors="replace")
                    self.send_response(status)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

            try:
                httpd = ThreadingHTTPServer((self._host, self._port), _Handler)
            except OSError as e:
                reason_code = (
                    "loopback_port_busy"
                    if getattr(e, "errno", None) == errno.EADDRINUSE
                    else "loopback_start_failed"
                )
                raise OAuthLoopbackError(
                    reason_code,
                    f"Loopback listener start failed: {e}",
                ) from e
            httpd.daemon_threads = True
            httpd.allow_reuse_address = False
            thread = threading.Thread(
                target=httpd.serve_forever,
                name="loom-oauth-loopback",
                daemon=True,
            )
            thread.start()
            self._server = httpd
            self._thread = thread

    def stop(self) -> None:
        with self._lock:
            server = self._server
            thread = self._thread
            self._server = None
            self._thread = None
        if server is None:
            return
        try:
            server.shutdown()
        except Exception:
            pass
        try:
            server.server_close()
        except Exception:
            pass
        if thread is not None:
            thread.join(timeout=1.0)
