from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_smoke_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_macos_desktop_bundle.py"
    spec = importlib.util.spec_from_file_location("smoke_macos_desktop_bundle", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load smoke script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load_smoke_module()


class _DummyResponse:
    def __init__(self, payload: dict[str, object], *, status: int = 200) -> None:
        self.status = status
        self._payload = payload

    def read(self) -> bytes:
        return smoke.json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_wait_for_runtime_returns_payload_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        smoke.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _DummyResponse({"status": "ok", "ready": True}),
    )

    payload = smoke.wait_for_runtime("http://127.0.0.1:9000", 0.1)

    assert payload == {"status": "ok", "ready": True}


def test_wait_for_runtime_fails_fast_when_process_exits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "loomd.log"
    log_path.write_text("boom from packaged runtime\n", encoding="utf-8")

    class FakeProcess:
        def poll(self) -> int:
            return 17

    monkeypatch.setattr(
        smoke.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("urlopen should not be called after child exit"),
    )

    with pytest.raises(SystemExit) as exc_info:
        smoke.wait_for_runtime(
            "http://127.0.0.1:9000",
            0.1,
            process=FakeProcess(),
            log_path=log_path,
        )

    message = str(exc_info.value)
    assert "process exited before /runtime became ready with code 17" in message
    assert "boom from packaged runtime" in message


def test_wait_for_runtime_timeout_includes_last_error_and_log_tail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "loomd.log"
    log_path.write_text("still booting\n", encoding="utf-8")
    times = iter([0.0, 0.0, 0.2])

    def fake_urlopen(*args, **kwargs):
        raise smoke.urllib.error.URLError("connection refused")

    monkeypatch.setattr(smoke.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(smoke.time, "sleep", lambda _: None)
    monkeypatch.setattr(smoke.time, "monotonic", lambda: next(times))

    with pytest.raises(SystemExit) as exc_info:
        smoke.wait_for_runtime(
            "http://127.0.0.1:9000",
            0.1,
            log_path=log_path,
        )

    message = str(exc_info.value)
    assert "timed out after 0.1s" in message
    assert "connection refused" in message
    assert "still booting" in message
