"""Tests for bundled google-analytics live API tool."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from loom.auth.runtime import build_run_auth_context
from loom.tools.registry import ToolContext


def _load_ga_live_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "packages" / "google-analytics" / "tools" / "ga_live_api.py"
    module_name = "tests._ga_live_api_bundle"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeAsyncClient:
    calls: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}

    def __init__(self, *args, **kwargs):
        del args, kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False

    async def request(self, *, method, url, headers, params, json):
        self.__class__.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "params": dict(params),
                "json": json,
            }
        )
        return _FakeResponse(200, self.__class__.payload)


def _build_auth_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REAL_GA_ACCESS_TOKEN", "ya29.mock-token")
    auth_path = tmp_path / "auth.toml"
    auth_path.write_text(
        """
[auth.profiles.ga_prod]
provider = "google_analytics"
mode = "env_passthrough"

[auth.profiles.ga_prod.env]
GA_ACCESS_TOKEN = "${REAL_GA_ACCESS_TOKEN}"
GA_PROPERTY_ID = "123456789"
""",
        encoding="utf-8",
    )
    return build_run_auth_context(
        workspace=tmp_path,
        metadata={"auth_config_path": str(auth_path)},
        required_resources=[{"provider": "google_analytics", "source": "api"}],
    )


class TestGALiveApiTool:
    @pytest.mark.asyncio
    async def test_run_report_uses_auth_profile_and_writes_csv(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        module = _load_ga_live_module()
        tool = module.GALiveApiTool()

        _FakeAsyncClient.calls = []
        _FakeAsyncClient.payload = {
            "dimensionHeaders": [{"name": "date"}],
            "metricHeaders": [{"name": "sessions"}, {"name": "conversions"}],
            "rows": [
                {
                    "dimensionValues": [{"value": "2026-02-20"}],
                    "metricValues": [{"value": "101"}, {"value": "7"}],
                }
            ],
            "rowCount": 1,
        }
        monkeypatch.setattr(module.httpx, "AsyncClient", _FakeAsyncClient)

        auth_context = _build_auth_context(tmp_path, monkeypatch)
        ctx = ToolContext(workspace=tmp_path, auth_context=auth_context)

        result = await tool.execute(
            {
                "operation": "run_report",
                "metrics": ["sessions", "conversions"],
                "dimensions": ["date"],
                "start_date": "2026-02-01",
                "end_date": "2026-02-20",
                "output_path": "ga-report.csv",
            },
            ctx,
        )

        assert result.success
        assert result.data is not None
        assert result.data["property_id"] == "123456789"
        assert result.data["row_count"] == 1
        assert result.files_changed == ["ga-report.csv"]

        csv_out = (tmp_path / "ga-report.csv").read_text(encoding="utf-8")
        assert "date,sessions,conversions" in csv_out
        assert "2026-02-20,101,7" in csv_out

        assert _FakeAsyncClient.calls
        call = _FakeAsyncClient.calls[0]
        assert call["method"] == "POST"
        assert call["url"].endswith("/properties/123456789:runReport")
        assert call["headers"]["Authorization"] == "Bearer ya29.mock-token"
        assert call["params"] == {}

    @pytest.mark.asyncio
    async def test_missing_auth_context_returns_error(self, tmp_path: Path):
        module = _load_ga_live_module()
        tool = module.GALiveApiTool()

        result = await tool.execute(
            {
                "operation": "run_report",
                "property_id": "123456789",
                "metrics": ["sessions"],
            },
            ToolContext(workspace=tmp_path),
        )

        assert not result.success
        assert result.error is not None
        assert "No credential resolved" in result.error

    def test_declares_google_analytics_auth_requirement(self):
        module = _load_ga_live_module()
        tool = module.GALiveApiTool()

        requirements = tool.auth_requirements
        assert requirements
        assert requirements[0].get("provider") == "google_analytics"
        assert requirements[0].get("source") == "api"
