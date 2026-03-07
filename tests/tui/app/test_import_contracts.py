from __future__ import annotations

import loom.tui.app as app
from loom.tui.app import (
    LoomApp,
    ProcessRunLaunchRequest,
    ProcessRunList,
    ProcessRunPane,
    ProcessRunState,
    SteeringDirective,
)


def test_import_contract_exports() -> None:
    assert LoomApp.__name__ == "LoomApp"
    assert ProcessRunList.__name__ == "ProcessRunList"
    assert ProcessRunPane.__name__ == "ProcessRunPane"
    assert ProcessRunState.__name__ == "ProcessRunState"
    assert ProcessRunLaunchRequest.__name__ == "ProcessRunLaunchRequest"
    assert SteeringDirective.__name__ == "SteeringDirective"


def test_facade_attribute_patch_propagates_to_core(monkeypatch) -> None:
    import loom.tui.app.core as core

    class DummySession:
        pass

    monkeypatch.setattr(app, "CoworkSession", DummySession)
    assert core.CoworkSession is DummySession
