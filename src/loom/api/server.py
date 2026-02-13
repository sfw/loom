"""FastAPI application factory for Loom API server."""

from __future__ import annotations

from fastapi import FastAPI

from loom.config import Config


def create_app(config: Config | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Loom",
        description="Local model orchestration engine",
        version="0.1.0",
    )

    if config is not None:
        app.state.config = config

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": "0.1.0"}

    return app
