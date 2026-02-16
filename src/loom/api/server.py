"""FastAPI application factory for Loom API server."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from loom.config import Config


def create_app(config: Config | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Two modes:
    - With config: full engine initialization via lifespan (production)
    - Without config or with config but no models: lightweight app (testing)
    """
    resolved_config = config or Config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import logging

        from loom.api.engine import create_engine

        logger = logging.getLogger("loom.server")
        try:
            engine = await create_engine(resolved_config)
        except Exception as e:
            logger.error("Failed to initialize engine: %s", e)
            raise
        app.state.engine = engine
        yield
        await engine.shutdown()

    app = FastAPI(
        title="Loom",
        description="Local model orchestration engine",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.config = resolved_config

    # CORS — local only for V1
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    from loom.api.routes import router

    app.include_router(router)

    return app
