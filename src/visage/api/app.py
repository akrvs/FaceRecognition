from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from visage.api.routes import recognizer_provider, router
from visage.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if not recognizer_provider.is_set:
            from visage.api.dependencies import build_default_recognizer

            recognizer_provider.set(build_default_recognizer(settings))
        yield

    app = FastAPI(title=settings.api_title, version="0.1.0", lifespan=lifespan)
    app.include_router(router)
    return app
