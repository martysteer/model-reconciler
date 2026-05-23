"""Model Reconciler — W3C Reconciliation API."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model_reconciler.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    logging.basicConfig(level=settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: profile loading will go here in Task 3
        yield
        # Shutdown: nothing to clean up

    application = FastAPI(
        title="Model Reconciler",
        description="Profile-driven W3C Reconciliation API",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.state.settings = settings

    @application.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return application


app = create_app()
