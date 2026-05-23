"""Model Reconciler — W3C Reconciliation API with profile-based routing."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from cachetools import TTLCache
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model_reconciler.config import Settings
from model_reconciler.models import ProfileConfig, ServiceManifest
from model_reconciler.profiles import load_all_profiles


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    logging.basicConfig(level=settings.log_level)
    logger = logging.getLogger(__name__)

    # Registry populated during lifespan startup
    registry: dict[str, tuple[ProfileConfig, TTLCache]] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        profiles_dir = Path(settings.profiles_dir)
        if profiles_dir.exists():
            for profile in load_all_profiles(profiles_dir):
                cache = TTLCache(maxsize=1000, ttl=profile.cache_ttl)
                registry[profile.slug] = (profile, cache)
                logger.info(f"Mounted /reconcile/{profile.slug} -> {profile.name}")
        else:
            logger.warning(f"Profiles directory not found: {profiles_dir}")
        yield
        registry.clear()

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
    application.state.registry = registry

    def _get_profile(slug: str) -> tuple[ProfileConfig, TTLCache]:
        if slug not in registry:
            raise HTTPException(404, detail=f"Profile not found: {slug}")
        return registry[slug]

    @application.get("/")
    async def list_services():
        """List all loaded reconciliation services."""
        return [
            {
                "slug": slug,
                "name": profile.name,
                "description": profile.description or profile.name,
                "url": f"/reconcile/{slug}",
            }
            for slug, (profile, _) in registry.items()
        ]

    @application.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "profiles_loaded": len(registry),
            "profiles": list(registry.keys()),
        }

    @application.get("/reconcile/{slug}")
    async def get_manifest(slug: str):
        """W3C Reconciliation Service manifest."""
        profile, _ = _get_profile(slug)
        return ServiceManifest(
            name=profile.name,
            identifierSpace=f"/entity/{slug}/",
            schemaSpace=f"/schema/{slug}/",
            defaultTypes=profile.types,
        ).model_dump()

    return application


app = create_app()
