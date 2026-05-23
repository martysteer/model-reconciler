"""Model Reconciler — W3C Reconciliation API with profile-based routing."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from cachetools import TTLCache
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model_reconciler.config import Settings
from model_reconciler.models import (
    ProfileConfig,
    ReconciliationQuery,
    ServiceManifest,
)
from model_reconciler.profiles import load_all_profiles
from model_reconciler.reconcile import reconcile_query


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    logging.basicConfig(level=settings.log_level)
    logger = logging.getLogger(__name__)

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
        profile, _ = _get_profile(slug)
        return ServiceManifest(
            name=profile.name,
            identifierSpace=f"/entity/{slug}/",
            schemaSpace=f"/schema/{slug}/",
            defaultTypes=profile.types,
        ).model_dump()

    @application.post("/reconcile/{slug}")
    async def reconcile(
        slug: str,
        queries: Optional[str] = Form(default=None),
        query: Optional[str] = Form(default=None),
    ):
        profile, cache = _get_profile(slug)
        base_url = settings.llm_base_url

        if queries:
            try:
                batch = json.loads(queries)
            except json.JSONDecodeError as e:
                raise HTTPException(400, detail=f"Invalid JSON in queries: {e}")
            return JSONResponse(
                content=await _run_batch(profile, cache, batch, base_url)
            )

        if query:
            candidates = await _run_single(profile, cache, query, base_url)
            return JSONResponse(content={"result": candidates})

        return await get_manifest(slug)

    async def _run_batch(
        profile: ProfileConfig,
        cache: TTLCache,
        batch: dict[str, Any],
        base_url: str,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        uncached: dict[str, tuple[ReconciliationQuery, str]] = {}

        for qid, qdata in batch.items():
            q = ReconciliationQuery(
                query=qdata.get("query", ""),
                type=qdata.get("type"),
                limit=qdata.get("limit", 5),
                properties=qdata.get("properties", []),
            )
            cache_key = f"{q.query}:{q.type}:{q.limit}"

            if cache_key in cache:
                results[qid] = {"result": [c.model_dump() for c in cache[cache_key]]}
            else:
                uncached[qid] = (q, cache_key)

        if uncached:
            coros = [
                reconcile_query(q, profile, base_url)
                for q, _ in uncached.values()
            ]
            completed = await asyncio.gather(*coros)

            for (qid, (_, cache_key)), candidates in zip(
                uncached.items(), completed
            ):
                cache[cache_key] = candidates
                results[qid] = {"result": [c.model_dump() for c in candidates]}

        return results

    async def _run_single(
        profile: ProfileConfig,
        cache: TTLCache,
        query_text: str,
        base_url: str,
    ) -> list[dict[str, Any]]:
        q = ReconciliationQuery(query=query_text)
        cache_key = f"{q.query}:{q.type}:{q.limit}"

        if cache_key in cache:
            return [c.model_dump() for c in cache[cache_key]]

        candidates = await reconcile_query(q, profile, base_url)
        cache[cache_key] = candidates
        return [c.model_dump() for c in candidates]

    return application


app = create_app()
