# Model Reconciler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build model-reconciler — a containerized W3C Reconciliation API that translates OpenRefine queries into LLM prompts via any OpenAI-compatible inference engine.

**Architecture:** FastAPI in Docker, inference engine native on host. YAML profiles configure prompts + entity types → each mounts as `/reconcile/{slug}`. Container reaches host engine via `host.docker.internal`.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, httpx, pyyaml, pydantic, pydantic-settings, cachetools, Docker, pytest, ruff

**Spec:** `docs2/superpowers/specs/2026-05-23-model-reconciler-design-v2.md`

---

## Approach

Build in **vertical slices** — each task delivers a working, testable increment:

1. Scaffold + running health endpoint (prove Docker works)
2. Profile system + validation (prove YAML loading works, with tests)
3. W3C service discovery + manifest (prove OpenRefine can connect)
4. LLM client + reconciliation logic (prove queries produce results)
5. Wire reconciliation into API with caching (prove end-to-end flow)
6. README

---

## File Map

```
model-reconciler/
├── src/model_reconciler/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── profiles.py
│   ├── llm.py
│   └── reconcile.py
│   └── main.py
├── profiles/
│   ├── library.yaml
│   └── general.yaml
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   └── valid.yaml
│   ├── test_profiles.py
│   ├── test_health.py
│   └── test_manifest.py
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

### Task 1: Scaffold + Running Health Endpoint

**Goal:** `docker compose up` starts a container, `GET /health` returns 200. Proves the entire toolchain works before writing any business logic.

**Files:**
- Create: all directories, `.gitignore`, `requirements.txt`, `requirements-dev.txt`, `Dockerfile`, `docker-compose.yml`, `Makefile`, `src/model_reconciler/__init__.py`, `src/model_reconciler/config.py`, `src/model_reconciler/main.py`

- [ ] **Step 1: Create directory structure and initialise git**

```bash
mkdir -p model-reconciler/src/model_reconciler
mkdir -p model-reconciler/profiles
mkdir -p model-reconciler/tests/fixtures
cd model-reconciler
git init
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
dist/
build/
.coverage
htmlcov/
.env
```

- [ ] **Step 3: Create `requirements.txt`**

```
fastapi
uvicorn[standard]
httpx
pyyaml
pydantic
pydantic-settings
cachetools
```

- [ ] **Step 4: Create `requirements-dev.txt`**

```
-r requirements.txt
pytest
pytest-asyncio
ruff
```

Note: `-r requirements.txt` pulls in prod deps so dev install is a single command.

- [ ] **Step 5: Create `src/model_reconciler/__init__.py`**

```python
```

(Empty file.)

- [ ] **Step 6: Create `src/model_reconciler/config.py`**

```python
"""Global configuration — three environment variables, nothing else."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_base_url: str = "http://host.docker.internal:8080/v1"
    profiles_dir: str = "profiles"
    log_level: str = "INFO"

    model_config = {"env_prefix": "", "case_sensitive": False}
```

- [ ] **Step 7: Create `src/model_reconciler/main.py` — minimal, health only**

```python
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
```

- [ ] **Step 8: Create `Dockerfile`**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY src/ ./src/
COPY profiles/ ./profiles/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8001

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "model_reconciler.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

- [ ] **Step 9: Create `docker-compose.yml`**

```yaml
services:
  api:
    build: .
    container_name: model-reconciler
    ports:
      - "8001:8001"
    volumes:
      - ./profiles:/app/profiles:ro
      - ./src:/app/src
    environment:
      - LLM_BASE_URL=http://host.docker.internal:8080/v1
      - PROFILES_DIR=/app/profiles
      - LOG_LEVEL=INFO
    command: ["uvicorn", "model_reconciler.main:app",
              "--host", "0.0.0.0", "--port", "8001", "--reload"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped
```

- [ ] **Step 10: Create `Makefile`**

```makefile
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := up

.PHONY: up stop test lint logs clean

up:
	@echo "Model Reconciler — http://127.0.0.1:8001/"
	@docker compose up --build

stop:
	@docker compose down

test:
	@docker compose run --rm --no-deps \
		-v "./tests:/app/tests:ro" \
		api pytest tests/ -v

lint:
	@docker compose run --rm --no-deps \
		-v "./tests:/app/tests:ro" \
		api ruff check src/ tests/

logs:
	@docker compose logs -f

clean:
	@docker compose down -v --remove-orphans 2>/dev/null || true
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .coverage htmlcov
```

- [ ] **Step 11: Build and verify health endpoint**

Run: `docker compose build && docker compose run --rm api curl -sf http://localhost:8001/health`

If that's awkward in one shot, verify with:

```bash
docker compose up -d --build
sleep 3
curl -sf http://localhost:8001/health
docker compose down
```

Expected: `{"status":"healthy"}`

- [ ] **Step 12: Commit**

```bash
git add .
git commit -m "feat: scaffold with running health endpoint"
```

---

### Task 2: Profile System

**Goal:** YAML profiles load, validate, and reject bad input. Smoke tests prove it.

**Files:**
- Create: `src/model_reconciler/models.py`, `src/model_reconciler/profiles.py`
- Create: `profiles/library.yaml`, `profiles/general.yaml`
- Create: `tests/fixtures/valid.yaml`, `tests/conftest.py`, `tests/test_profiles.py`

- [ ] **Step 1: Create `src/model_reconciler/models.py`**

```python
"""Pydantic models for profiles, W3C types, and reconciliation results."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ProfileConfig(BaseModel):
    """A reconciliation profile loaded from YAML."""

    name: str
    prompt: str
    types: list[dict[str, str]]
    slug: Optional[str] = None
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_tokens: int = Field(default=800, ge=1)
    cache_ttl: int = Field(default=3600, ge=0)
    description: Optional[str] = None
    use_dspy: bool = False


class ReconciliationQuery(BaseModel):
    """A single reconciliation query from OpenRefine."""

    query: str
    type: Optional[str] = None
    limit: int = Field(default=5, ge=1, le=25)
    properties: list[dict[str, Any]] = Field(default_factory=list)


class ReconciliationCandidate(BaseModel):
    """A single match result returned to OpenRefine."""

    id: str
    name: str
    score: float = Field(ge=0, le=100)
    match: bool = False
    type: list[dict[str, str]] = Field(default_factory=list)
    description: Optional[str] = None


class ServiceManifest(BaseModel):
    """W3C Reconciliation Service Manifest (v0.2)."""

    versions: list[str] = ["0.2"]
    name: str
    identifierSpace: str
    schemaSpace: str
    defaultTypes: list[dict[str, str]] = Field(default_factory=list)
    view: Optional[dict[str, str]] = None
    preview: Optional[dict[str, Any]] = None
    suggest: Optional[dict[str, Any]] = None
    extend: Optional[dict[str, Any]] = None
```

- [ ] **Step 2: Create `src/model_reconciler/profiles.py`**

```python
"""Load YAML profile files into validated ProfileConfig objects."""

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from model_reconciler.models import ProfileConfig

logger = logging.getLogger(__name__)


def load_profile(path: Path) -> ProfileConfig:
    """Load one YAML file, derive slug from filename if absent."""
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Profile must be a YAML mapping: {path}")

    if not data.get("slug"):
        data["slug"] = path.stem

    try:
        return ProfileConfig(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid profile {path}: {e}") from e


def load_all_profiles(profiles_dir: Path) -> list[ProfileConfig]:
    """Load all *.yaml files from a directory. Rejects duplicate slugs."""
    profiles = []
    for path in sorted(profiles_dir.glob("*.yaml")):
        profile = load_profile(path)
        profiles.append(profile)
        logger.info(f"Loaded profile: {profile.slug} ({profile.name})")

    slugs = [p.slug for p in profiles]
    dupes = [s for s in slugs if slugs.count(s) > 1]
    if dupes:
        raise ValueError(f"Duplicate profile slugs: {set(dupes)}")

    return profiles
```

- [ ] **Step 3: Create `profiles/library.yaml`**

```yaml
name: "SOAS Library Reconciliation"

prompt: |
  You are an expert at matching search queries to controlled vocabulary terms
  and authority records. Find the best matching entities for the given query.

  Guidelines:
  1. Consider exact matches and semantic equivalents
  2. Account for synonyms, alternate forms, and related concepts
  3. Prefer specific matches over general ones
  4. Consider the entity type filter if provided
  5. Score matches 0-100:
     - 95-100: Exact or near-exact match
     - 80-94: Strong semantic match
     - 60-79: Related concept, good candidate
     - 40-59: Loosely related
     - Below 40: Weak match

  Return a JSON array:
  [{"id": "unique_id", "name": "Preferred label", "score": 85, "description": "Why this matches"}]

types:
  - id: "personal"
    name: "Personal Name"
  - id: "topical"
    name: "Topical Subject"
  - id: "corporate"
    name: "Corporate Name"
  - id: "geographic"
    name: "Geographic Name"
```

- [ ] **Step 4: Create `profiles/general.yaml`**

```yaml
name: "General Entity Reconciliation"

prompt: |
  Match the query to known entities. Return a JSON array of matches:
  [{"id": "unique_id", "name": "Entity label", "score": 85, "description": "Why this matches"}]

  Score: 95-100 exact, 80-94 strong, 60-79 related, below 60 weak.

types:
  - id: "entity"
    name: "Entity"
```

- [ ] **Step 5: Create `tests/fixtures/valid.yaml`**

```yaml
name: "Test Profile"
prompt: "Match queries. Return JSON array of matches."
types:
  - id: "test"
    name: "Test Type"
```

- [ ] **Step 6: Create `tests/conftest.py`**

```python
"""Shared test fixtures."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    """Path to test fixture YAMLs."""
    return FIXTURES_DIR


@pytest.fixture
def app_client():
    """FastAPI TestClient with profiles from tests/fixtures/."""
    os.environ["PROFILES_DIR"] = str(FIXTURES_DIR)
    os.environ["LLM_BASE_URL"] = "http://fake:8080/v1"

    from model_reconciler.main import create_app

    app = create_app()
    with TestClient(app) as client:
        yield client
```

- [ ] **Step 7: Create `tests/test_profiles.py`**

```python
"""Smoke tests: profile loading and validation."""

import pytest

from model_reconciler.profiles import load_all_profiles, load_profile


def test_valid_profile_loads(fixtures_dir):
    """Valid YAML loads with slug derived from filename."""
    p = load_profile(fixtures_dir / "valid.yaml")
    assert p.name == "Test Profile"
    assert p.slug == "valid"
    assert len(p.types) == 1
    assert p.temperature == 0.1
    assert p.cache_ttl == 3600
    assert p.use_dspy is False


def test_missing_file_raises(tmp_path):
    """FileNotFoundError for nonexistent YAML."""
    with pytest.raises(FileNotFoundError):
        load_profile(tmp_path / "ghost.yaml")


def test_missing_required_field_raises(tmp_path):
    """ValueError when required field (types) is absent."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: Broken\nprompt: No types here\n")
    with pytest.raises(ValueError, match="Invalid profile"):
        load_profile(bad)


def test_bad_yaml_raises(tmp_path):
    """ValueError when file isn't a YAML mapping."""
    bad = tmp_path / "list.yaml"
    bad.write_text("- item1\n- item2\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_profile(bad)


def test_load_all_profiles(fixtures_dir):
    """load_all_profiles returns validated list."""
    profiles = load_all_profiles(fixtures_dir)
    assert len(profiles) >= 1
    assert all(p.slug for p in profiles)


def test_duplicate_slugs_rejected(tmp_path):
    """ValueError when two profiles share a slug."""
    for name in ("a.yaml", "b.yaml"):
        (tmp_path / name).write_text(
            "name: Dupe\nprompt: Test\nslug: same\ntypes:\n  - id: x\n    name: X\n"
        )
    with pytest.raises(ValueError, match="Duplicate"):
        load_all_profiles(tmp_path)
```

- [ ] **Step 8: Run profile tests**

Run: `PYTHONPATH=src pytest tests/test_profiles.py -v`
Expected: 6 tests PASS

- [ ] **Step 9: Commit**

```bash
git add src/model_reconciler/models.py src/model_reconciler/profiles.py
git add profiles/ tests/
git commit -m "feat: profile system with YAML loading, validation, and tests"
```

---

### Task 3: Service Discovery + W3C Manifest

**Goal:** Profiles load on startup, `GET /` lists them, `GET /reconcile/{slug}` returns a W3C manifest. OpenRefine can now connect and see the service.

**Files:**
- Modify: `src/model_reconciler/main.py`
- Create: `tests/test_health.py`, `tests/test_manifest.py`

- [ ] **Step 1: Update `src/model_reconciler/main.py` — add profile registry + manifest routes**

Replace the entire file with:

```python
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
```

- [ ] **Step 2: Create `tests/test_health.py`**

```python
"""Smoke tests: health and service listing."""


def test_health_returns_200(app_client):
    r = app_client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["profiles_loaded"] >= 1
    assert isinstance(data["profiles"], list)


def test_root_lists_services(app_client):
    r = app_client.get("/")
    assert r.status_code == 200
    services = r.json()
    assert len(services) >= 1
    for s in services:
        assert "slug" in s
        assert "name" in s
        assert "url" in s
        assert s["url"].startswith("/reconcile/")
```

- [ ] **Step 3: Create `tests/test_manifest.py`**

```python
"""Smoke tests: W3C manifest endpoint."""


def test_manifest_shape(app_client):
    """GET /reconcile/{slug} returns valid W3C manifest."""
    services = app_client.get("/").json()
    slug = services[0]["slug"]

    r = app_client.get(f"/reconcile/{slug}")
    assert r.status_code == 200

    m = r.json()
    assert "0.2" in m["versions"]
    assert m["name"]
    assert m["identifierSpace"]
    assert m["schemaSpace"]
    assert isinstance(m["defaultTypes"], list)
    assert len(m["defaultTypes"]) >= 1
    assert "id" in m["defaultTypes"][0]
    assert "name" in m["defaultTypes"][0]


def test_manifest_404_for_unknown_slug(app_client):
    r = app_client.get("/reconcile/nonexistent")
    assert r.status_code == 404
```

- [ ] **Step 4: Run all tests**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: 10 tests PASS (6 profile + 2 health + 2 manifest)

- [ ] **Step 5: Commit**

```bash
git add src/model_reconciler/main.py tests/test_health.py tests/test_manifest.py
git commit -m "feat: service discovery and W3C manifest endpoint"
```

---

### Task 4: LLM Client + Reconciliation Logic

**Goal:** `llm.py` can call any OpenAI-compatible endpoint. `reconcile.py` builds prompts, calls the LLM, and parses JSON responses into typed candidates.

**Files:**
- Create: `src/model_reconciler/llm.py`, `src/model_reconciler/reconcile.py`

- [ ] **Step 1: Create `src/model_reconciler/llm.py`**

```python
"""Async HTTP client for OpenAI-compatible chat completions."""

import logging

import httpx

logger = logging.getLogger(__name__)


async def chat_completion(
    base_url: str,
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 800,
) -> str:
    """POST to /chat/completions with JSON mode. Return content string.

    Args:
        base_url: e.g. http://localhost:8080/v1
        messages: [{"role": "system", "content": "..."}, ...]
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate.

    Returns:
        Raw content string from the model's response.

    Raises:
        httpx.HTTPStatusError: On non-2xx response.
        httpx.ConnectError: If the inference engine is unreachable.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            url,
            json={
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            },
        )
        r.raise_for_status()

    return r.json()["choices"][0]["message"]["content"]
```

- [ ] **Step 2: Create `src/model_reconciler/reconcile.py`**

```python
"""Reconciliation logic — stateless functions, no framework coupling."""

import json
import logging

from model_reconciler.llm import chat_completion
from model_reconciler.models import (
    ProfileConfig,
    ReconciliationCandidate,
    ReconciliationQuery,
)

logger = logging.getLogger(__name__)


async def reconcile_query(
    query: ReconciliationQuery,
    profile: ProfileConfig,
    base_url: str,
) -> list[ReconciliationCandidate]:
    """Build prompt, call LLM, parse JSON, return candidates."""
    if profile.use_dspy:
        raise NotImplementedError(
            f"Profile '{profile.slug}' has use_dspy=true but DSPy "
            "support is not yet implemented. Set use_dspy: false."
        )

    messages = [
        {"role": "system", "content": profile.prompt},
        {"role": "user", "content": _format_user_message(query)},
    ]

    try:
        raw = await chat_completion(
            base_url=base_url,
            messages=messages,
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
        )
    except Exception:
        logger.exception(f"LLM call failed for profile '{profile.slug}'")
        return []

    return parse_llm_response(raw)[: query.limit]


def _format_user_message(query: ReconciliationQuery) -> str:
    """Format the user message sent to the LLM."""
    parts = [f"Query: {query.query}"]
    if query.type:
        parts.append(f"Type: {query.type}")
    return "\n".join(parts)


def parse_llm_response(text: str) -> list[ReconciliationCandidate]:
    """Parse LLM JSON output into candidate list.

    Handles three shapes:
      1. Direct array: [{"id": ..., "name": ..., "score": ...}, ...]
      2. Wrapped object: {"matches": [...]} / {"results": [...]} / {"entities": [...]}
      3. Markdown fences: ```json [...] ```

    Returns empty list on any parse failure.
    """
    try:
        cleaned = _strip_markdown_fences(text.strip())
        data = json.loads(cleaned)
        items = _extract_array(data)
        if items is None:
            return []
        return [_to_candidate(m, i) for i, m in enumerate(items)]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM response: {e}")
        return []


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrapping if present."""
    if "```" not in text:
        return text
    inner = text.split("```")[1]
    if inner.startswith("json"):
        inner = inner[4:]
    return inner.strip()


def _extract_array(data) -> list[dict] | None:
    """Pull the match array from a direct list or wrapped object."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("matches", "results", "entities"):
            if key in data and isinstance(data[key], list):
                return data[key]
        logger.warning(f"Unexpected JSON object keys: {list(data.keys())}")
    return None


def _to_candidate(m: dict, index: int) -> ReconciliationCandidate:
    """Map a raw dict from the LLM into a typed ReconciliationCandidate."""
    score = float(m.get("score", 50))
    return ReconciliationCandidate(
        id=m.get("id", f"gen_{index}"),
        name=m.get("name", ""),
        score=score,
        match=score >= 90,
        description=m.get("description") or m.get("reasoning", ""),
    )
```

- [ ] **Step 3: Verify imports**

Run: `PYTHONPATH=src python -c "from model_reconciler.reconcile import reconcile_query, parse_llm_response; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/model_reconciler/llm.py src/model_reconciler/reconcile.py
git commit -m "feat: LLM client and reconciliation logic"
```

---

### Task 5: Reconciliation Endpoint with Caching

**Goal:** `POST /reconcile/{slug}` handles OpenRefine batch queries. Uncached queries run concurrently. Cached queries return immediately.

**Files:**
- Modify: `src/model_reconciler/main.py`

- [ ] **Step 1: Add POST endpoint and batch processing to `src/model_reconciler/main.py`**

Add these imports at the top of the file:

```python
import asyncio
import json
from typing import Any, Optional

from fastapi import Form
from fastapi.responses import JSONResponse
```

Add the full import list (replace existing imports):

```python
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
```

Add this route inside `create_app`, after the `get_manifest` route:

```python
    @application.post("/reconcile/{slug}")
    async def reconcile(
        slug: str,
        queries: Optional[str] = Form(default=None),
        query: Optional[str] = Form(default=None),
    ):
        """W3C batch reconciliation endpoint."""
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
        """Process batch queries. Cached hits return immediately; misses run concurrently."""
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
        """Process a single query string."""
        q = ReconciliationQuery(query=query_text)
        cache_key = f"{q.query}:{q.type}:{q.limit}"

        if cache_key in cache:
            return [c.model_dump() for c in cache[cache_key]]

        candidates = await reconcile_query(q, profile, base_url)
        cache[cache_key] = candidates
        return [c.model_dump() for c in candidates]
```

- [ ] **Step 2: Verify the full main.py is correct**

Write the complete final `main.py` (to avoid partial-edit confusion):

```python
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
```

- [ ] **Step 3: Run all existing tests (nothing should break)**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: 10 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/model_reconciler/main.py
git commit -m "feat: batch reconciliation endpoint with caching"
```

---

### Task 6: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create `README.md`**

```markdown
# Model Reconciler

Profile-driven [W3C Reconciliation API](https://www.w3.org/community/reports/reconciliation/CG-FINAL-specs-0.2-20230410/) powered by local LLMs. Model-agnostic — works with any OpenAI-compatible inference engine.

## How It Works

1. You run an inference engine on your machine (llama-server, Ollama, omlx, vLLM, LM Studio — anything with an OpenAI-compatible API)
2. Model Reconciler runs in a Docker container, translating OpenRefine's reconciliation queries into LLM prompts
3. Drop a YAML profile into `profiles/` to create a new reconciliation service — no code changes

```
profiles/library.yaml  ->  /reconcile/library
profiles/general.yaml  ->  /reconcile/general
```

## Quick Start

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) and an OpenAI-compatible inference engine.

1. Start your inference engine:

   ```bash
   # llama-server
   llama-server -hf ggml-org/gemma-3-4b-it-GGUF:Q4_K_M --port 8080

   # Ollama (uses port 11434 — set LLM_BASE_URL in docker-compose.yml)
   ollama serve

   # omlx
   omlx serve --port 8080
   ```

2. Start the API:

   ```bash
   make
   ```

3. Connect OpenRefine:
   - Column dropdown > Reconcile > Start reconciling...
   - Add Standard Service: `http://localhost:8001/reconcile/library`

## Profiles

A YAML file with three required fields creates a reconciliation service:

```yaml
name: "SOAS Library Reconciliation"
prompt: |
  Match the query to known bibliographic entities.
  Return a JSON array: [{"id": "...", "name": "...", "score": 85, "description": "..."}]
types:
  - id: "topical"
    name: "Topical Subject"
```

Optional fields:

| Field | Default | Description |
|-------|---------|-------------|
| `slug` | filename stem | URL path segment |
| `temperature` | `0.1` | LLM sampling temperature |
| `max_tokens` | `800` | Max generation tokens |
| `cache_ttl` | `3600` | Cache TTL in seconds (0 to disable) |
| `description` | profile name | Shown in W3C manifest |
| `use_dspy` | `false` | Opt-in DSPy structured output |

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_BASE_URL` | `http://host.docker.internal:8080/v1` | Inference engine endpoint |
| `PROFILES_DIR` | `profiles` | Profile YAML directory |
| `LOG_LEVEL` | `INFO` | Logging level |

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | List available services |
| `/health` | GET | Health check |
| `/reconcile/{slug}` | GET | W3C manifest |
| `/reconcile/{slug}` | POST | Reconcile queries |

## Commands

| Command | Description |
|---------|-------------|
| `make` | Build and start |
| `make stop` | Stop |
| `make test` | Run tests |
| `make lint` | Run linter |
| `make logs` | Follow logs |
| `make clean` | Remove everything |

## Architecture

```
Host (native, GPU-accelerated):
  ┌──────────────────────────────┐
  │  Any OpenAI-compatible       │
  │  inference engine            │
  │  http://localhost:8080/v1    │
  └──────────────┬───────────────┘
                 │
  Container (Docker):
  ┌──────────────┴───────────────┐
  │   model-reconciler (FastAPI) │
  │   http://localhost:8001      │
  │                              │
  │   profiles/*.yaml -> routes  │
  │   /reconcile/{slug}          │
  └──────────────────────────────┘
```

The inference engine runs natively for GPU access (Metal on Mac has no container passthrough). The API container is model-agnostic — swap engines by changing `LLM_BASE_URL`.
```

- [ ] **Step 2: Lint everything**

Run: `PYTHONPATH=src ruff check src/ tests/`
Expected: No errors

- [ ] **Step 3: Run full test suite**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: 10 tests PASS

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add README"
```

---

## Verification Checklist

After all tasks:

- [ ] `PYTHONPATH=src pytest tests/ -v` — 10 tests pass
- [ ] `PYTHONPATH=src ruff check src/ tests/` — clean
- [ ] `docker compose build` — image builds
- [ ] `docker compose up -d && sleep 3 && curl -sf http://localhost:8001/health && docker compose down` — health check passes
- [ ] `curl -sf http://localhost:8001/reconcile/library` — returns W3C manifest (with profiles mounted)
- [ ] OpenRefine can add `http://localhost:8001/reconcile/library` as a standard service
