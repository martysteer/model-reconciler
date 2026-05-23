# Model Reconciler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a model-agnostic, profile-driven W3C Reconciliation API as a containerized FastAPI app that talks to any external OpenAI-compatible inference engine.

**Architecture:** FastAPI container handles W3C reconciliation logic. Inference engine runs natively on host (user's choice — llama-server, Ollama, omlx, etc.). Container reaches host via `host.docker.internal`. Profiles are YAML files that configure prompts and entity types per endpoint.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, httpx, pyyaml, pydantic, pydantic-settings, cachetools, Docker, pytest, ruff

**Spec:** `docs2/superpowers/specs/2026-05-23-model-reconciler-design.md`

---

## File Map

```
model-reconciler/
├── src/model_reconciler/
│   ├── __init__.py          # Empty package marker
│   ├── config.py            # Settings: 3 env vars (LLM_BASE_URL, PROFILES_DIR, LOG_LEVEL)
│   ├── models.py            # Pydantic: ProfileConfig, ReconciliationQuery, ReconciliationCandidate, ServiceManifest
│   ├── profiles.py          # load_profile(), load_all_profiles() — YAML → validated ProfileConfig
│   ├── llm.py               # chat_completion() — async httpx POST to OpenAI-compat endpoint
│   ├── reconcile.py         # reconcile_query(), parse_llm_response() — prompt + LLM + parse → candidates
│   └── main.py              # create_app() factory, route mounting, caching, request/response handling
├── profiles/
│   ├── library.yaml         # SOAS library reconciliation profile
│   └── general.yaml         # General entity matching profile
├── tests/
│   ├── conftest.py          # Shared fixtures: test app client, test profiles dir
│   ├── fixtures/
│   │   └── valid.yaml       # Minimal valid profile for tests
│   ├── test_profiles.py     # Smoke: YAML loading, validation, slug derivation
│   ├── test_health.py       # Smoke: /health returns 200, correct profile count
│   └── test_manifest.py     # Smoke: /reconcile/{slug} returns valid W3C manifest
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

### Task 1: Project Scaffold

**Files:**
- Create: all directories, `requirements.txt`, `requirements-dev.txt`, `Dockerfile`, `docker-compose.yml`, `Makefile`, `src/model_reconciler/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p model-reconciler/src/model_reconciler
mkdir -p model-reconciler/profiles
mkdir -p model-reconciler/tests/fixtures
cd model-reconciler
git init
```

- [ ] **Step 2: Create `requirements.txt`**

```
fastapi
uvicorn[standard]
httpx
pyyaml
pydantic
pydantic-settings
cachetools
```

- [ ] **Step 3: Create `requirements-dev.txt`**

```
pytest
pytest-asyncio
httpx
ruff
```

- [ ] **Step 4: Create `Dockerfile`**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

COPY src/ ./src/
COPY profiles/ ./profiles/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8001

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "model_reconciler.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

- [ ] **Step 5: Create `docker-compose.yml`**

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

networks:
  default:
    name: model-reconciler-network
```

- [ ] **Step 6: Create `Makefile`**

```makefile
# Model Reconciler — Makefile

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := up

DOCKER_COMPOSE := docker compose

.PHONY: up
up:
	@echo "Model Reconciler — http://127.0.0.1:8001/"
	@$(DOCKER_COMPOSE) up --build

.PHONY: stop
stop:
	@$(DOCKER_COMPOSE) down

.PHONY: test
test:
	@$(DOCKER_COMPOSE) run --rm --no-deps -v "./tests:/app/tests:ro" api pytest tests/ -v

.PHONY: lint
lint:
	@$(DOCKER_COMPOSE) run --rm --no-deps -v "./tests:/app/tests:ro" api ruff check src/ tests/

.PHONY: logs
logs:
	@$(DOCKER_COMPOSE) logs -f

.PHONY: clean
clean:
	@$(DOCKER_COMPOSE) down -v --remove-orphans 2>/dev/null || true
	@rm -rf __pycache__ src/**/__pycache__ tests/**/__pycache__
	@rm -rf .pytest_cache .coverage htmlcov
```

- [ ] **Step 7: Create `src/model_reconciler/__init__.py`**

```python
```

(Empty file.)

- [ ] **Step 8: Commit**

```bash
git add .
git commit -m "chore: project scaffold — dirs, deps, Docker, Makefile"
```

---

### Task 2: Data Layer — config.py + models.py

**Files:**
- Create: `src/model_reconciler/config.py`
- Create: `src/model_reconciler/models.py`

- [ ] **Step 1: Create `src/model_reconciler/config.py`**

```python
"""Global configuration — 3 env vars, nothing else."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global settings loaded from environment variables."""

    llm_base_url: str = "http://host.docker.internal:8080/v1"
    profiles_dir: str = "profiles"
    log_level: str = "INFO"

    model_config = {"env_prefix": "", "case_sensitive": False}
```

- [ ] **Step 2: Create `src/model_reconciler/models.py`**

```python
"""Pydantic models for W3C Reconciliation API and profile configuration."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ReconciliationQuery(BaseModel):
    """A single reconciliation query from OpenRefine."""

    query: str = Field(..., description="The search term to reconcile")
    type: Optional[str] = Field(default=None, description="Optional type filter")
    limit: int = Field(default=5, ge=1, le=25, description="Maximum results")
    properties: list[dict[str, Any]] = Field(default_factory=list)


class ReconciliationCandidate(BaseModel):
    """A candidate match result."""

    id: str = Field(..., description="Entity identifier")
    name: str = Field(..., description="Entity name/label")
    score: float = Field(..., ge=0, le=100, description="Match confidence 0-100")
    match: bool = Field(default=False, description="True if high-confidence match")
    type: list[dict[str, str]] = Field(default_factory=list, description="Entity types")
    description: Optional[str] = Field(default=None, description="Brief description")


class ServiceManifest(BaseModel):
    """W3C Reconciliation Service Manifest."""

    versions: list[str] = ["0.2"]
    name: str
    identifierSpace: str
    schemaSpace: str
    defaultTypes: list[dict[str, str]] = Field(default_factory=list)
    view: Optional[dict[str, str]] = None
    preview: Optional[dict[str, Any]] = None
    suggest: Optional[dict[str, Any]] = None
    extend: Optional[dict[str, Any]] = None


class ProfileConfig(BaseModel):
    """Configuration for a reconciliation profile loaded from YAML."""

    name: str
    prompt: str
    types: list[dict[str, str]]
    slug: Optional[str] = None
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_tokens: int = Field(default=800, ge=1)
    cache_ttl: int = Field(default=3600, ge=0)
    description: Optional[str] = None
    use_dspy: bool = False
```

- [ ] **Step 3: Verify imports work**

Run: `PYTHONPATH=src python -c "from model_reconciler.config import Settings; from model_reconciler.models import ProfileConfig; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/model_reconciler/config.py src/model_reconciler/models.py
git commit -m "feat: add config and Pydantic models"
```

---

### Task 3: Profile Loader

**Files:**
- Create: `src/model_reconciler/profiles.py`

- [ ] **Step 1: Create `src/model_reconciler/profiles.py`**

```python
"""Profile loader: YAML files -> validated ProfileConfig objects."""

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from model_reconciler.models import ProfileConfig

logger = logging.getLogger(__name__)


def load_profile(path: Path) -> ProfileConfig:
    """Load and validate a single profile YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Validated ProfileConfig with slug derived from filename if not set.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the YAML is invalid or missing required fields.
    """
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Profile must be a YAML mapping: {path}")

    if "slug" not in data or data["slug"] is None:
        data["slug"] = path.stem

    try:
        return ProfileConfig(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid profile {path}: {e}") from e


def load_all_profiles(profiles_dir: Path) -> list[ProfileConfig]:
    """Load all YAML profiles from a directory.

    Args:
        profiles_dir: Directory containing .yaml profile files.

    Returns:
        List of validated ProfileConfig objects.

    Raises:
        ValueError: If duplicate slugs are found.
    """
    profiles: list[ProfileConfig] = []
    yaml_files = sorted(profiles_dir.glob("*.yaml"))

    for path in yaml_files:
        profile = load_profile(path)
        profiles.append(profile)
        logger.info(f"Loaded profile: {profile.slug} ({profile.name})")

    seen: set[str] = set()
    for p in profiles:
        if p.slug in seen:
            raise ValueError(
                f"Duplicate slug: '{p.slug}' — each profile needs a unique slug"
            )
        seen.add(p.slug)

    return profiles
```

- [ ] **Step 2: Verify import works**

Run: `PYTHONPATH=src python -c "from model_reconciler.profiles import load_profile; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/model_reconciler/profiles.py
git commit -m "feat: add profile YAML loader with validation"
```

---

### Task 4: Profile YAMLs + Profile Smoke Test

**Files:**
- Create: `profiles/library.yaml`
- Create: `profiles/general.yaml`
- Create: `tests/fixtures/valid.yaml`
- Create: `tests/conftest.py`
- Create: `tests/test_profiles.py`

- [ ] **Step 1: Create `profiles/library.yaml`**

```yaml
name: "SOAS Library Reconciliation"

prompt: |
  You are an expert at matching search queries to controlled vocabulary terms
  and authority records. Your task is to find the best matching entities for
  the given query.

  Guidelines:
  1. Consider both exact matches and semantic equivalents
  2. Account for synonyms, alternate forms, and related concepts
  3. Prefer more specific matches over general ones
  4. Consider the entity type filter if provided
  5. Assign confidence scores from 0-100 based on match quality:
     - 95-100: Exact or near-exact match
     - 80-94: Strong semantic match
     - 60-79: Related concept, good candidate
     - 40-59: Loosely related
     - Below 40: Weak match, include only if no better options

  Return your matches as a JSON array with the following structure:
  [
    {
      "id": "unique_identifier",
      "name": "Entity preferred label",
      "score": 85,
      "description": "Brief explanation of why this matches"
    }
  ]

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

- [ ] **Step 2: Create `profiles/general.yaml`**

```yaml
name: "General Entity Reconciliation"

prompt: |
  Match the query to known entities. Return a JSON array of matches with fields:
  id (unique identifier), name (entity label), score (0-100 confidence),
  and description (brief explanation).

  Score guidelines:
  - 95-100: Exact match
  - 80-94: Strong match
  - 60-79: Related concept
  - Below 60: Weak match

types:
  - id: "entity"
    name: "Entity"
```

- [ ] **Step 3: Create `tests/fixtures/valid.yaml`**

```yaml
name: "Test Profile"
prompt: "Return matches as JSON array."
types:
  - id: "test"
    name: "Test Type"
```

- [ ] **Step 4: Create `tests/conftest.py`**

```python
"""Shared test fixtures."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Point at test fixtures for profile loading tests
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture
def test_client():
    """Create a test client with profiles loaded from tests/fixtures/."""
    os.environ["PROFILES_DIR"] = str(FIXTURES_DIR)
    os.environ["LLM_BASE_URL"] = "http://fake:8080/v1"

    # Import after env is set so Settings picks up overrides
    from model_reconciler.main import create_app

    app = create_app()
    with TestClient(app) as client:
        yield client
```

- [ ] **Step 5: Create `tests/test_profiles.py`**

```python
"""Smoke tests for profile loading and validation."""

from pathlib import Path

import pytest

from model_reconciler.profiles import load_all_profiles, load_profile


def test_load_valid_profile(fixtures_dir):
    """Valid YAML loads and slug is derived from filename."""
    profile = load_profile(fixtures_dir / "valid.yaml")
    assert profile.name == "Test Profile"
    assert profile.slug == "valid"
    assert len(profile.types) == 1
    assert profile.temperature == 0.1


def test_load_missing_file(tmp_path):
    """Missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_profile(tmp_path / "nonexistent.yaml")


def test_load_missing_required_field(tmp_path):
    """YAML missing required 'types' field raises ValueError."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: Bad\nprompt: No types\n")
    with pytest.raises(ValueError, match="Invalid profile"):
        load_profile(bad)


def test_load_all_profiles(fixtures_dir):
    """load_all_profiles returns list of validated configs."""
    profiles = load_all_profiles(fixtures_dir)
    assert len(profiles) >= 1
    assert all(p.slug is not None for p in profiles)
```

- [ ] **Step 6: Run profile tests**

Run: `PYTHONPATH=src pytest tests/test_profiles.py -v`
Expected: 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add profiles/ tests/
git commit -m "feat: add profile YAMLs and profile loader smoke tests"
```

---

### Task 5: LLM Client

**Files:**
- Create: `src/model_reconciler/llm.py`

- [ ] **Step 1: Create `src/model_reconciler/llm.py`**

```python
"""Thin LLM client — async httpx call to any OpenAI-compatible endpoint."""

import logging

import httpx

logger = logging.getLogger(__name__)


async def chat_completion(
    base_url: str,
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 800,
) -> str:
    """Call an OpenAI-compatible /chat/completions endpoint.

    Args:
        base_url: Base URL (e.g. http://localhost:8080/v1).
        messages: Chat messages (system + user).
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate.

    Returns:
        The response content string.

    Raises:
        httpx.HTTPStatusError: On non-2xx response.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"

    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]
```

- [ ] **Step 2: Verify import works**

Run: `PYTHONPATH=src python -c "from model_reconciler.llm import chat_completion; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/model_reconciler/llm.py
git commit -m "feat: add async LLM client for OpenAI-compat endpoints"
```

---

### Task 6: Reconciliation Logic

**Files:**
- Create: `src/model_reconciler/reconcile.py`

- [ ] **Step 1: Create `src/model_reconciler/reconcile.py`**

```python
"""Core reconciliation logic — stateless functions, no framework coupling."""

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
    """Build prompt from profile, call LLM, parse response, return candidates.

    Args:
        query: The reconciliation query.
        profile: Profile with prompt and settings.
        base_url: OpenAI-compatible endpoint URL.

    Returns:
        List of candidates, truncated to query.limit.
    """
    if profile.use_dspy:
        raise NotImplementedError(
            f"Profile '{profile.slug}' has use_dspy=true, but DSPy support "
            "is not yet implemented. Set use_dspy: false or omit it."
        )

    messages = _build_messages(query, profile)

    try:
        raw = await chat_completion(
            base_url=base_url,
            messages=messages,
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
        )
    except Exception as e:
        logger.error(f"LLM call failed for profile '{profile.slug}': {e}")
        return []

    candidates = parse_llm_response(raw)
    return candidates[: query.limit]


def _build_messages(
    query: ReconciliationQuery, profile: ProfileConfig
) -> list[dict]:
    """Compose system + user messages for the LLM."""
    user_content = f"Query: {query.query}"
    if query.type:
        user_content += f"\nType: {query.type}"

    return [
        {"role": "system", "content": profile.prompt},
        {"role": "user", "content": user_content},
    ]


def parse_llm_response(text: str) -> list[ReconciliationCandidate]:
    """Parse JSON from LLM output into ReconciliationCandidate list.

    Handles:
    - Direct JSON array: [{"id": ..., "name": ..., "score": ...}, ...]
    - Wrapped in object: {"matches": [...]} or {"results": [...]}
    - Markdown-wrapped: ```json ... ```

    Returns empty list on parse failure.
    """
    try:
        cleaned = text.strip()

        # Strip markdown code fences if present
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        data = json.loads(cleaned)

        # Handle array or object wrapper
        if isinstance(data, list):
            matches = data
        elif isinstance(data, dict):
            for key in ("matches", "results", "entities"):
                if key in data and isinstance(data[key], list):
                    matches = data[key]
                    break
            else:
                logger.warning(f"Unexpected JSON structure: {list(data.keys())}")
                return []
        else:
            return []

        return [
            ReconciliationCandidate(
                id=m.get("id", f"gen_{i}"),
                name=m.get("name", ""),
                score=float(m.get("score", 50)),
                match=float(m.get("score", 0)) >= 90,
                description=m.get("description", m.get("reasoning", "")),
            )
            for i, m in enumerate(matches)
        ]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM output: {e}")
        return []
```

- [ ] **Step 2: Verify import works**

Run: `PYTHONPATH=src python -c "from model_reconciler.reconcile import reconcile_query, parse_llm_response; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/model_reconciler/reconcile.py
git commit -m "feat: add reconciliation logic with JSON parsing"
```

---

### Task 7: FastAPI Application

**Files:**
- Create: `src/model_reconciler/main.py`

- [ ] **Step 1: Create `src/model_reconciler/main.py`**

```python
"""Model Reconciler — W3C Reconciliation API with profile-based routing."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from cachetools import TTLCache
from fastapi import FastAPI, Form, HTTPException, Request
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
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings override (for testing).

    Returns:
        Configured FastAPI app with profile routes mounted.
    """
    if settings is None:
        settings = Settings()

    logging.basicConfig(level=settings.log_level)
    logger = logging.getLogger(__name__)

    application = FastAPI(
        title="Model Reconciler",
        description="Profile-driven W3C Reconciliation API",
        version="0.1.0",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Profile registry: slug -> (profile, cache)
    registry: dict[str, tuple[ProfileConfig, TTLCache]] = {}

    profiles_dir = Path(settings.profiles_dir)
    if profiles_dir.exists():
        for profile in load_all_profiles(profiles_dir):
            cache: TTLCache = TTLCache(maxsize=1000, ttl=profile.cache_ttl)
            registry[profile.slug] = (profile, cache)
            logger.info(f"Mounted /reconcile/{profile.slug} -> {profile.name}")
    else:
        logger.warning(f"Profiles directory not found: {profiles_dir}")

    # Store in app state for route access
    application.state.registry = registry
    application.state.settings = settings

    @application.get("/")
    async def list_profiles():
        """List all available reconciliation services."""
        return [
            {
                "slug": slug,
                "name": profile.name,
                "description": profile.description or profile.name,
                "url": f"/reconcile/{slug}",
            }
            for slug, (profile, _cache) in registry.items()
        ]

    @application.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "profiles_loaded": len(registry),
            "profiles": list(registry.keys()),
        }

    @application.get("/reconcile/{slug}")
    async def get_manifest(slug: str):
        """Return W3C Reconciliation Service manifest for a profile."""
        if slug not in registry:
            raise HTTPException(status_code=404, detail=f"Profile not found: {slug}")

        profile, _cache = registry[slug]

        return ServiceManifest(
            versions=["0.2"],
            name=profile.name,
            identifierSpace=f"/entity/{slug}/",
            schemaSpace=f"/schema/{slug}/",
            defaultTypes=[
                {"id": t["id"], "name": t["name"]} for t in profile.types
            ],
        ).model_dump()

    @application.post("/reconcile/{slug}")
    async def reconcile(
        slug: str,
        request: Request,
        queries: Optional[str] = Form(default=None),
        query: Optional[str] = Form(default=None),
    ):
        """Main reconciliation endpoint (W3C Reconciliation API)."""
        if slug not in registry:
            raise HTTPException(status_code=404, detail=f"Profile not found: {slug}")

        profile, cache = registry[slug]
        base_url = settings.llm_base_url

        # Batch queries (OpenRefine standard)
        if queries:
            try:
                queries_dict = json.loads(queries)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

            results = await _process_batch(profile, cache, queries_dict, base_url)
            return JSONResponse(content=results)

        # Single query
        if query:
            result = await _process_single(profile, cache, query, base_url)
            return JSONResponse(content={"result": result})

        # No query — return manifest
        return await get_manifest(slug)

    async def _process_batch(
        profile: ProfileConfig,
        cache: TTLCache,
        queries_dict: dict[str, Any],
        base_url: str,
    ) -> dict[str, Any]:
        """Process batch reconciliation queries."""
        results = {}
        tasks = []
        query_ids = []

        for qid, qdata in queries_dict.items():
            query_ids.append(qid)
            q = ReconciliationQuery(
                query=qdata.get("query", ""),
                type=qdata.get("type"),
                limit=qdata.get("limit", 5),
                properties=qdata.get("properties", []),
            )

            cache_key = f"{q.query}:{q.type}:{q.limit}"
            if cache_key in cache:
                tasks.append(None)  # placeholder for cached result
                results[qid] = {
                    "result": [r.model_dump() for r in cache[cache_key]]
                }
            else:
                tasks.append((qid, q, cache_key))

        # Run uncached queries concurrently
        async def run_query(qid, q, cache_key):
            candidates = await reconcile_query(q, profile, base_url)
            cache[cache_key] = candidates
            return qid, [r.model_dump() for r in candidates]

        pending = [
            run_query(item[0], item[1], item[2])
            for item in tasks
            if item is not None
        ]

        if pending:
            completed = await asyncio.gather(*pending)
            for qid, result_list in completed:
                results[qid] = {"result": result_list}

        return results

    async def _process_single(
        profile: ProfileConfig,
        cache: TTLCache,
        query_text: str,
        base_url: str,
    ) -> list[dict[str, Any]]:
        """Process a single reconciliation query."""
        q = ReconciliationQuery(query=query_text)
        cache_key = f"{q.query}:{q.type}:{q.limit}"

        if cache_key in cache:
            return [r.model_dump() for r in cache[cache_key]]

        candidates = await reconcile_query(q, profile, base_url)
        cache[cache_key] = candidates
        return [r.model_dump() for r in candidates]

    return application


# Module-level app for uvicorn
app = create_app()
```

- [ ] **Step 2: Verify app imports**

Run: `PYTHONPATH=src python -c "from model_reconciler.main import create_app; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/model_reconciler/main.py
git commit -m "feat: add FastAPI app with profile routing and caching"
```

---

### Task 8: Smoke Tests — Health + Manifest

**Files:**
- Create: `tests/test_health.py`
- Create: `tests/test_manifest.py`

- [ ] **Step 1: Create `tests/test_health.py`**

```python
"""Smoke tests for health endpoint."""


def test_health_returns_200(test_client):
    """GET /health returns 200 with profile count."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["profiles_loaded"] >= 1
    assert isinstance(data["profiles"], list)


def test_root_lists_profiles(test_client):
    """GET / returns list of loaded profiles."""
    response = test_client.get("/")
    assert response.status_code == 200
    profiles = response.json()
    assert isinstance(profiles, list)
    assert len(profiles) >= 1
    for p in profiles:
        assert "slug" in p
        assert "name" in p
        assert "url" in p
```

- [ ] **Step 2: Create `tests/test_manifest.py`**

```python
"""Smoke tests for W3C manifest endpoint."""


def test_manifest_valid_slug(test_client):
    """GET /reconcile/{slug} returns valid W3C manifest shape."""
    # Get first available slug from root
    profiles = test_client.get("/").json()
    slug = profiles[0]["slug"]

    response = test_client.get(f"/reconcile/{slug}")
    assert response.status_code == 200

    manifest = response.json()
    assert "versions" in manifest
    assert "0.2" in manifest["versions"]
    assert "name" in manifest
    assert "identifierSpace" in manifest
    assert "schemaSpace" in manifest
    assert "defaultTypes" in manifest
    assert isinstance(manifest["defaultTypes"], list)


def test_manifest_unknown_slug(test_client):
    """GET /reconcile/nonexistent returns 404."""
    response = test_client.get("/reconcile/nonexistent")
    assert response.status_code == 404
```

- [ ] **Step 3: Run all tests**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: All tests PASS (4 profile + 2 health + 2 manifest = 8 tests)

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "feat: add smoke tests for health, profiles, and manifest"
```

---

### Task 9: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create `README.md`**

```markdown
# Model Reconciler

Profile-driven W3C Reconciliation API. Model-agnostic — works with any OpenAI-compatible inference engine (llama-server, Ollama, omlx, vLLM, LM Studio, etc.).

## How It Works

Drop a YAML profile into `profiles/` — each profile defines a prompt and entity types. Each profile mounts as its own `/reconcile/{slug}` endpoint. Point OpenRefine at the URL for the profile you want.

```
profiles/library.yaml  -->  /reconcile/library
profiles/general.yaml  -->  /reconcile/general
```

The inference engine is external — you run it on the host, the API talks to it via `LLM_BASE_URL`.

## Quick Start

Prerequisites: [Docker Desktop](https://www.docker.com/products/docker-desktop/) and an OpenAI-compatible inference engine running on your host.

1. Start your inference engine:
   ```bash
   # Example: llama-server
   llama-server -hf ggml-org/gemma-3-4b-it-GGUF:Q4_K_M --port 8080

   # Example: Ollama
   ollama serve  # default port 11434
   ```

2. Start the API:
   ```bash
   make
   ```

3. In OpenRefine: Column dropdown > Reconcile > Start reconciling...
   Add Standard Service: `http://localhost:8001/reconcile/library`

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | List all available reconciliation profiles |
| `GET /health` | Health check with loaded profile count |
| `GET /reconcile/{slug}` | W3C manifest for a profile |
| `POST /reconcile/{slug}` | Reconcile queries against a profile |

## Profiles

A profile is a YAML file with three required fields:

```yaml
name: "SOAS Library Reconciliation"
prompt: |
  You are a library cataloguing expert. Match the query to known
  bibliographic entities. Return JSON array of matches...
types:
  - id: "topical"
    name: "Topical Subject"
```

Optional fields (all have defaults):

| Field | Default | Description |
|-------|---------|-------------|
| `slug` | filename stem | URL path segment |
| `temperature` | `0.1` | LLM sampling temperature |
| `max_tokens` | `800` | Max generation tokens |
| `cache_ttl` | `3600` | Result cache TTL (seconds) |
| `description` | profile name | For W3C manifest |
| `use_dspy` | `false` | Opt-in DSPy structured output |

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_BASE_URL` | `http://host.docker.internal:8080/v1` | Inference engine endpoint |
| `PROFILES_DIR` | `profiles` | Profile YAML directory |
| `LOG_LEVEL` | `INFO` | Logging level |

## Commands

| Command | Description |
|---------|-------------|
| `make` | Build and start API container |
| `make stop` | Stop services |
| `make test` | Run pytest suite (in Docker) |
| `make lint` | Run ruff linter (in Docker) |
| `make logs` | Follow logs |
| `make clean` | Remove containers + caches |

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
  │   profiles/*.yaml → routes   │
  │   /reconcile/{slug}          │
  └──────────────────────────────┘
```

- [ ] **Step 2: Run linter**

Run: `PYTHONPATH=src ruff check src/ tests/`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README with quick start and architecture"
```

---

## Verification Checklist

After all tasks are complete:

- [ ] `PYTHONPATH=src pytest tests/ -v` — all 8 tests pass
- [ ] `PYTHONPATH=src ruff check src/ tests/` — no lint errors
- [ ] `docker compose build` — image builds
- [ ] `docker compose up` — container starts, health check passes (with inference engine running)
- [ ] OpenRefine can add `http://localhost:8001/reconcile/library` as a reconciliation service
