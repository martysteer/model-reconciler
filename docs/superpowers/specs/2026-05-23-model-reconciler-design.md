# Model Reconciler — Design Spec

**Date:** 2026-05-23
**Status:** Approved
**Predecessor:** granite-reconcile (this repo) — lessons learned, fresh build

---

## Goal

Build a model-agnostic, profile-driven W3C Reconciliation API. Containerized FastAPI app talks to any OpenAI-compatible inference engine running natively on the host. Compatible with OpenRefine.

## Decisions

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Language | Python | Continuity from granite-reconcile |
| Framework | FastAPI + uvicorn | Async, fast, OpenRefine-compatible |
| LLM transport | httpx → OpenAI-compat `/v1/chat/completions` | Model-agnostic by design |
| Structured output | JSON mode (default), DSPy (opt-in per profile) | Lean default, structured fallback |
| Profile system | YAML files, one per endpoint | Proven pattern from granite-reconcile |
| Inference engine | External — user's choice (llama-server, Ollama, omlx, vLLM, etc.) | Not our concern |
| Mac architecture | Native inference on host (Metal) + containerized API | GPU requires host access, no Metal passthrough in containers |
| Caching | In-memory TTL per profile | Simple, no external deps |
| W3C compliance | Core only (manifest + batch reconcile), extensible for future | YAGNI — preview/suggest/extend deferred |
| Testing | Smoke tests (health, profiles, manifest) | No live model needed |
| Tooling | pip + requirements.txt | Traditional, Docker-friendly |
| Project name | model-reconciler | Model-agnostic, descriptive |

---

## Architecture

```
Host (macOS, native, Metal-accelerated):
  ┌──────────────────────────────────┐
  │  Any OpenAI-compatible server    │
  │  (llama-server / Ollama / omlx)  │
  │  http://localhost:8080/v1        │
  └──────────────┬───────────────────┘
                 │
        OpenAI-compat API
                 │
  Container (Docker/OrbStack):
  ┌──────────────┴───────────────────┐
  │      model-reconciler (FastAPI)  │
  │      http://localhost:8001       │
  │                                  │
  │  Startup:                        │
  │    profiles/*.yaml → validate    │
  │    → mount /reconcile/{slug}     │
  │                                  │
  │  Request flow:                   │
  │    OpenRefine POST               │
  │    → parse W3C batch query       │
  │    → check cache                 │
  │    → build prompt (from profile) │
  │    → call LLM (httpx)            │
  │    → parse JSON response         │
  │    → format W3C result           │
  │    → cache + return              │
  └──────────────────────────────────┘
```

Key separation: inference engine = infrastructure config (user sets `LLM_BASE_URL`). Reconciliation logic = app config (YAML profiles). They never meet.

---

## Project Structure

```
model-reconciler/
├── src/model_reconciler/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, profile discovery, route mounting
│   ├── config.py         # Settings: LLM_BASE_URL, PROFILES_DIR, LOG_LEVEL
│   ├── models.py         # Pydantic: W3C types, ProfileConfig, ReconciliationCandidate
│   ├── profiles.py       # YAML loader + validation
│   ├── llm.py            # Thin httpx wrapper for OpenAI-compat /v1/chat/completions
│   └── reconcile.py      # Core logic: build prompt, call LLM, parse, format W3C
├── profiles/
│   ├── library.yaml      # SOAS library reconciliation
│   └── general.yaml      # General entity matching
├── tests/
│   ├── test_health.py    # Smoke: health endpoint, profile count
│   ├── test_profiles.py  # Smoke: YAML loading, validation
│   └── test_manifest.py  # Smoke: W3C manifest format
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

### Module Responsibilities

| Module | Does | Doesn't |
|--------|------|---------|
| `main.py` | Route mounting, request parsing, response formatting, cache | Know about LLM |
| `config.py` | Global env vars (3 total) | Profile-specific config |
| `models.py` | Pydantic types for validation | Business logic |
| `profiles.py` | Load YAML, validate, derive slug | Know about routes |
| `llm.py` | HTTP call to OpenAI-compat endpoint | Know about reconciliation |
| `reconcile.py` | Build prompt, call llm.py, parse JSON, return candidates | Know about FastAPI |

---

## Profile Schema

### Required Fields

```yaml
# profiles/library.yaml
name: "SOAS Library Reconciliation"

prompt: |
  You are a library cataloguing expert. Given a query term,
  match it to known bibliographic entities. Return a JSON array
  of matches, each with: name, score (0-100), description, type.

types:
  - id: /library/subject
    name: Subject Heading
  - id: /library/name
    name: Personal Name
```

### Optional Fields

| Field | Default | Notes |
|-------|---------|-------|
| `slug` | filename stem | `library.yaml` → `library` |
| `temperature` | `0.1` | Low for deterministic matching |
| `max_tokens` | `800` | Enough for JSON array |
| `cache_ttl` | `3600` | 1 hour |
| `description` | `"{name}"` | For W3C manifest |
| `use_dspy` | `false` | Opt-in structured output via DSPy |

### Dropped from granite-reconcile

| Field | Reason |
|-------|--------|
| `model` | Infrastructure concern, not profile concern |
| `signature` | DSPy-specific, only relevant when `use_dspy: true` |
| `few_shot` | Deferred to v2 |
| `vllm_url` | Replaced by global `LLM_BASE_URL` |

---

## LLM Client

`llm.py` — one async function, no class, no state:

```python
async def chat_completion(
    base_url: str,
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 800,
) -> str:
    """Call OpenAI-compat endpoint. Return content string."""
    # POST {base_url}/chat/completions
    # Request JSON mode: response_format: {"type": "json_object"}
    # Return response.choices[0].message.content
```

Works with: llama-server, Ollama, omlx, vLLM, Docker Model Runner, LM Studio — anything exposing `/v1/chat/completions`.

---

## Reconciliation Logic

`reconcile.py` — stateless async function:

```python
async def reconcile_query(
    query: ReconciliationQuery,
    profile: ProfileConfig,
    base_url: str,
) -> list[ReconciliationCandidate]:
    """Build prompt from profile, call LLM, parse JSON, return candidates."""
    # 1. Compose system message from profile.prompt
    # 2. Compose user message from query.query + query.type
    # 3. Call llm.chat_completion()
    # 4. Parse JSON array from response
    # 5. Map to ReconciliationCandidate list
    # 6. Truncate to query.limit
```

- Default path: JSON mode, no DSPy
- When `use_dspy: true`: `reconcile_dspy()` in `reconcile.py`, same return type. Uses DSPy `ChainOfThought` with a built-in `EntityMatcher` signature. DSPy is an optional dependency — only imported when a profile requests it.
- Bad JSON from LLM: return empty results + log warning

---

## W3C Reconciliation API — Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | List all loaded profiles |
| `/health` | GET | Health check with profile count |
| `/reconcile/{slug}` | GET | W3C service manifest for profile |
| `/reconcile/{slug}` | POST | Batch reconciliation (OpenRefine format) |

### W3C Manifest Shape

```json
{
  "versions": ["0.2"],
  "name": "SOAS Library Reconciliation",
  "identifierSpace": "/entity/library/",
  "schemaSpace": "/schema/library/",
  "defaultTypes": [
    {"id": "/library/subject", "name": "Subject Heading"}
  ]
}
```

### Batch Query Shape (OpenRefine)

```json
{
  "q0": {"query": "Shakespeare", "type": "/library/name", "limit": 5},
  "q1": {"query": "Postcolonialism", "type": "/library/subject", "limit": 5}
}
```

### Response Shape

```json
{
  "q0": {"result": [
    {"id": "gen_0", "name": "Shakespeare, William", "score": 95, "match": true, "description": "..."}
  ]},
  "q1": {"result": [...]}
}
```

### Future Extensions (not in v1)

- Preview service (`/reconcile/{slug}/preview`)
- Suggest API (`/reconcile/{slug}/suggest`)
- Data extension (`/reconcile/{slug}/extend`)
- Property filtering in queries

Architecture supports these — add new route handlers in `main.py`, new logic functions in `reconcile.py`.

---

## Docker & Deployment

### docker-compose.yml

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
```

No inference engine in compose. User runs their own on host. `host.docker.internal` bridges container to host.

### Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
CMD ["uvicorn", "model_reconciler.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

Production: no `--reload`, no src volume mount, no dev deps.

### Configuration — 3 env vars total

| Var | Default | Purpose |
|-----|---------|---------|
| `LLM_BASE_URL` | `http://host.docker.internal:8080/v1` | Inference engine endpoint |
| `PROFILES_DIR` | `profiles` | Profile YAML directory |
| `LOG_LEVEL` | `INFO` | Logging level |

### Makefile

| Target | Does |
|--------|------|
| `make` | `docker compose up --build` |
| `make stop` | `docker compose down` |
| `make test` | `docker compose run --rm --no-deps api pytest` |
| `make lint` | `docker compose run --rm --no-deps api ruff check` |
| `make clean` | Remove containers + caches |

---

## Testing

Smoke tests only for v1. All use FastAPI `TestClient`. No live LLM.

| Test file | Covers |
|-----------|--------|
| `test_health.py` | GET `/health` returns 200, profile count matches |
| `test_profiles.py` | Valid YAML loads, missing required fields rejected, slug derivation |
| `test_manifest.py` | GET `/reconcile/{slug}` returns valid W3C manifest shape |

Mock `llm.chat_completion` where needed.

---

## Dependencies

### requirements.txt (production)

```
fastapi
uvicorn[standard]
httpx
pyyaml
pydantic
pydantic-settings
cachetools
```

### requirements-dev.txt (testing)

```
pytest
pytest-asyncio
httpx
ruff
```

### Optional

```
dspy  # only if any profile sets use_dspy: true
```

---

## What Changed from granite-reconcile

| Aspect | granite-reconcile | model-reconciler |
|--------|-------------------|------------------|
| Inference engine | Docker Model Runner (locked in) | Any OpenAI-compat (user's choice) |
| Engine in compose | Yes (DMR `models:` block) | No (external, host-native) |
| Pipeline | DSPy always, class-based, stateful | Stateless functions, DSPy opt-in |
| LLM client | DSPy's internal client | Plain httpx |
| Profile required fields | `name`, `prompt` | `name`, `prompt`, `types` |
| `model` in profile | Yes | No (infrastructure concern) |
| Mac GPU | Via Docker Model Runner host process | Via any native inference engine |
| Caching | Per-pipeline instance | Per-route in main.py |
| Tests | Full pytest suite | Smoke tests |
| Project name | granite-reconcile | model-reconciler |
