# Model Reconciler — Design Spec

**Date:** 2026-05-23
**Status:** Approved

---

## Problem

[OpenRefine](https://openrefine.org/) is the standard tool for cleaning and reconciling messy data in libraries, archives, and research workflows. Its reconciliation feature matches column values (e.g. "Shakespeare", "Postcolonialism") against known entities via a [W3C Reconciliation Service API](https://www.w3.org/community/reports/reconciliation/CG-FINAL-specs-0.2-20230410/).

Existing reconciliation services (Wikidata, VIAF, FAST) query remote databases. There is no simple way to run a local, LLM-powered reconciliation service that uses a small language model to match entities — one that works offline, can be customised with domain-specific prompts, and doesn't depend on a particular model or inference engine.

## Goal

Build **model-reconciler**: a model-agnostic, profile-driven W3C Reconciliation API. A containerized Python app that translates OpenRefine's reconciliation requests into LLM prompts, sends them to any OpenAI-compatible inference engine, and returns W3C-compliant results.

**Key properties:**
- **Model-agnostic** — works with any OpenAI-compatible endpoint (llama-server, Ollama, omlx, vLLM, LM Studio, Docker Model Runner)
- **Profile-driven** — each YAML profile defines a prompt and entity types, mounting as its own `/reconcile/{slug}` endpoint
- **Containerized** — the API runs in Docker; the inference engine runs natively on the host for GPU access
- **OpenRefine-compatible** — implements the W3C Reconciliation Service API v0.2
- **API key support** — optional `LLM_API_KEY` for authenticated inference endpoints

---

## Design Constraints

### Why the inference engine runs outside the container

On Apple Silicon Macs, Docker containers have no access to the Metal GPU. Docker Desktop, OrbStack, and Apple's own `container` tool all run Linux containers inside a VM with no GPU passthrough. Metal-accelerated inference (3-5x faster than CPU) requires a native host process.

The clean architecture follows a well-established pattern: **inference engine runs natively on the host, everything else is containerized.** The container reaches the host inference engine via `host.docker.internal`. This also makes the API completely inference-engine-agnostic — it only needs an HTTP endpoint that speaks the OpenAI chat completions protocol.

### Why profiles instead of a single endpoint

Different reconciliation tasks need different prompts. A library cataloguer matching subject headings needs different LLM instructions than someone matching corporate names or geographic entities. Profiles let you define multiple reconciliation services from a single deployment — each with its own prompt, entity types, and tuning parameters — without writing code.

### Why JSON mode instead of a structured output framework

Modern LLMs reliably return valid JSON when requested via `response_format: {"type": "json_object"}`. For the default code path, a thin `httpx` call with JSON mode is simpler and has fewer dependencies than a framework like DSPy. However, DSPy's structured output guarantees (retries, signature enforcement) are valuable for complex reconciliation tasks, so it's available as an opt-in per profile.

---

## Decisions

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Language | Python | FastAPI ecosystem, LLM tooling |
| Framework | FastAPI + uvicorn | Async, fast, good OpenAPI docs |
| LLM transport | httpx → OpenAI-compat `/v1/chat/completions` | Model-agnostic, zero coupling |
| Structured output | JSON mode (default), DSPy (opt-in per profile) | Lean default, structured fallback when needed |
| Profile system | YAML files, one per endpoint | No code changes to add reconciliation services |
| Inference engine | External — user's choice | Not our concern; any OpenAI-compat server |
| Deployment | API in Docker container; inference native on host | GPU access requires host process |
| Authentication | Optional `LLM_API_KEY` env var | Bearer token for hosted/authenticated endpoints |
| Caching | In-memory TTL per profile | Simple, no external dependencies |
| W3C compliance | Core only (manifest + batch reconcile) | Extensible architecture for future additions |
| Testing | Smoke tests (health, profiles, manifest) | No live model needed for CI |
| Tooling | pip + requirements.txt | Traditional, Docker-friendly |

---

## Architecture

```
Host (native, GPU-accelerated):
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

**Key separation:** the inference engine is infrastructure config (user sets `LLM_BASE_URL`). Reconciliation logic is app config (YAML profiles). They never meet. Swapping from llama-server to Ollama is a one-line environment variable change.

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

Each module has one job and doesn't know about its neighbours' internals:

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

Profiles are YAML files dropped into the `profiles/` directory. Each file becomes a reconciliation endpoint at `/reconcile/{slug}`, where slug defaults to the filename stem.

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

- **`name`** — Human-readable service name, shown in the W3C manifest and OpenRefine's service list.
- **`prompt`** — System prompt sent to the LLM. Should instruct the model what kind of entities to match and what JSON structure to return.
- **`types`** — W3C entity types this service can reconcile against. Each has an `id` and `name`. OpenRefine shows these as type options when reconciling.

### Optional Fields

| Field | Default | Notes |
|-------|---------|-------|
| `slug` | filename stem | `library.yaml` → `library`. Override to decouple URL from filename. |
| `temperature` | `0.1` | Low for deterministic, consistent matching. |
| `max_tokens` | `800` | Sufficient for a JSON array of 5-10 matches. |
| `cache_ttl` | `3600` | 1 hour. Set to `0` to disable caching. |
| `description` | `"{name}"` | Used in the W3C manifest `name` field and profile listings. |
| `use_dspy` | `false` | When `true`, uses DSPy `ChainOfThought` for structured output instead of raw JSON mode. Requires `dspy` to be installed. |

### Design Rationale

- **No `model` field.** Which model is served is an infrastructure decision — set once in the inference engine config, shared by all profiles. Profiles only control the prompt.
- **No per-profile LLM URL.** All profiles talk to the same `LLM_BASE_URL`. Multi-model setups are handled by running multiple inference engines, not by profile config.
- **No few-shot examples in v1.** Few-shot examples are a powerful tuning mechanism but add schema complexity. Deferred to a future version.

---

## LLM Client

`llm.py` — one async function, no class, no state:

```python
async def chat_completion(
    base_url: str,
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 800,
    api_key: str | None = None,
) -> str:
    """Call OpenAI-compat endpoint. Return content string."""
    # POST {base_url}/chat/completions
    # If api_key: set Authorization: Bearer header
    # Request JSON mode: response_format: {"type": "json_object"}
    # Return response.choices[0].message.content
```

This function works with any server exposing the OpenAI `/v1/chat/completions` endpoint: llama-server, Ollama, omlx, vLLM, Docker Model Runner, LM Studio. The caller doesn't know or care which engine is behind the URL.

When `api_key` is provided, the request includes an `Authorization: Bearer {api_key}` header. This supports hosted inference endpoints (OpenRouter, Together, etc.) alongside local engines that don't require auth.

**Why not use the `openai` Python SDK?** httpx is already a dependency (used by FastAPI's test client), and the chat completions call is a single POST with a well-known JSON schema. Adding the `openai` package would be one more dependency for one HTTP call.

---

## Reconciliation Logic

`reconcile.py` — stateless async functions, no framework coupling:

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

### Message Construction

The LLM receives two messages:

| Role | Content |
|------|---------|
| `system` | The profile's `prompt` field — full instructions on entity type, matching strategy, and JSON output format |
| `user` | `Query: {query_text}` with optional `Type: {type}` on a second line |

The profile prompt is responsible for telling the model what JSON structure to return. This keeps the code simple — no template engine, no format negotiation.

### Response Parsing

The `parse_llm_response()` function handles three common LLM output shapes:

1. **Direct JSON array:** `[{"id": "...", "name": "...", "score": 85}, ...]`
2. **Object wrapper:** `{"matches": [...]}` or `{"results": [...]}` or `{"entities": [...]}`
3. **Markdown-wrapped:** `` ```json [...] ``` `` (shouldn't happen with JSON mode, but handled as defense)

Each match is mapped to a `ReconciliationCandidate` with sensible defaults:
- `id` defaults to `gen_0`, `gen_1`, etc. if not provided
- `score` defaults to `50` if not provided
- `match` is `true` when `score >= 90`
- `description` falls back to `reasoning` field (common LLM output pattern)

On parse failure: return empty results and log a warning. Reconciliation is best-effort — a parse error for one query shouldn't crash the batch.

### DSPy Opt-In

When a profile sets `use_dspy: true`, the reconciliation uses DSPy's `ChainOfThought` with an `EntityMatcher` signature instead of raw JSON mode. This gives:
- Automatic retries on malformed output
- Signature enforcement (input/output field contracts)
- Chain-of-thought reasoning before the final answer

DSPy is an optional dependency — only imported when a profile requests it. Not installed by default.

---

## W3C Reconciliation API

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | List all loaded profiles with slugs and URLs |
| `/health` | GET | Health check with loaded profile count |
| `/reconcile/{slug}` | GET | W3C service manifest for a profile |
| `/reconcile/{slug}` | POST | Batch reconciliation (OpenRefine sends queries here) |

### W3C Manifest (GET /reconcile/{slug})

When OpenRefine connects to a reconciliation service, it first fetches the manifest to learn the service name, supported types, and API version.

```json
{
  "versions": ["0.2"],
  "name": "SOAS Library Reconciliation",
  "identifierSpace": "/entity/library/",
  "schemaSpace": "/schema/library/",
  "defaultTypes": [
    {"id": "/library/subject", "name": "Subject Heading"},
    {"id": "/library/name", "name": "Personal Name"}
  ]
}
```

### Batch Query (POST /reconcile/{slug})

OpenRefine sends reconciliation queries as a form-encoded `queries` parameter containing JSON:

```json
{
  "q0": {"query": "Shakespeare", "type": "/library/name", "limit": 5},
  "q1": {"query": "Postcolonialism", "type": "/library/subject", "limit": 5}
}
```

Each query has:
- `query` — the text to match (required)
- `type` — optional type filter from the profile's `types` list
- `limit` — max results to return (default 5, max 25)

### Response Shape

```json
{
  "q0": {"result": [
    {"id": "gen_0", "name": "Shakespeare, William, 1564-1616", "score": 98, "match": true, "description": "Most prominent person with this name"},
    {"id": "gen_1", "name": "Shakespeare, William, 1564-1616 -- Criticism", "score": 72, "match": false, "description": "Related topical heading"}
  ]},
  "q1": {"result": [
    {"id": "gen_0", "name": "Postcolonialism", "score": 95, "match": true, "description": "Exact topical match"}
  ]}
}
```

Each result has:
- `id` — entity identifier
- `name` — entity label
- `score` — confidence 0-100
- `match` — `true` if high-confidence match (score >= 90)
- `description` — optional explanation
- `type` — optional entity type list

### Future Extensions (not in v1)

The W3C Reconciliation API v0.2 also specifies:
- **Preview service** (`/reconcile/{slug}/preview`) — HTML preview of an entity
- **Suggest API** (`/reconcile/{slug}/suggest`) — autocomplete for entity search
- **Data extension** (`/reconcile/{slug}/extend`) — fetch additional properties for matched entities
- **Property filtering** — use extra column data to improve matching

The architecture supports all of these as future additions: new route handlers in `main.py`, new logic functions in `reconcile.py`. No refactoring needed.

---

## Caching

Each profile gets its own in-memory TTL cache (via `cachetools.TTLCache`).

- **Cache key:** `{query_text}:{type}:{limit}`
- **Max entries:** 1000 per profile
- **TTL:** configurable per profile via `cache_ttl` (default 3600 seconds / 1 hour)
- **Scope:** lives in `main.py` alongside the route handlers, not in the reconciliation logic

Cache is lost on container restart. This is acceptable — LLM reconciliation results are non-deterministic anyway, and a cold cache just means slightly slower first requests.

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

**No inference engine in compose.** The user runs their own on the host. `host.docker.internal` is a Docker-provided hostname that resolves to the host machine — this is how the containerized API reaches the native inference engine.

Dev mode: `--reload` watches `./src` for code changes. Profiles are mounted read-only.

### Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
CMD ["uvicorn", "model_reconciler.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

Production builds omit `--reload`, the `./src` volume mount, and dev dependencies.

### Configuration

Four environment variables. Nothing else.

| Var | Default | Purpose |
|-----|---------|---------|
| `LLM_BASE_URL` | `http://host.docker.internal:8080/v1` | Inference engine endpoint |
| `LLM_API_KEY` | `None` | Optional Bearer token for authenticated endpoints |
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

Smoke tests only for v1. All use FastAPI's `TestClient` with a test fixture profile. No live LLM needed — the tests validate API shape and profile loading, not LLM output quality.

| Test file | Covers |
|-----------|--------|
| `test_health.py` | GET `/health` returns 200, profile count matches loaded YAMLs |
| `test_profiles.py` | Valid YAML loads correctly, missing required fields rejected, slug derived from filename |
| `test_manifest.py` | GET `/reconcile/{slug}` returns valid W3C manifest with required fields |

`conftest.py` provides a shared `test_client` fixture that creates the app with `PROFILES_DIR` pointed at a `tests/fixtures/` directory containing a minimal valid profile. `LLM_BASE_URL` is set to a fake URL — no inference engine needed.

Where tests exercise code paths that would call the LLM, `llm.chat_completion` is mocked.

---

## Dependencies

### requirements.txt (production)

```
fastapi
uvicorn[standard]
python-multipart
httpx
pyyaml
pydantic
pydantic-settings
cachetools
```

Eight dependencies. FastAPI and uvicorn for the web server. python-multipart for form data parsing. httpx for the LLM client. pyyaml for profile loading. pydantic and pydantic-settings for data validation and config. cachetools for TTL caching.

### requirements-dev.txt (testing)

```
pytest
pytest-asyncio
httpx
ruff
```

### Optional

```
dspy  # only needed if any profile sets use_dspy: true
```

---

## Usage

### Starting the Service

1. Start any OpenAI-compatible inference engine on the host:

```bash
# llama-server (llama.cpp)
llama-server -hf ggml-org/gemma-3-4b-it-GGUF:Q4_K_M --port 8080

# Ollama
ollama serve  # default port 11434, set LLM_BASE_URL accordingly

# omlx
omlx serve --port 8080
```

2. Start the API:

```bash
make
```

The API starts at `http://localhost:8001/`. Each profile YAML in `profiles/` appears as a reconciliation endpoint.

### Connecting OpenRefine

1. Open a project in OpenRefine
2. Column dropdown → Reconcile → Start reconciling...
3. Add Standard Service: `http://localhost:8001/reconcile/library`
4. Select the entity type to match against
5. Start reconciling

### Adding a New Reconciliation Service

1. Create a YAML file in `profiles/` with `name`, `prompt`, and `types`
2. Restart the API (`make stop && make`)
3. The new service appears at `/reconcile/{filename-stem}`

### Changing the Inference Engine

Change one environment variable:

```bash
# In docker-compose.yml:
LLM_BASE_URL=http://host.docker.internal:11434/v1  # Ollama
LLM_BASE_URL=http://host.docker.internal:8080/v1   # llama-server
LLM_BASE_URL=http://host.docker.internal:8000/v1   # omlx / vLLM
```

No code changes. No profile changes. The API doesn't know or care what's behind the URL.
