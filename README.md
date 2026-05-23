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
