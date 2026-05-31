# LLM Client Improvements — Design Spec

**Date:** 2026-05-31
**Status:** Approved
**Predecessor:** `2026-05-23-model-reconciler-design-v2.md`

---

## Problem

model-reconciler's LLM client (`llm.py`) works but has four weaknesses:

1. **New HTTP connection per request** — a batch of N queries creates N `httpx.AsyncClient` instances with separate TCP/TLS handshakes
2. **Unbounded concurrency** — `asyncio.gather` fires all uncached queries simultaneously, overwhelming local engines and triggering rate limits on hosted APIs
3. **No schema enforcement** — `json_object` mode gives valid JSON but no shape guarantee, requiring a 3-shape fallback parser
4. **No provider awareness** — sends `response_format: json_object` and all params to every engine, but some engines don't support them

## Goal

Four targeted improvements to `llm.py` and its callers. No new modules. No new endpoints. Same external API surface.

---

## Decisions

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| HTTP client lifecycle | App-scoped, created in lifespan | Connection reuse across all requests |
| Concurrency control | Global `asyncio.Semaphore` | Single `LLM_BASE_URL` shared by all profiles |
| Concurrency default | 4 | Reasonable for local engines + within hosted rate limits |
| Response format | `json_schema` primary, `json_object` fallback | Schema enforcement eliminates most parse failures |
| Schema scope | Fixed canonical schema in code | All profiles return same shape — consistent, simple |
| Provider detection | Auto-detect from URL patterns | Zero config for users |
| Provider edge cases | User proxies through LiteLLM | No need for manual override env var |

---

## Design

### 1. HTTP Client Lifecycle

Create one `httpx.AsyncClient` in `main.py` lifespan. Pass it through the call chain. Close on shutdown.

**Lifespan changes:**

```python
# Startup
http_client = httpx.AsyncClient(timeout=60.0)

# Shutdown
await http_client.aclose()
```

**`chat_completion()` signature change:**

```python
async def chat_completion(
    client: httpx.AsyncClient,   # NEW — replaces internal client creation
    base_url: str,
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 800,
    api_key: str | None = None,
    provider: ProviderConfig | None = None,  # NEW — see Section 4
) -> str:
```

**Call chain:** `main.py` → `reconcile_query()` → `chat_completion()`. Both intermediate functions gain `client` parameter.

### 2. Global Concurrency Semaphore

**New env var:**

| Var | Default | Purpose |
|-----|---------|---------|
| `LLM_CONCURRENCY` | `4` | Max concurrent LLM requests |

Added to `config.py`:

```python
llm_concurrency: int = 4
```

**Semaphore created in lifespan**, stored alongside `http_client`:

```python
semaphore = asyncio.Semaphore(settings.llm_concurrency)
```

**Used inside `reconcile_query()`** — wraps the actual LLM call:

```python
async def reconcile_query(
    query: ReconciliationQuery,
    profile: ProfileConfig,
    base_url: str,
    api_key: str | None = None,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    provider: ProviderConfig | None = None,
) -> list[ReconciliationCandidate]:
    ...
    async with semaphore:
        raw = await chat_completion(client, base_url, messages, ...)
    ...
```

`asyncio.gather` in `_run_batch` still launches all coroutines — the semaphore gates HTTP calls. Up to `LLM_CONCURRENCY` queries execute simultaneously; the rest queue.

### 3. `json_schema` Response Format

**Canonical schema** — defined as a constant in `llm.py`:

```python
RECONCILIATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "reconciliation_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "matches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "score": {"type": "number"},
                            "description": {"type": "string"},
                        },
                        "required": ["name", "score"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["matches"],
            "additionalProperties": False,
        },
    },
}
```

**Only `name` and `score` required** from the LLM. `id` defaults to `gen_N`, `description` defaults to empty. Same defaults as current `_to_candidate()`.

**Response parsing when schema mode active:**

```python
data = json.loads(raw)
items = data["matches"]  # guaranteed by schema enforcement
```

No fallback logic needed. The 3-shape parser stays for `json_object` fallback mode (when provider doesn't support `json_schema`).

**`json_object` fallback schema hint:** When falling back to `json_object` mode, `reconcile.py` appends a format instruction as a separate system message (after the profile prompt, before the user message) so the LLM still targets the right shape:

```
Respond with JSON: {"matches": [{"name": "...", "score": N, "id": "...", "description": "..."}]}
```

This does not mutate the profile's prompt field. It's an additional message in the messages list. The existing 3-shape fallback parser handles any deviations.

### 4. Provider-Aware Parameter Handling

**`ProviderConfig` dataclass:**

```python
@dataclass(frozen=True)
class ProviderConfig:
    supports_json_schema: bool = True
    supports_seed: bool = True
    supports_response_format: bool = True
```

**`detect_provider(base_url: str) -> ProviderConfig`:**

Detection rules, checked in order:

| URL pattern | Provider | Config |
|---|---|---|
| `generativelanguage.googleapis.com` | Google AI | `supports_json_schema=False, supports_seed=False` |
| `huggingface` | HuggingFace | `supports_response_format=False` |
| `localhost`, `127.0.0.1`, `host.docker.internal` | Local engine | All defaults (full support) |
| Everything else | Generic hosted | All defaults (full support) |

**Called once at startup** in `main.py` lifespan. Result passed through call chain to `chat_completion()`.

**Request building in `chat_completion()`:**

```python
body = {
    "messages": messages,
    "temperature": temperature,
    "max_tokens": max_tokens,
}

if provider.supports_response_format:
    if provider.supports_json_schema:
        body["response_format"] = RECONCILIATION_SCHEMA
    else:
        body["response_format"] = {"type": "json_object"}
# else: omit response_format entirely
```

---

## Files Touched

| File | Changes |
|------|---------|
| `config.py` | Add `llm_concurrency: int = 4` |
| `llm.py` | Add `ProviderConfig`, `detect_provider()`, `RECONCILIATION_SCHEMA`. Change `chat_completion()` signature: add `client`, `provider` params, remove internal client creation, conditional request building |
| `reconcile.py` | Change `reconcile_query()` signature: add `client`, `semaphore`, `provider` params. Wrap LLM call in semaphore. Dual parse path (schema vs fallback) |
| `main.py` | Lifespan: create `http_client`, `semaphore`, call `detect_provider()`. Pass all three through `_run_batch` / `_run_single` → `reconcile_query()` |

No new modules. No new endpoints. No profile schema changes.

---

## Configuration Summary (after this change)

| Var | Default | Purpose |
|-----|---------|---------|
| `LLM_BASE_URL` | `http://host.docker.internal:8080/v1` | Inference engine endpoint |
| `LLM_API_KEY` | `None` | Optional Bearer token |
| `LLM_CONCURRENCY` | `4` | Max concurrent LLM requests |
| `PROFILES_DIR` | `profiles` | Profile YAML directory |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Testing

Existing smoke tests (health, profiles, manifest) continue passing unchanged — they don't call the LLM.

New unit tests for the additions:

| Test | Covers |
|------|--------|
| `test_detect_provider()` | URL pattern → ProviderConfig mapping for each known pattern |
| `test_reconciliation_schema_parse()` | Parse response conforming to canonical schema |
| `test_fallback_parse()` | Parse all 3 legacy shapes when schema mode off |
| `test_semaphore_limits_concurrency()` | Mock LLM with delay, verify max N concurrent calls |

All tests mock `chat_completion` or `httpx.AsyncClient` — no live inference engine needed.

---

## What This Does NOT Change

- Profile YAML schema (no new fields)
- W3C API surface (same endpoints, same response format)
- External behavior from OpenRefine's perspective
- Caching logic
- DSPy stub
