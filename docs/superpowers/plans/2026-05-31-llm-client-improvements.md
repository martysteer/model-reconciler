# LLM Client Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve model-reconciler's LLM client with HTTP connection reuse, concurrency control, json_schema response format, and provider-aware parameter handling.

**Architecture:** All changes are internal to 4 existing files (`config.py`, `llm.py`, `reconcile.py`, `main.py`). No new modules, no new endpoints. The HTTP client, semaphore, and provider config are created in lifespan and passed through the call chain.

**Tech Stack:** Python 3.12, httpx, asyncio, FastAPI, pytest, pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-05-31-llm-client-improvements-design.md`

---

## Approach

Build in 4 vertical slices. Each task delivers a working, testable increment. Existing tests must pass after every task.

1. Provider detection + schema constant (pure functions, easy to test in isolation)
2. HTTP client lifecycle (lifespan change, signature threading)
3. Concurrency semaphore (config + lifespan + reconcile.py)
4. json_schema response format + conditional request building (ties it all together)

---

## File Map

```
src/model_reconciler/
├── config.py        # Add llm_concurrency
├── llm.py           # Add ProviderConfig, detect_provider(), RECONCILIATION_SCHEMA
│                    #   Rewrite chat_completion() signature + body building
├── reconcile.py     # New signature, semaphore wrap, dual parse path, schema hint
└── main.py          # Lifespan: http_client, semaphore, provider. Thread through calls.

tests/
├── test_llm.py      # NEW — provider detection, schema parse, semaphore
└── (existing tests unchanged)
```

---

### Task 1: Provider Detection + Schema Constant

**Goal:** Add `ProviderConfig` dataclass, `detect_provider()` function, and `RECONCILIATION_SCHEMA` constant to `llm.py`. Test in isolation.

**Files:**
- Modify: `src/model_reconciler/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write tests for provider detection**

Create `tests/test_llm.py`:

```python
"""Tests for LLM client utilities."""

from model_reconciler.llm import ProviderConfig, detect_provider


def test_detect_local_localhost():
    p = detect_provider("http://localhost:8080/v1")
    assert p.supports_json_schema is True
    assert p.supports_seed is True
    assert p.supports_response_format is True


def test_detect_local_docker_internal():
    p = detect_provider("http://host.docker.internal:8080/v1")
    assert p.supports_json_schema is True
    assert p.supports_seed is True
    assert p.supports_response_format is True


def test_detect_local_127():
    p = detect_provider("http://127.0.0.1:8080/v1")
    assert p.supports_json_schema is True


def test_detect_google_ai():
    p = detect_provider("https://generativelanguage.googleapis.com/v1beta/openai")
    assert p.supports_json_schema is False
    assert p.supports_seed is False
    assert p.supports_response_format is True


def test_detect_huggingface():
    p = detect_provider("https://api-inference.huggingface.co/models/meta-llama/Llama-3")
    assert p.supports_response_format is False


def test_detect_generic_hosted():
    p = detect_provider("https://api.openrouter.ai/v1")
    assert p.supports_json_schema is True
    assert p.supports_seed is True
    assert p.supports_response_format is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_llm.py -v`
Expected: FAIL — `ImportError: cannot import name 'ProviderConfig'`

- [ ] **Step 3: Add ProviderConfig and detect_provider to llm.py**

Add these at the top of `src/model_reconciler/llm.py`, after the existing imports:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderConfig:
    """Capabilities of the target inference engine."""

    supports_json_schema: bool = True
    supports_seed: bool = True
    supports_response_format: bool = True


def detect_provider(base_url: str) -> ProviderConfig:
    """Auto-detect provider capabilities from URL patterns."""
    url = base_url.lower()

    if "generativelanguage.googleapis.com" in url:
        return ProviderConfig(supports_json_schema=False, supports_seed=False)

    if "huggingface" in url:
        return ProviderConfig(supports_response_format=False)

    # Local engines and generic hosted: full support
    return ProviderConfig()
```

- [ ] **Step 4: Add RECONCILIATION_SCHEMA constant to llm.py**

Add after `detect_provider`:

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
                            "id": {"type": ["string", "null"]},
                            "name": {"type": "string"},
                            "score": {"type": "number"},
                            "description": {"type": ["string", "null"]},
                        },
                        "required": ["id", "name", "score", "description"],
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

- [ ] **Step 5: Run provider detection tests**

Run: `PYTHONPATH=src pytest tests/test_llm.py -v`
Expected: 6 tests PASS

- [ ] **Step 6: Run all existing tests to verify nothing broke**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: All existing tests PASS (10 + 6 new = 16 total)

- [ ] **Step 7: Commit**

```bash
git add src/model_reconciler/llm.py tests/test_llm.py
git commit -m "feat: add provider detection and reconciliation schema constant"
```

---

### Task 2: HTTP Client Lifecycle

**Goal:** Create `httpx.AsyncClient` in lifespan, pass through call chain, close on shutdown. Remove per-request client creation from `chat_completion()`.

**Files:**
- Modify: `src/model_reconciler/llm.py`
- Modify: `src/model_reconciler/reconcile.py`
- Modify: `src/model_reconciler/main.py`

- [ ] **Step 1: Rewrite `chat_completion()` to accept a client parameter**

Replace the entire `chat_completion` function in `src/model_reconciler/llm.py` with:

```python
async def chat_completion(
    client: httpx.AsyncClient,
    base_url: str,
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 800,
    api_key: str | None = None,
) -> str:
    """POST to /chat/completions with JSON mode. Return content string.

    Args:
        client: Shared async HTTP client.
        base_url: e.g. http://localhost:8080/v1
        messages: [{"role": "system", "content": "..."}, ...]
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate.
        api_key: Optional API key for authentication.

    Returns:
        Raw content string from the model's response.

    Raises:
        httpx.HTTPStatusError: On non-2xx response.
        httpx.ConnectError: If the inference engine is unreachable.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    r = await client.post(
        url,
        headers=headers,
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

- [ ] **Step 2: Update `reconcile_query()` in `reconcile.py` to accept client**

Replace the function signature and LLM call section in `src/model_reconciler/reconcile.py`:

```python
import httpx

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
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
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
            client=client,
            base_url=base_url,
            messages=messages,
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            api_key=api_key,
        )
    except Exception:
        logger.exception(f"LLM call failed for profile '{profile.slug}'")
        return []

    return parse_llm_response(raw)[: query.limit]
```

- [ ] **Step 3: Update `main.py` lifespan to create and close the HTTP client**

Add `import httpx` to main.py imports, then replace the lifespan function:

```python
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal http_client
        http_client = httpx.AsyncClient(timeout=60.0)

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
        await http_client.aclose()
```

Add `http_client: httpx.AsyncClient | None = None` declaration before the lifespan, alongside `registry`:

```python
    registry: dict[str, tuple[ProfileConfig, TTLCache]] = {}
    http_client: httpx.AsyncClient | None = None
```

- [ ] **Step 4: Thread `http_client` through `_run_batch` and `_run_single`**

Update `_run_batch` coroutine list:

```python
        if uncached:
            coros = [
                reconcile_query(q, profile, base_url, settings.llm_api_key, client=http_client)
                for q, _ in uncached.values()
            ]
            completed = await asyncio.gather(*coros)
```

Update `_run_single`:

```python
        candidates = await reconcile_query(q, profile, base_url, settings.llm_api_key, client=http_client)
```

- [ ] **Step 5: Run all tests**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: All 16 tests PASS. (Existing smoke tests use TestClient which triggers lifespan.)

- [ ] **Step 6: Commit**

```bash
git add src/model_reconciler/llm.py src/model_reconciler/reconcile.py src/model_reconciler/main.py
git commit -m "feat: reuse HTTP client across requests via lifespan"
```

---

### Task 3: Concurrency Semaphore

**Goal:** Add `LLM_CONCURRENCY` config, create semaphore in lifespan, use it in `reconcile_query()`.

**Files:**
- Modify: `src/model_reconciler/config.py`
- Modify: `src/model_reconciler/reconcile.py`
- Modify: `src/model_reconciler/main.py`
- Modify: `tests/test_llm.py`

- [ ] **Step 1: Write semaphore concurrency test**

Add to `tests/test_llm.py`:

```python
import asyncio
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency():
    """Verify semaphore caps concurrent LLM calls."""
    call_count = 0
    max_concurrent = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count, max_concurrent
        call_count += 1
        max_concurrent = max(max_concurrent, call_count)
        await asyncio.sleep(0.05)
        call_count -= 1
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = lambda: None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"matches": []}'}}]
        }
        return mock_response

    from model_reconciler.llm import ProviderConfig
    from model_reconciler.models import ProfileConfig, ReconciliationQuery
    from model_reconciler.reconcile import reconcile_query

    profile = ProfileConfig(
        name="Test", prompt="Test prompt", types=[{"id": "t", "name": "T"}], slug="test"
    )
    semaphore = asyncio.Semaphore(2)
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = mock_post
    provider = ProviderConfig()

    queries = [
        ReconciliationQuery(query=f"q{i}") for i in range(6)
    ]

    coros = [
        reconcile_query(
            q, profile, "http://localhost:8080/v1",
            api_key=None, client=client, semaphore=semaphore, provider=provider,
        )
        for q in queries
    ]
    await asyncio.gather(*coros)

    assert max_concurrent <= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_llm.py::test_semaphore_limits_concurrency -v`
Expected: FAIL — `reconcile_query() got an unexpected keyword argument 'semaphore'`

- [ ] **Step 3: Add `llm_concurrency` to config.py**

Replace `src/model_reconciler/config.py`:

```python
"""Global configuration — environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_base_url: str = "http://host.docker.internal:8080/v1"
    llm_api_key: str | None = None
    llm_concurrency: int = 4
    profiles_dir: str = "profiles"
    log_level: str = "INFO"

    model_config = {"env_prefix": "", "case_sensitive": False}
```

- [ ] **Step 4: Update `reconcile_query()` to accept and use semaphore**

Update signature and body in `src/model_reconciler/reconcile.py`:

```python
import asyncio

import httpx

from model_reconciler.llm import ProviderConfig, chat_completion
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
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
    semaphore: asyncio.Semaphore | None = None,
    provider: ProviderConfig | None = None,
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
        if semaphore:
            async with semaphore:
                raw = await chat_completion(
                    client=client,
                    base_url=base_url,
                    messages=messages,
                    temperature=profile.temperature,
                    max_tokens=profile.max_tokens,
                    api_key=api_key,
                )
        else:
            raw = await chat_completion(
                client=client,
                base_url=base_url,
                messages=messages,
                temperature=profile.temperature,
                max_tokens=profile.max_tokens,
                api_key=api_key,
            )
    except Exception:
        logger.exception(f"LLM call failed for profile '{profile.slug}'")
        return []

    return parse_llm_response(raw)[: query.limit]
```

- [ ] **Step 5: Update main.py to create semaphore and pass it through**

Add `semaphore` creation in lifespan (after `http_client`):

```python
    http_client: httpx.AsyncClient | None = None
    semaphore: asyncio.Semaphore | None = None
```

In lifespan startup, after `http_client = httpx.AsyncClient(timeout=60.0)`:

```python
        semaphore = asyncio.Semaphore(settings.llm_concurrency)
```

Update `_run_batch` coroutine list:

```python
        if uncached:
            coros = [
                reconcile_query(
                    q, profile, base_url, settings.llm_api_key,
                    client=http_client, semaphore=semaphore,
                )
                for q, _ in uncached.values()
            ]
            completed = await asyncio.gather(*coros)
```

Update `_run_single`:

```python
        candidates = await reconcile_query(
            q, profile, base_url, settings.llm_api_key,
            client=http_client, semaphore=semaphore,
        )
```

- [ ] **Step 6: Run semaphore test**

Run: `PYTHONPATH=src pytest tests/test_llm.py::test_semaphore_limits_concurrency -v`
Expected: PASS

- [ ] **Step 7: Run all tests**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: All 17 tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/model_reconciler/config.py src/model_reconciler/reconcile.py src/model_reconciler/main.py tests/test_llm.py
git commit -m "feat: add concurrency semaphore for LLM requests"
```

---

### Task 4: json_schema Response Format + Provider-Aware Request Building

**Goal:** `chat_completion()` uses `json_schema` when provider supports it, `json_object` otherwise, or omits `response_format` entirely for HuggingFace. `reconcile.py` adds schema hint message for `json_object` fallback and uses direct parse path for schema mode.

**Files:**
- Modify: `src/model_reconciler/llm.py`
- Modify: `src/model_reconciler/reconcile.py`
- Modify: `src/model_reconciler/main.py`
- Modify: `tests/test_llm.py`

- [ ] **Step 1: Write test for schema-mode response parsing**

Add to `tests/test_llm.py`:

```python
from model_reconciler.reconcile import parse_llm_response, parse_schema_response


def test_parse_schema_response_valid():
    """Parse response conforming to canonical json_schema."""
    raw = '{"matches": [{"id": "abc", "name": "Shakespeare", "score": 95, "description": "Exact match"}]}'
    candidates = parse_schema_response(raw)
    assert len(candidates) == 1
    assert candidates[0].name == "Shakespeare"
    assert candidates[0].score == 95
    assert candidates[0].id == "abc"
    assert candidates[0].match is True


def test_parse_schema_response_nullables():
    """Null id and description get defaults."""
    raw = '{"matches": [{"id": null, "name": "Test", "score": 70, "description": null}]}'
    candidates = parse_schema_response(raw)
    assert len(candidates) == 1
    assert candidates[0].id == "gen_0"
    assert candidates[0].description == ""
    assert candidates[0].match is False


def test_parse_schema_response_empty_matches():
    """Empty matches array returns empty list."""
    raw = '{"matches": []}'
    candidates = parse_schema_response(raw)
    assert candidates == []


def test_parse_schema_response_invalid_json():
    """Invalid JSON returns empty list."""
    candidates = parse_schema_response("not json at all")
    assert candidates == []


def test_fallback_parse_direct_array():
    """Existing fallback parser handles direct array."""
    raw = '[{"name": "Test", "score": 80}]'
    candidates = parse_llm_response(raw)
    assert len(candidates) == 1
    assert candidates[0].name == "Test"


def test_fallback_parse_wrapped_object():
    """Existing fallback parser handles wrapped object."""
    raw = '{"results": [{"name": "Test", "score": 60}]}'
    candidates = parse_llm_response(raw)
    assert len(candidates) == 1


def test_fallback_parse_markdown_fences():
    """Existing fallback parser handles markdown fences."""
    raw = '```json\n[{"name": "Test", "score": 55}]\n```'
    candidates = parse_llm_response(raw)
    assert len(candidates) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_llm.py::test_parse_schema_response_valid -v`
Expected: FAIL — `ImportError: cannot import name 'parse_schema_response'`

- [ ] **Step 3: Add `parse_schema_response()` to `reconcile.py`**

Add after the existing `parse_llm_response` function in `src/model_reconciler/reconcile.py`:

```python
def parse_schema_response(text: str) -> list[ReconciliationCandidate]:
    """Parse LLM response that conforms to RECONCILIATION_SCHEMA.

    Expects: {"matches": [{"id": ..., "name": ..., "score": ..., "description": ...}]}
    Handles null values for id and description.
    Returns empty list on parse failure.
    """
    try:
        data = json.loads(text.strip())
        items = data["matches"]
        return [_to_candidate(m, i) for i, m in enumerate(items)]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.warning(f"Failed to parse schema response: {e}")
        return []
```

- [ ] **Step 4: Run schema parse tests**

Run: `PYTHONPATH=src pytest tests/test_llm.py -k "parse" -v`
Expected: All 7 parse tests PASS

- [ ] **Step 5: Update `chat_completion()` to use provider-aware request building**

Replace `chat_completion` in `src/model_reconciler/llm.py`:

```python
async def chat_completion(
    client: httpx.AsyncClient,
    base_url: str,
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 800,
    api_key: str | None = None,
    provider: ProviderConfig | None = None,
) -> str:
    """POST to /chat/completions. Return content string.

    Args:
        client: Shared async HTTP client.
        base_url: e.g. http://localhost:8080/v1
        messages: [{"role": "system", "content": "..."}, ...]
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate.
        api_key: Optional API key for authentication.
        provider: Provider capabilities (controls response_format).

    Returns:
        Raw content string from the model's response.

    Raises:
        httpx.HTTPStatusError: On non-2xx response.
        httpx.ConnectError: If the inference engine is unreachable.
    """
    if provider is None:
        provider = ProviderConfig()

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body: dict = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if provider.supports_response_format:
        if provider.supports_json_schema:
            body["response_format"] = RECONCILIATION_SCHEMA
        else:
            body["response_format"] = {"type": "json_object"}

    r = await client.post(url, headers=headers, json=body)
    r.raise_for_status()

    return r.json()["choices"][0]["message"]["content"]
```

- [ ] **Step 6: Update `reconcile_query()` to use dual parse path and schema hint**

Replace `reconcile_query` in `src/model_reconciler/reconcile.py`:

```python
SCHEMA_HINT_MESSAGE = {
    "role": "system",
    "content": 'Respond with JSON: {"matches": [{"name": "...", "score": N, "id": "...", "description": "..."}]}',
}


async def reconcile_query(
    query: ReconciliationQuery,
    profile: ProfileConfig,
    base_url: str,
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
    semaphore: asyncio.Semaphore | None = None,
    provider: ProviderConfig | None = None,
) -> list[ReconciliationCandidate]:
    """Build prompt, call LLM, parse JSON, return candidates."""
    if profile.use_dspy:
        raise NotImplementedError(
            f"Profile '{profile.slug}' has use_dspy=true but DSPy "
            "support is not yet implemented. Set use_dspy: false."
        )

    if provider is None:
        provider = ProviderConfig()

    use_schema = provider.supports_response_format and provider.supports_json_schema

    messages = [
        {"role": "system", "content": profile.prompt},
    ]

    # Add schema hint for json_object fallback mode
    if not use_schema and provider.supports_response_format:
        messages.append(SCHEMA_HINT_MESSAGE)

    messages.append({"role": "user", "content": _format_user_message(query)})

    try:
        if semaphore:
            async with semaphore:
                raw = await chat_completion(
                    client=client,
                    base_url=base_url,
                    messages=messages,
                    temperature=profile.temperature,
                    max_tokens=profile.max_tokens,
                    api_key=api_key,
                    provider=provider,
                )
        else:
            raw = await chat_completion(
                client=client,
                base_url=base_url,
                messages=messages,
                temperature=profile.temperature,
                max_tokens=profile.max_tokens,
                api_key=api_key,
                provider=provider,
            )
    except Exception:
        logger.exception(f"LLM call failed for profile '{profile.slug}'")
        return []

    if use_schema:
        return parse_schema_response(raw)[: query.limit]
    return parse_llm_response(raw)[: query.limit]
```

- [ ] **Step 7: Update main.py to detect provider and pass it through**

Add import at top of `main.py`:

```python
from model_reconciler.llm import detect_provider
```

In lifespan startup, after semaphore creation:

```python
        provider = detect_provider(settings.llm_base_url)
        logger.info(f"Provider detected: json_schema={provider.supports_json_schema}, "
                    f"response_format={provider.supports_response_format}")
```

Add `provider` declaration alongside other closure vars:

```python
    registry: dict[str, tuple[ProfileConfig, TTLCache]] = {}
    http_client: httpx.AsyncClient | None = None
    semaphore: asyncio.Semaphore | None = None
    provider = None
```

Update `_run_batch` coroutine list:

```python
        if uncached:
            coros = [
                reconcile_query(
                    q, profile, base_url, settings.llm_api_key,
                    client=http_client, semaphore=semaphore, provider=provider,
                )
                for q, _ in uncached.values()
            ]
            completed = await asyncio.gather(*coros)
```

Update `_run_single`:

```python
        candidates = await reconcile_query(
            q, profile, base_url, settings.llm_api_key,
            client=http_client, semaphore=semaphore, provider=provider,
        )
```

- [ ] **Step 8: Run all tests**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: All 24 tests PASS (10 existing + 6 provider + 1 semaphore + 7 parse)

- [ ] **Step 9: Commit**

```bash
git add src/model_reconciler/llm.py src/model_reconciler/reconcile.py src/model_reconciler/main.py tests/test_llm.py
git commit -m "feat: json_schema response format with provider-aware request building"
```

---

## Verification Checklist

After all tasks:

- [ ] `PYTHONPATH=src pytest tests/ -v` — 24 tests pass
- [ ] `PYTHONPATH=src ruff check src/ tests/` — clean
- [ ] `docker compose build` — image builds
- [ ] `docker compose up -d && sleep 3 && curl -sf http://localhost:8001/health && docker compose down` — health check passes
- [ ] Existing smoke tests (health, profiles, manifest) unchanged and passing
- [ ] No new env vars required — `LLM_CONCURRENCY` has sensible default of 4
