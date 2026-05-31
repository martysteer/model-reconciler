"""Tests for LLM client utilities."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

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


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency():
    """Verify semaphore caps concurrent LLM calls."""
    active = 0
    max_concurrent = 0

    async def mock_post(*args, **kwargs):
        nonlocal active, max_concurrent
        active += 1
        max_concurrent = max(max_concurrent, active)
        await asyncio.sleep(0.05)
        active -= 1
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = lambda: None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"matches": []}'}}]
        }
        return mock_response

    from model_reconciler.models import ProfileConfig, ReconciliationQuery
    from model_reconciler.reconcile import reconcile_query

    profile = ProfileConfig(
        name="Test", prompt="Test prompt", types=[{"id": "t", "name": "T"}], slug="test"
    )
    semaphore = asyncio.Semaphore(2)
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = mock_post

    queries = [ReconciliationQuery(query=f"q{i}") for i in range(6)]

    coros = [
        reconcile_query(
            q, profile, "http://localhost:8080/v1",
            api_key=None, client=client, semaphore=semaphore,
        )
        for q in queries
    ]
    await asyncio.gather(*coros)

    assert max_concurrent <= 2
