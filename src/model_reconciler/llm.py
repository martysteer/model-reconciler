"""Async HTTP client for OpenAI-compatible chat completions."""

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


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
