"""Async HTTP client for OpenAI-compatible chat completions."""

import logging

import httpx

logger = logging.getLogger(__name__)


async def chat_completion(
    base_url: str,
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 800,
    api_key: str | None = None,
) -> str:
    """POST to /chat/completions with JSON mode. Return content string.

    Args:
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

    async with httpx.AsyncClient(timeout=60.0) as client:
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
