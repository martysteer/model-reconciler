"""Reconciliation logic — stateless functions, no framework coupling."""

import asyncio
import json
import logging

import httpx

from model_reconciler.llm import ProviderConfig, chat_completion
from model_reconciler.models import (
    ProfileConfig,
    ReconciliationCandidate,
    ReconciliationQuery,
)

logger = logging.getLogger(__name__)

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


def _to_candidate(m: dict, index: int) -> ReconciliationCandidate:
    """Map a raw dict from the LLM into a typed ReconciliationCandidate."""
    score = float(m.get("score", 50))
    return ReconciliationCandidate(
        id=m.get("id") or f"gen_{index}",
        name=m.get("name", ""),
        score=score,
        match=score >= 90,
        description=m.get("description") or m.get("reasoning") or "",
    )
