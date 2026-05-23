"""Reconciliation logic — stateless functions, no framework coupling."""

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
            base_url=base_url,
            messages=messages,
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
        )
    except Exception:
        logger.exception(f"LLM call failed for profile '{profile.slug}'")
        return []

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


def _to_candidate(m: dict, index: int) -> ReconciliationCandidate:
    """Map a raw dict from the LLM into a typed ReconciliationCandidate."""
    score = float(m.get("score", 50))
    return ReconciliationCandidate(
        id=m.get("id", f"gen_{index}"),
        name=m.get("name", ""),
        score=score,
        match=score >= 90,
        description=m.get("description") or m.get("reasoning", ""),
    )
